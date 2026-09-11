"""
Tests for the analyze/health/metrics/recent endpoints (LEX-1).

The real analyzer pipeline needs sentence-transformers/torch loaded, which is
slow (multi-second) and, on at least one known environment, deadlocks on
import entirely (see analyzer/inference.py's module docstring). Unit tests
for a *view* have no business depending on that: they mock `get_analyzer()`
and `ModelCache` instead of touching the real model. That's not a compromise
made to work around a broken environment — it's just correct test design;
a view test should verify request handling and response shape, not
re-verify that the ML model works (that's what
api/measure_confidence_impact.py and manual verification against a real
model are for).
"""

from unittest.mock import patch, MagicMock

import numpy as np
from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from .inference import LegalAnalyzer
from .models import AnalysisResult


def fake_analysis(risky_count=1, total_clauses=3):
    """A representative return value from LegalAnalyzer.analyze_clauses,
    shaped exactly like the real thing (see analyzer/inference.py) so the
    view code that reads specific keys off it is exercised faithfully."""
    return {
        'risky_clauses': [
            {
                'clause_id': 0,
                'text': 'This clause limits liability in unforeseen circumstances.',
                'risk': 'risky',
                'confidence': 0.91,
                'safe_prob': 0.09,
                'risky_prob': 0.91,
            }
        ][:risky_count],
        'all_clauses': [],
        'processing_time_ms': 42.5,
        'total_clauses': total_clauses,
        'risky_count': risky_count,
    }


class AnalyzeEndpointTests(APITestCase):
    def setUp(self):
        self.url = reverse('document-analyze')

    def test_rejects_empty_text(self):
        response = self.client.post(self.url, {'text': '', 'filename': 'a.txt'}, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.data)

    def test_rejects_too_short_text(self):
        response = self.client.post(self.url, {'text': 'short', 'filename': 'a.txt'}, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('10 characters', response.data['error'])

    def test_rejects_missing_text_field(self):
        response = self.client.post(self.url, {'filename': 'a.txt'}, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_malformed_json_body(self):
        # Current behavior: the view's own try/except catches DRF's
        # ParseError (raised lazily when request.data is first accessed)
        # and reports it as a 500, not a 400 — arguably a client error
        # should be a 400, not a server error. Documented here rather than
        # silently treated as correct; worth a follow-up if it matters for
        # API consumers distinguishing "your fault" from "our fault".
        response = self.client.post(
            self.url, data='{not valid json', content_type='application/json'
        )
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)

    @patch('analyzer.views.get_analyzer')
    def test_rejects_when_no_clauses_extracted(self, mock_get_analyzer):
        mock_analyzer = MagicMock()
        mock_analyzer.extract_clauses.return_value = []
        mock_get_analyzer.return_value = mock_analyzer

        response = self.client.post(
            self.url, {'text': 'a document with only whitespace  ', 'filename': 'a.txt'}, format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('No clauses', response.data['error'])

    @patch('analyzer.views.get_analyzer')
    def test_successful_analysis_returns_expected_shape_and_persists(self, mock_get_analyzer):
        mock_analyzer = MagicMock()
        mock_analyzer.extract_clauses.return_value = ['Clause one.', 'Clause two.', 'Clause three.']
        mock_analyzer.analyze_clauses.return_value = fake_analysis(risky_count=1, total_clauses=3)
        mock_get_analyzer.return_value = mock_analyzer

        response = self.client.post(
            self.url,
            {'text': 'Clause one. Clause two. Clause three.', 'filename': 'contract.pdf'},
            format='json',
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['filename'], 'contract.pdf')
        self.assertEqual(response.data['total_clauses'], 3)
        self.assertEqual(response.data['risky_count'], 1)
        self.assertEqual(len(response.data['risky_clauses']), 1)
        self.assertIn('document_id', response.data)
        self.assertIn('processing_time_ms', response.data)
        self.assertIn('api_response_time_ms', response.data)

        # Side effect that matters as much as the response: a record must
        # actually be persisted, since /recent and future history features
        # depend on it existing.
        self.assertEqual(AnalysisResult.objects.count(), 1)
        saved = AnalysisResult.objects.first()
        self.assertEqual(saved.document_id, response.data['document_id'])
        self.assertEqual(saved.filename, 'contract.pdf')
        self.assertEqual(saved.model_used, 'sbert')

    @patch('analyzer.views.get_analyzer')
    def test_risky_clauses_truncated_to_top_ten(self, mock_get_analyzer):
        many_risky = [
            {'clause_id': i, 'text': f'clause {i}', 'risk': 'risky', 'confidence': 0.8, 'safe_prob': 0.2, 'risky_prob': 0.8}
            for i in range(15)
        ]
        analysis = fake_analysis()
        analysis['risky_clauses'] = many_risky
        analysis['risky_count'] = 15

        mock_analyzer = MagicMock()
        mock_analyzer.extract_clauses.return_value = ['x'] * 15
        mock_analyzer.analyze_clauses.return_value = analysis
        mock_get_analyzer.return_value = mock_analyzer

        response = self.client.post(
            self.url, {'text': 'a' * 20, 'filename': 'big.pdf'}, format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        # response.risky_clauses is capped even though the full analysis had 15
        self.assertEqual(len(response.data['risky_clauses']), 10)
        self.assertEqual(response.data['risky_count'], 15)

    @patch('analyzer.views.get_analyzer')
    def test_internal_error_during_analysis_returns_500_not_a_crash(self, mock_get_analyzer):
        mock_analyzer = MagicMock()
        mock_analyzer.extract_clauses.return_value = ['a clause']
        mock_analyzer.analyze_clauses.side_effect = RuntimeError('model exploded')
        mock_get_analyzer.return_value = mock_analyzer

        response = self.client.post(
            self.url, {'text': 'a' * 20, 'filename': 'a.txt'}, format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn('model exploded', response.data['error'])
        # A failed analysis must not leave a half-written record behind.
        self.assertEqual(AnalysisResult.objects.count(), 0)


class RecentEndpointTests(APITestCase):
    def setUp(self):
        self.url = reverse('document-recent')

    def test_empty_when_no_analyses_exist(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data, {'count': 0, 'results': []})

    def test_returns_created_results(self):
        AnalysisResult.objects.create(
            document_id='doc-1', filename='one.pdf', clauses=['a'],
            analysis={'risky_count': 2}, processing_time_ms=10.0, model_used='sbert',
        )
        AnalysisResult.objects.create(
            document_id='doc-2', filename='two.pdf', clauses=['b'],
            analysis={'risky_count': 0}, processing_time_ms=12.0, model_used='sbert',
        )

        response = self.client.get(self.url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['count'], 2)
        filenames = {r['filename'] for r in response.data['results']}
        self.assertEqual(filenames, {'one.pdf', 'two.pdf'})

    def test_respects_limit_query_param(self):
        for i in range(5):
            AnalysisResult.objects.create(
                document_id=f'doc-{i}', filename=f'{i}.pdf', clauses=['a'],
                analysis={'risky_count': 0}, processing_time_ms=1.0, model_used='sbert',
            )

        response = self.client.get(self.url, {'limit': 2})
        self.assertEqual(response.data['count'], 2)


class HealthEndpointTests(APITestCase):
    def setUp(self):
        self.url = reverse('health')

    @patch('analyzer.inference.ModelCache.is_loaded', return_value=False)
    def test_reports_degraded_when_models_not_loaded(self, _mock):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertEqual(response.data['status'], 'degraded')
        self.assertFalse(response.data['models_loaded'])

    @patch('analyzer.inference.ModelCache.is_loaded', return_value=True)
    def test_reports_healthy_when_models_loaded(self, _mock):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], 'healthy')
        self.assertTrue(response.data['models_loaded'])

    @patch('analyzer.inference.ModelCache.is_loaded')
    def test_never_constructs_modelcache_just_to_check_health(self, mock_is_loaded):
        """Regression test for the bug fixed in this same session: /health
        must call the classmethod check, never `ModelCache()`, because
        instantiating it triggers a real model load as a side effect of a
        status check."""
        mock_is_loaded.return_value = False
        self.client.get(self.url)
        mock_is_loaded.assert_called_once()

    def test_unexpected_error_reports_unhealthy_not_a_crash(self):
        with patch('analyzer.inference.ModelCache.is_loaded', side_effect=RuntimeError('boom')):
            response = self.client.get(self.url)
        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertEqual(response.data['status'], 'unhealthy')
        self.assertIn('boom', response.data['error'])


class MetricsEndpointTests(APITestCase):
    def setUp(self):
        self.url = reverse('metrics')

    @patch('analyzer.inference.ModelCache.is_loaded', return_value=False)
    def test_zero_metrics_when_models_not_loaded(self, _mock):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['total_requests'], 0)
        self.assertIn('not loaded', response.data['message'])

    @patch('analyzer.views.get_analyzer')
    @patch('analyzer.inference.ModelCache.is_loaded', return_value=True)
    def test_returns_real_metrics_when_models_loaded(self, _mock_loaded, mock_get_analyzer):
        mock_analyzer = MagicMock()
        mock_analyzer.get_metrics.return_value = {
            'total_requests': 42,
            'avg_latency_ms': 147.3,
            'max_latency_ms': 310.1,
            'p50_latency_ms': 'See monitoring dashboard',
        }
        mock_get_analyzer.return_value = mock_analyzer

        response = self.client.get(self.url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['total_requests'], 42)
        self.assertIn('147.3ms', response.data['message'])
        self.assertIn('42 requests', response.data['message'])


class ProbabilityMappingRegressionTests(TestCase):
    """Regression test for a real bug found during LEX-11 manual Docker
    verification: analyze_clauses() hardcoded probs[0]=safe_prob,
    probs[1]=risky_prob, but scikit-learn orders predict_proba's columns by
    clf.classes_ (alphabetical for string labels — 'risky' sorts before
    'safe'), so the two fields were silently swapped. A clause correctly
    labeled "risky" shipped with a 99%+ "safe_prob" next to it. Fixed by
    reading classes_ directly instead of assuming a fixed order.
    """

    def _analyzer_with_fake_classifier(self, classes):
        """Builds a LegalAnalyzer bypassing __init__, which would otherwise
        construct a real ModelCache and attempt to import
        sentence-transformers/torch — exactly what this test needs to avoid
        depending on, so it can run anywhere, including this sandbox."""
        analyzer = LegalAnalyzer.__new__(LegalAnalyzer)
        analyzer.metrics = {'total_requests': 0, 'total_time_ms': 0, 'avg_latency_ms': 0, 'max_latency_ms': 0}

        fake_model = MagicMock()
        fake_model.encode.return_value = np.zeros((1, 4))

        fake_clf = MagicMock()
        fake_clf.classes_ = classes
        fake_clf.predict.return_value = ['risky']
        # Probabilities placed to MATCH `classes`, whatever order it's in —
        # the fix under test must read classes_, not assume a layout.
        risky_col = classes.index('risky')
        probs_row = [0.0, 0.0]
        probs_row[risky_col] = 0.97
        probs_row[1 - risky_col] = 0.03
        fake_clf.predict_proba.return_value = np.array([probs_row])

        fake_cache = MagicMock()
        fake_cache.get_sbert_model.return_value = fake_model
        fake_cache.get_sbert_classifier.return_value = fake_clf
        analyzer.model_cache = fake_cache
        return analyzer

    def test_probabilities_map_correctly_when_risky_is_class_zero(self):
        # This is the real trained classifier's actual order today
        # (confirmed via clf.classes_ during LEX-11) — the exact ordering
        # that triggered the original bug.
        analyzer = self._analyzer_with_fake_classifier(['risky', 'safe'])
        clause = analyzer.analyze_clauses(['a clause'])['all_clauses'][0]

        self.assertEqual(clause['risk'], 'risky')
        self.assertAlmostEqual(clause['risky_prob'], 0.97)
        self.assertAlmostEqual(clause['safe_prob'], 0.03)
        # The bug's exact symptom was internal inconsistency, not a missing
        # field — assert the numbers actually agree with the label.
        self.assertGreater(clause['risky_prob'], clause['safe_prob'])

    def test_probabilities_map_correctly_when_safe_is_class_zero(self):
        # The other possible ordering, proving the fix reads classes_
        # rather than special-casing the one order seen above.
        analyzer = self._analyzer_with_fake_classifier(['safe', 'risky'])
        clause = analyzer.analyze_clauses(['a clause'])['all_clauses'][0]

        self.assertEqual(clause['risk'], 'risky')
        self.assertAlmostEqual(clause['risky_prob'], 0.97)
        self.assertAlmostEqual(clause['safe_prob'], 0.03)
