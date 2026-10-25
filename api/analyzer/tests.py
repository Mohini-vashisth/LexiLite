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

import io
from unittest.mock import patch, MagicMock

import docx
import numpy as np
import pymupdf
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.test import APITestCase

from .document_parsers import (
    extract_text_from_pdf_bytes,
    extract_text_from_docx_bytes,
    extract_text_from_upload,
    UnsupportedFileType,
    DocumentParseError,
)
from .inference import LegalAnalyzer
from .models import AnalysisResult


def make_pdf_bytes(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def make_docx_bytes(text: str) -> bytes:
    document = docx.Document()
    document.add_paragraph(text)
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


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
        self.user = User.objects.create_user(username='alice', password='pw')
        self.client.force_authenticate(user=self.user)

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
        # request.data is accessed in `analyze` itself (to pull out text/
        # filename) BEFORE _run_analysis's try/except starts, so a parse
        # failure here propagates to DRF's own exception handler rather
        # than being swallowed into a 500 — DRF correctly reports a
        # ParseError as 400, which is the right status for a client's
        # malformed input. (An earlier version of this code wrapped the
        # request.data access inside the same try/except as everything
        # else, which incorrectly reported this exact case as a 500 — the
        # LEX-3 refactor fixed this as a side effect, not the goal.)
        response = self.client.post(
            self.url, data='{not valid json', content_type='application/json'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

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
    def test_successful_analysis_returns_expected_shape_and_persists_when_saved(self, mock_get_analyzer):
        mock_analyzer = MagicMock()
        mock_analyzer.extract_clauses.return_value = ['Clause one.', 'Clause two.', 'Clause three.']
        mock_analyzer.analyze_clauses.return_value = fake_analysis(risky_count=1, total_clauses=3)
        mock_get_analyzer.return_value = mock_analyzer

        response = self.client.post(
            self.url,
            {'text': 'Clause one. Clause two. Clause three.', 'filename': 'contract.pdf', 'save': True},
            format='json',
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['filename'], 'contract.pdf')
        self.assertEqual(response.data['total_clauses'], 3)
        self.assertEqual(response.data['risky_count'], 1)
        self.assertEqual(len(response.data['risky_clauses']), 1)
        self.assertTrue(response.data['saved'])
        self.assertIn('document_id', response.data)
        self.assertIn('processing_time_ms', response.data)
        self.assertIn('api_response_time_ms', response.data)

        # When the user asked to save, the record must actually be persisted,
        # so the returned document_id can be used later.
        self.assertEqual(AnalysisResult.objects.count(), 1)
        saved = AnalysisResult.objects.first()
        self.assertEqual(saved.document_id, response.data['document_id'])
        self.assertEqual(saved.filename, 'contract.pdf')
        self.assertEqual(saved.model_used, 'sbert')
        self.assertEqual(saved.owner, self.user)  # LEX-6: ownership recorded, not just accepted

    @patch('analyzer.views.get_analyzer')
    def test_analysis_stores_nothing_by_default(self, mock_get_analyzer):
        mock_analyzer = MagicMock()
        mock_analyzer.extract_clauses.return_value = ['Clause one.', 'Clause two.']
        mock_analyzer.analyze_clauses.return_value = fake_analysis(risky_count=1, total_clauses=2)
        mock_get_analyzer.return_value = mock_analyzer

        response = self.client.post(
            self.url,
            {'text': 'Clause one. Clause two.', 'filename': 'contract.pdf'},
            format='json',
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertFalse(response.data['saved'])
        self.assertNotIn('document_id', response.data)
        self.assertEqual(response.data['risky_count'], 1)
        self.assertEqual(AnalysisResult.objects.count(), 0)

    @patch('analyzer.views.get_analyzer')
    def test_only_a_true_save_value_stores_the_analysis(self, mock_get_analyzer):
        mock_analyzer = MagicMock()
        mock_analyzer.extract_clauses.return_value = ['Clause one.', 'Clause two.']
        mock_analyzer.analyze_clauses.return_value = fake_analysis(risky_count=0, total_clauses=2)
        mock_get_analyzer.return_value = mock_analyzer

        for value in (False, 'false', 'no', None, 0, ''):
            with self.subTest(save=value):
                response = self.client.post(
                    self.url,
                    {'text': 'Clause one. Clause two.', 'save': value},
                    format='json',
                )
                self.assertEqual(response.status_code, status.HTTP_201_CREATED)
                self.assertNotIn('document_id', response.data)
        self.assertEqual(AnalysisResult.objects.count(), 0)

    @patch('analyzer.views.get_analyzer')
    def test_risky_clauses_are_most_risky_first_and_capped(self, mock_get_analyzer):
        # 60 risky clauses with distinct scores, deliberately NOT in score order
        many_risky = [
            {'clause_id': i, 'text': f'clause {i}', 'risk': 'risky', 'confidence': 0.5 + (i * 7 % 60) / 130,
             'safe_prob': 0.0, 'risky_prob': 0.5 + (i * 7 % 60) / 130}
            for i in range(60)
        ]
        analysis = fake_analysis()
        analysis['risky_clauses'] = many_risky
        analysis['risky_count'] = 60

        mock_analyzer = MagicMock()
        mock_analyzer.extract_clauses.return_value = ['x'] * 60
        mock_analyzer.analyze_clauses.return_value = analysis
        mock_get_analyzer.return_value = mock_analyzer

        response = self.client.post(
            self.url, {'text': 'a' * 20, 'filename': 'big.pdf'}, format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        # capped at 50 even though the full analysis had 60; the total is still reported
        shown = response.data['risky_clauses']
        self.assertEqual(len(shown), 50)
        self.assertEqual(response.data['risky_count'], 60)
        # and they are the 50 highest-scoring ones, highest first
        scores = [c['risky_prob'] for c in shown]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(scores[0], max(c['risky_prob'] for c in many_risky))
        self.assertGreaterEqual(min(scores), sorted((c['risky_prob'] for c in many_risky), reverse=True)[49])

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
        # LEX-9: the raw exception text must NOT reach the client
        self.assertNotIn('model exploded', response.data['error'])
        self.assertIn('Internal error', response.data['error'])
        # A failed analysis must not leave a half-written record behind.
        self.assertEqual(AnalysisResult.objects.count(), 0)


class AuthenticationRequiredTests(APITestCase):
    """LEX-6: /analyze, /upload, /recent must all reject unauthenticated
    requests. Uses a plain (non-force_authenticate) client throughout,
    since force_authenticate bypasses the real auth mechanism entirely —
    it proves IsAuthenticated blocks anonymous requests, but wouldn't
    catch a real wiring bug in TokenAuthentication/SessionAuthentication
    themselves. The token test below exercises the real header-based path.
    """

    def test_analyze_rejects_unauthenticated(self):
        response = self.client.post(
            reverse('document-analyze'), {'text': 'a' * 20, 'filename': 'a.txt'}, format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    def test_upload_rejects_unauthenticated(self):
        upload = SimpleUploadedFile("contract.pdf", make_pdf_bytes("text"), content_type="application/pdf")
        response = self.client.post(reverse('document-upload'), {'file': upload}, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    def test_recent_rejects_unauthenticated(self):
        response = self.client.get(reverse('document-recent'))
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    def test_health_and_metrics_stay_open_when_unauthenticated(self):
        # These two are deliberately AllowAny — confirms LEX-6 didn't
        # accidentally lock out infra/monitoring endpoints too. Checking
        # "not 403" rather than a specific 200, since /health's own status
        # code legitimately depends on whether the model is loaded
        # (covered separately by HealthEndpointTests) — that's a different
        # concern from whether auth is required to reach it at all.
        self.assertNotEqual(self.client.get(reverse('health')).status_code, status.HTTP_403_FORBIDDEN)
        self.assertNotEqual(self.client.get(reverse('metrics')).status_code, status.HTTP_403_FORBIDDEN)

    def test_raw_create_and_update_are_not_routed(self):
        # DocumentAnalysisViewSet is deliberately not a full ModelViewSet —
        # records are only ever created via analyze/upload (which set
        # owner server-side). A raw POST here would bypass that and hit
        # an IntegrityError; confirms it's not reachable at all (405, not
        # 403 — the method itself isn't registered, regardless of auth).
        user = User.objects.create_user(username='carol', password='pw')
        self.client.force_authenticate(user=user)

        list_url = reverse('document-list')
        response = self.client.post(list_url, {'filename': 'x'}, format='json')
        self.assertEqual(response.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)

        record = AnalysisResult.objects.create(
            owner=user, document_id='d1', filename='a.pdf', clauses=['a'],
            analysis={'risky_count': 0}, processing_time_ms=1.0, model_used='sbert',
        )
        detail_url = reverse('document-detail', args=[record.id])
        self.assertEqual(self.client.put(detail_url, {}, format='json').status_code, status.HTTP_405_METHOD_NOT_ALLOWED)
        self.assertEqual(self.client.patch(detail_url, {}, format='json').status_code, status.HTTP_405_METHOD_NOT_ALLOWED)
        # DELETE of your own record is intentionally supported now (LEX-9
        # data retention) — covered in DataDeletionTests.

    def test_real_token_header_authenticates_successfully(self):
        # Goes through the actual TokenAuthentication class via a real
        # Authorization header — not force_authenticate — to prove the
        # signal-created token really works end-to-end, not just that
        # *some* authenticated request would pass.
        user = User.objects.create_user(username='bob', password='pw')
        token = Token.objects.get(user=user)  # auto-created by the post_save signal

        response = self.client.get(
            reverse('document-recent'), HTTP_AUTHORIZATION=f'Token {token.key}'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)


class RecentEndpointTests(APITestCase):
    def setUp(self):
        self.url = reverse('document-recent')
        self.user = User.objects.create_user(username='alice', password='pw')
        self.other_user = User.objects.create_user(username='mallory', password='pw')
        self.client.force_authenticate(user=self.user)

    def test_empty_when_no_analyses_exist(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data, {'count': 0, 'results': []})

    def test_returns_created_results(self):
        AnalysisResult.objects.create(
            owner=self.user, document_id='doc-1', filename='one.pdf', clauses=['a'],
            analysis={'risky_count': 2}, processing_time_ms=10.0, model_used='sbert',
        )
        AnalysisResult.objects.create(
            owner=self.user, document_id='doc-2', filename='two.pdf', clauses=['b'],
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
                owner=self.user, document_id=f'doc-{i}', filename=f'{i}.pdf', clauses=['a'],
                analysis={'risky_count': 0}, processing_time_ms=1.0, model_used='sbert',
            )

        response = self.client.get(self.url, {'limit': 2})
        self.assertEqual(response.data['count'], 2)

    def test_only_returns_the_authenticated_users_own_results(self):
        # The actual point of LEX-6's /recent scoping — not just "auth is
        # required somewhere", but "you specifically can't see someone
        # else's analyses even though you're logged in as a real user".
        AnalysisResult.objects.create(
            owner=self.user, document_id='mine', filename='mine.pdf', clauses=['a'],
            analysis={'risky_count': 1}, processing_time_ms=1.0, model_used='sbert',
        )
        AnalysisResult.objects.create(
            owner=self.other_user, document_id='theirs', filename='theirs.pdf', clauses=['a'],
            analysis={'risky_count': 1}, processing_time_ms=1.0, model_used='sbert',
        )

        response = self.client.get(self.url)
        self.assertEqual(response.data['count'], 1)
        self.assertEqual(response.data['results'][0]['filename'], 'mine.pdf')


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


class DocumentParserTests(TestCase):
    """Exercises the real PyMuPDF/python-docx extraction against real
    (minimal, generated) files — neither library touches torch, so unlike
    the ML pipeline these can run for real in any environment, not mocked."""

    def test_extracts_text_from_real_pdf(self):
        pdf_bytes = make_pdf_bytes("This clause limits liability.")
        text = extract_text_from_pdf_bytes(pdf_bytes)
        self.assertIn("This clause limits liability.", text)

    def test_extracts_text_from_real_docx(self):
        docx_bytes = make_docx_bytes("This clause limits liability.")
        text = extract_text_from_docx_bytes(docx_bytes)
        self.assertIn("This clause limits liability.", text)

    def test_corrupt_pdf_raises_parse_error(self):
        with self.assertRaises(DocumentParseError):
            extract_text_from_pdf_bytes(b"this is not a real pdf")

    def test_corrupt_docx_raises_parse_error(self):
        with self.assertRaises(DocumentParseError):
            extract_text_from_docx_bytes(b"this is not a real docx")

    def test_dispatches_by_extension_not_just_content_type(self):
        # A client that mislabels the content-type but names the file
        # correctly should still work — extension is checked first.
        upload = SimpleUploadedFile(
            "contract.pdf", make_pdf_bytes("Clause text here."), content_type="application/octet-stream"
        )
        text = extract_text_from_upload(upload)
        self.assertIn("Clause text here.", text)

    def test_unsupported_extension_raises(self):
        upload = SimpleUploadedFile("notes.txt", b"plain text", content_type="text/plain")
        with self.assertRaises(UnsupportedFileType):
            extract_text_from_upload(upload)


class UploadEndpointTests(APITestCase):
    def setUp(self):
        self.url = reverse('document-upload')
        self.user = User.objects.create_user(username='alice', password='pw')
        self.client.force_authenticate(user=self.user)

    def test_rejects_missing_file(self):
        response = self.client.post(self.url, {}, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('No file provided', response.data['error'])

    def test_rejects_unsupported_file_type(self):
        upload = SimpleUploadedFile("notes.txt", b"plain text content", content_type="text/plain")
        response = self.client.post(self.url, {'file': upload}, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('Unsupported file type', response.data['error'])

    def test_rejects_corrupt_pdf(self):
        upload = SimpleUploadedFile("contract.pdf", b"not a real pdf", content_type="application/pdf")
        response = self.client.post(self.url, {'file': upload}, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('Could not read PDF', response.data['error'])

    @override_settings(MAX_UPLOAD_SIZE_BYTES=100)
    def test_rejects_file_over_size_limit(self):
        upload = SimpleUploadedFile(
            "contract.pdf", make_pdf_bytes("x" * 500), content_type="application/pdf"
        )
        response = self.client.post(self.url, {'file': upload}, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('too large', response.data['error'])

    @patch('analyzer.views.get_analyzer')
    def test_successful_pdf_upload_returns_same_shape_as_analyze(self, mock_get_analyzer):
        mock_analyzer = MagicMock()
        mock_analyzer.extract_clauses.return_value = ['Clause one.', 'Clause two.']
        mock_analyzer.analyze_clauses.return_value = fake_analysis(risky_count=1, total_clauses=2)
        mock_get_analyzer.return_value = mock_analyzer

        upload = SimpleUploadedFile(
            "contract.pdf",
            make_pdf_bytes("Clause one. Clause two."),
            content_type="application/pdf",
        )
        response = self.client.post(self.url, {'file': upload, 'save': 'true'}, format='multipart')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['filename'], 'contract.pdf')
        self.assertEqual(response.data['total_clauses'], 2)
        self.assertEqual(response.data['risky_count'], 1)
        self.assertEqual(AnalysisResult.objects.count(), 1)
        self.assertEqual(AnalysisResult.objects.first().filename, 'contract.pdf')

    @patch('analyzer.views.get_analyzer')
    def test_upload_stores_nothing_by_default(self, mock_get_analyzer):
        mock_analyzer = MagicMock()
        mock_analyzer.extract_clauses.return_value = ['Clause one.', 'Clause two.']
        mock_analyzer.analyze_clauses.return_value = fake_analysis(risky_count=1, total_clauses=2)
        mock_get_analyzer.return_value = mock_analyzer

        upload = SimpleUploadedFile(
            "contract.pdf",
            make_pdf_bytes("Clause one. Clause two."),
            content_type="application/pdf",
        )
        response = self.client.post(self.url, {'file': upload}, format='multipart')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertFalse(response.data['saved'])
        self.assertNotIn('document_id', response.data)
        self.assertEqual(AnalysisResult.objects.count(), 0)

    @patch('analyzer.views.get_analyzer')
    def test_successful_docx_upload(self, mock_get_analyzer):
        mock_analyzer = MagicMock()
        mock_analyzer.extract_clauses.return_value = ['Clause one.', 'Clause two.']
        mock_analyzer.analyze_clauses.return_value = fake_analysis(risky_count=0, total_clauses=2)
        mock_get_analyzer.return_value = mock_analyzer

        upload = SimpleUploadedFile(
            "contract.docx",
            make_docx_bytes("Clause one. Clause two."),
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        response = self.client.post(self.url, {'file': upload}, format='multipart')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['filename'], 'contract.docx')
        self.assertEqual(response.data['risky_count'], 0)


class FrontendPageTests(TestCase):
    """The upload/results page (LEX-5). Just checks it renders and wires up
    to the real endpoint paths — the JS behavior itself isn't exercised by
    Django's test client (no browser), that's a manual/visual check."""

    def test_index_page_renders(self):
        response = self.client.get(reverse('index'))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'LexiLite', response.content)

    def test_index_page_references_real_api_paths(self):
        # Regression guard: if the upload/analyze URL names ever change,
        # this page's hardcoded fetch() paths would silently break with
        # no server-side error — this test exists so *something* fails
        # loudly instead.
        response = self.client.get(reverse('index'))
        self.assertIn(reverse('document-upload').encode(), response.content)
        self.assertIn(reverse('document-analyze').encode(), response.content)


class SecurityHardeningTests(APITestCase):
    """LEX-9: regressions for issues found in the security review."""

    def setUp(self):
        self.user = User.objects.create_user(username='alice', password='pw')
        self.client.force_authenticate(user=self.user)

    def test_non_string_text_is_a_400_not_a_500(self):
        response = self.client.post(
            reverse('document-analyze'), {'text': 12345678901234, 'filename': 'a.txt'}, format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_recent_rejects_non_integer_limit(self):
        response = self.client.get(reverse('document-recent'), {'limit': 'abc'})
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_recent_clamps_huge_and_negative_limits(self):
        for i in range(3):
            AnalysisResult.objects.create(
                owner=self.user, document_id=f'doc-{i}', filename='f.pdf', clauses=['a'],
                analysis={'risky_count': 0}, processing_time_ms=1.0, model_used='sbert',
            )
        self.assertEqual(self.client.get(reverse('document-recent'), {'limit': 10**9}).status_code, 200)
        negative = self.client.get(reverse('document-recent'), {'limit': -5})
        self.assertEqual(negative.status_code, 200)
        self.assertEqual(negative.data['count'], 1)  # clamped to the minimum of 1

    @patch('analyzer.views.get_analyzer')
    def test_internal_errors_do_not_leak_exception_text(self, mock_get_analyzer):
        mock_get_analyzer.return_value.extract_clauses.side_effect = RuntimeError('secret path C:\\internal\\x')
        response = self.client.post(
            reverse('document-analyze'), {'text': 'A long enough document text.', 'filename': 'a.txt'}, format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertNotIn('secret path', str(response.data))

    @patch('analyzer.views.get_analyzer')
    def test_overlong_filename_is_truncated_not_a_crash(self, mock_get_analyzer):
        analyzer = mock_get_analyzer.return_value
        analyzer.extract_clauses.return_value = ['one clause']
        analyzer.analyze_clauses.return_value = {
            'total_clauses': 1, 'risky_count': 0, 'risky_clauses': [], 'processing_time_ms': 1.0,
        }
        response = self.client.post(
            reverse('document-analyze'), {'text': 'A long enough document text.', 'filename': 'x' * 5000}, format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(len(response.data['filename']), 255)


class ProductionSettingsTests(TestCase):
    def test_refuses_to_start_non_debug_with_the_public_dev_secret_key(self):
        import os, subprocess, sys
        env = {**os.environ, 'DEBUG': 'False'}
        env.pop('DJANGO_SECRET_KEY', None)
        env['DJANGO_SETTINGS_MODULE'] = 'config.settings'
        result = subprocess.run(
            [sys.executable, '-c', 'import django; from django.conf import settings; settings.SECRET_KEY'],
            env=env, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('DJANGO_SECRET_KEY', result.stderr)


def _make_result(owner, doc_id, age_days=0):
    from datetime import timedelta
    from django.utils import timezone
    obj = AnalysisResult.objects.create(
        owner=owner, document_id=doc_id, filename=f'{doc_id}.pdf', clauses=['secret clause text'],
        analysis={'risky_count': 0}, processing_time_ms=1.0, model_used='sbert',
    )
    # created_at is auto_now_add, so backdate with an update() instead.
    AnalysisResult.objects.filter(pk=obj.pk).update(created_at=timezone.now() - timedelta(days=age_days))
    return obj


class DataDeletionTests(APITestCase):
    """LEX-9: users can delete their own stored data, never anyone else's."""

    def setUp(self):
        self.alice = User.objects.create_user(username='alice', password='pw')
        self.mallory = User.objects.create_user(username='mallory', password='pw')

    def test_user_can_delete_their_own_analysis(self):
        obj = _make_result(self.alice, 'a1')
        self.client.force_authenticate(user=self.alice)
        response = self.client.delete(reverse('document-detail', args=[obj.pk]))
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertFalse(AnalysisResult.objects.filter(pk=obj.pk).exists())

    def test_cannot_delete_another_users_analysis(self):
        obj = _make_result(self.alice, 'a1')
        self.client.force_authenticate(user=self.mallory)
        response = self.client.delete(reverse('document-detail', args=[obj.pk]))
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertTrue(AnalysisResult.objects.filter(pk=obj.pk).exists())

    def test_delete_all_only_removes_the_callers_data(self):
        _make_result(self.alice, 'a1'); _make_result(self.alice, 'a2'); _make_result(self.mallory, 'm1')
        self.client.force_authenticate(user=self.alice)
        response = self.client.delete(reverse('document-delete-all'))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data, {'deleted': 2})
        self.assertEqual(AnalysisResult.objects.filter(owner=self.mallory).count(), 1)
        self.assertEqual(AnalysisResult.objects.filter(owner=self.alice).count(), 0)

    def test_deletion_requires_authentication(self):
        obj = _make_result(self.alice, 'a1')
        self.assertEqual(self.client.delete(reverse('document-detail', args=[obj.pk])).status_code, 403)
        self.assertEqual(self.client.delete(reverse('document-delete-all')).status_code, 403)
        self.assertTrue(AnalysisResult.objects.filter(pk=obj.pk).exists())


class PurgeExpiredAnalysesTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='alice', password='pw')
        _make_result(self.user, 'old', age_days=40)
        _make_result(self.user, 'fresh', age_days=5)

    def _run(self, *args):
        from django.core.management import call_command
        out = io.StringIO()
        call_command('purge_expired_analyses', *args, stdout=out)
        return out.getvalue()

    @override_settings(RETENTION_DAYS=30)
    def test_deletes_only_rows_older_than_the_retention_period(self):
        self._run()
        self.assertEqual(list(AnalysisResult.objects.values_list('document_id', flat=True)), ['fresh'])

    @override_settings(RETENTION_DAYS=30)
    def test_dry_run_deletes_nothing(self):
        output = self._run('--dry-run')
        self.assertIn('1 analyses', output)
        self.assertEqual(AnalysisResult.objects.count(), 2)

    @override_settings(RETENTION_DAYS=0)
    def test_zero_days_disables_retention(self):
        self._run()
        self.assertEqual(AnalysisResult.objects.count(), 2)

    @override_settings(RETENTION_DAYS=30)
    def test_days_flag_overrides_the_setting(self):
        self._run('--days', '3')
        self.assertEqual(AnalysisResult.objects.count(), 0)


class ConfidenceScorerTests(TestCase):
    """LEX-13: the confidence gate, and why the margin check is opt-in."""

    LABELS = np.array(['risky', 'safe'])

    def _score(self, probs, **kwargs):
        from .confidence import ConfidenceScorer
        return ConfidenceScorer(**kwargs).score(np.array(probs), self.LABELS)

    def test_accepts_above_threshold_and_rejects_below(self):
        high, low = self._score([[0.10, 0.90], [0.40, 0.60]], min_confidence=0.65)
        self.assertTrue(high['accepted'])
        self.assertEqual(high['label'], 'safe')
        self.assertFalse(low['accepted'])
        self.assertEqual(low['label'], 'uncertain')
        self.assertEqual(low['raw_label'], 'safe')  # the guess is kept, just not trusted

    def test_margin_is_reported_but_not_enforced_by_default(self):
        (result,) = self._score([[0.30, 0.70]])
        self.assertAlmostEqual(result['margin'], 0.40, places=4)
        self.assertIsNone(__import__('analyzer.confidence', fromlist=['x']).ConfidenceScorer().min_margin)

    def test_for_two_classes_a_margin_threshold_is_just_a_confidence_threshold(self):
        # The LEX-13 finding as an executable fact: margin = 2p - 1, so
        # (confidence >= 0.65 AND margin >= 0.40) accepts exactly the same
        # clauses as (confidence >= 0.70) alone, for every possible p.
        # (Points within float-rounding distance of the 0.70 boundary are
        # skipped: there 2p - 1 can land at 0.3999999... and flip by 1 ulp.)
        probs = [[1 - p, p] for p in np.linspace(0.5, 1.0, 501) if abs(p - 0.70) > 1e-6]
        with_margin = [r['accepted'] for r in self._score(probs, min_confidence=0.65, min_margin=0.40)]
        confidence_only = [r['accepted'] for r in self._score(probs, min_confidence=0.70)]
        self.assertEqual(with_margin, confidence_only)

    def test_old_default_margin_never_rejected_anything_the_confidence_gate_passed(self):
        probs = [[1 - p, p] for p in np.linspace(0.5, 1.0, 501)]
        old = [r['accepted'] for r in self._score(probs, min_confidence=0.65, min_margin=0.15)]
        new = [r['accepted'] for r in self._score(probs, min_confidence=0.65)]
        self.assertEqual(old, new)

    def test_margin_does_matter_with_three_classes(self):
        # Top class 0.45 vs runner-up 0.44: confidence alone (>= 0.40) passes,
        # but the near-tie is exactly what a margin gate should catch.
        probs = [[0.45, 0.44, 0.11]]
        from .confidence import ConfidenceScorer
        labels = np.array(['a', 'b', 'c'])
        self.assertTrue(ConfidenceScorer(min_confidence=0.40).score(np.array(probs), labels)[0]['accepted'])
        self.assertFalse(ConfidenceScorer(min_confidence=0.40, min_margin=0.15).score(np.array(probs), labels)[0]['accepted'])


class FallbackRouterTests(TestCase):
    def _primary(self, accepted, label='risky', confidence=0.9):
        return {'accepted': accepted, 'label': label, 'confidence': confidence}

    def _router(self, secondary=None):
        from .confidence import ConfidenceScorer, FallbackRouter
        return FallbackRouter(ConfidenceScorer(min_confidence=0.65), secondary_predict_fn=secondary)

    def test_accepted_primary_result_is_used_as_is(self):
        decision = self._router().resolve('clause', self._primary(True))
        self.assertEqual((decision['label'], decision['source']), ('risky', 'primary'))

    def test_rejected_without_secondary_goes_to_human_review(self):
        decision = self._router().resolve('clause', self._primary(False, 'uncertain', 0.55))
        self.assertEqual((decision['label'], decision['source']), ('needs_review', 'human_review'))

    def test_confident_secondary_model_overrides(self):
        decision = self._router(lambda text: ('safe', 0.9)).resolve('clause', self._primary(False, 'uncertain', 0.55))
        self.assertEqual((decision['label'], decision['source']), ('safe', 'secondary'))

    def test_unconfident_secondary_still_goes_to_human_review(self):
        decision = self._router(lambda text: ('safe', 0.5)).resolve('clause', self._primary(False, 'uncertain', 0.55))
        self.assertEqual(decision['source'], 'human_review')

    def test_secondary_failure_never_crashes_the_request(self):
        def boom(text):
            raise RuntimeError('secondary model down')
        decision = self._router(boom).resolve('clause', self._primary(False, 'uncertain', 0.55))
        self.assertEqual(decision['source'], 'human_review')


class IpReviewTests(TestCase):
    """The IP-assignment review: the only contract type LexiLite checks."""

    def test_unreadable_input_is_reported_not_treated_as_clean(self):
        from .ip_review import review_ip_clauses
        for clauses in ([], ['', '   '], None):
            with self.subTest(clauses=clauses):
                review = review_ip_clauses(clauses)
                self.assertTrue(review['unreadable'])
                self.assertEqual(review['flags'], [])

    def test_non_ip_clauses_are_not_checked_and_not_flagged(self):
        from .ip_review import review_ip_clauses
        review = review_ip_clauses([
            'Client will pay each invoice within 30 days of acceptance of the deliverable.',
            'Freelancer liability shall be unlimited.',
        ])
        self.assertFalse(review['unreadable'])
        self.assertEqual(review['ip_clause_count'], 0)
        self.assertEqual(review['flags'], [])

    def test_every_result_states_scope_and_draft_status(self):
        from .ip_review import review_ip_clauses, REVIEW_STATUS
        review = review_ip_clauses(['All work product vests in Client upon creation.'])
        self.assertEqual(review['status'], REVIEW_STATUS)
        self.assertIn('IP assignment', review['checked_scope'])
        self.assertIn('Liability and indemnity', review['not_checked'])

    def test_ownership_before_payment_is_not_flagged_when_payment_is_a_condition(self):
        from .ip_review import review_ip_clauses
        conditional = review_ip_clauses([
            'Ownership of the work product passes to Client upon creation, but only once the invoice is paid in full.',
        ])
        ids = [i['id'] for f in conditional['flags'] for i in f['issues']]
        self.assertNotIn('ownership_before_payment', ids)

        unconditional = review_ip_clauses([
            'Ownership of the work product passes to Client upon creation.',
        ])
        ids = [i['id'] for f in unconditional['flags'] for i in f['issues']]
        self.assertIn('ownership_before_payment', ids)

    def test_every_issue_carries_reason_paste_sentence_and_lawyer_note(self):
        from .ip_review import review_ip_clauses
        review = review_ip_clauses([
            'All work product vests in Client upon creation. Freelancer waives all moral rights.',
        ])
        self.assertTrue(review['flags'])
        for flag in review['flags']:
            for issue in flag['issues']:
                self.assertTrue(issue['reason'])
                self.assertTrue(issue['paste_sentence'])
                self.assertTrue(issue['lawyer_when'])


class SampleReportTests(TestCase):
    """The page's example is produced by the real review on a fixture contract.
    If the library changes what it flags on that contract, this test fails, so
    the example cannot quietly drift from the product."""

    EXPECTED_FLAGS = {
        4: ['all_work_assigned', 'ownership_before_payment'],
        5: ['pre_existing_materials'],
        6: ['perpetual_or_worldwide'],
        7: ['moral_rights_waiver'],
    }

    def test_sample_report_flags_match_the_reviewed_library_output(self):
        from .views import sample_report
        report = sample_report()
        flagged = {f['clause_id']: [i['id'] for i in f['issues']] for f in report['flags']}
        self.assertEqual(flagged, self.EXPECTED_FLAGS)
        self.assertEqual(report['ip_clause_count'], 4)

    def test_sample_report_is_shown_on_the_page(self):
        response = self.client.get(reverse('index'))
        self.assertContains(response, 'sample-report-data')
        self.assertContains(response, 'IP assignment clauses only')
        self.assertContains(response, 'draft, pending lawyer review')


class ExpiryOnReadTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='alice', password='pw')
        self.client.force_authenticate(user=self.user)

    @override_settings(RETENTION_DAYS=30)
    def test_expired_analysis_is_not_returned_even_if_purge_has_not_run(self):
        old = _make_result(self.user, 'old', age_days=40)
        fresh = _make_result(self.user, 'fresh', age_days=5)

        recent = self.client.get(reverse('document-recent'))
        self.assertEqual([r['document_id'] for r in recent.data['results']], ['fresh'])
        self.assertEqual(self.client.get(reverse('document-detail', args=[old.pk])).status_code, 404)
        self.assertEqual(self.client.get(reverse('document-detail', args=[fresh.pk])).status_code, 200)

    @override_settings(RETENTION_DAYS=0)
    def test_zero_retention_disables_expiry_on_read(self):
        old = _make_result(self.user, 'old', age_days=400)
        self.assertEqual(self.client.get(reverse('document-detail', args=[old.pk])).status_code, 200)


class IpReviewInResponseTests(APITestCase):
    def setUp(self):
        self.url = reverse('document-analyze')
        self.user = User.objects.create_user(username='alice', password='pw')
        self.client.force_authenticate(user=self.user)

    @patch('analyzer.views.get_analyzer')
    def test_analyze_response_includes_the_ip_review(self, mock_get_analyzer):
        mock_analyzer = MagicMock()
        mock_analyzer.extract_clauses.return_value = [
            'All work product vests in Client upon creation.',
            'Freelancer liability shall be unlimited.',
        ]
        mock_analyzer.analyze_clauses.return_value = fake_analysis(risky_count=1, total_clauses=2)
        mock_get_analyzer.return_value = mock_analyzer

        response = self.client.post(self.url, {'text': 'x' * 40, 'filename': 'c.pdf'}, format='json')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        review = response.data['ip_review']
        self.assertEqual(review['ip_clause_count'], 1)
        self.assertEqual(review['flags'][0]['clause_id'], 0)
        self.assertEqual(review['status'], 'draft_pending_lawyer_review')


class PaidReportTests(APITestCase):
    """The paid full report. The model is faked: these tests check the logic
    (access, fallbacks, privacy), not the quality of any explanation."""

    CLAUSES = [
        'Freelancer assigns all work product to Client upon creation.',
        'Freelancer liability shall be unlimited.',
    ]

    def setUp(self):
        from django.contrib.auth.models import Group
        self.url = reverse('document-report')
        self.user = User.objects.create_user(username='alice', password='pw')
        self.paid_group, _ = Group.objects.get_or_create(name='paid_report')
        self.client.force_authenticate(user=self.user)

    def _make_paid(self):
        self.user.groups.add(self.paid_group)

    def _post(self):
        return self.client.post(self.url, {'text': 'x' * 40}, format='json')

    def _mock_analyzer(self, mock_get_analyzer):
        analyzer = MagicMock()
        analyzer.extract_clauses.return_value = self.CLAUSES
        mock_get_analyzer.return_value = analyzer

    def test_anonymous_users_are_refused(self):
        self.client.force_authenticate(user=None)
        self.assertEqual(self._post().status_code, status.HTTP_403_FORBIDDEN)

    @patch('analyzer.views.get_analyzer')
    def test_users_not_on_the_paid_plan_are_refused(self, mock_get_analyzer):
        self._mock_analyzer(mock_get_analyzer)
        with override_settings(LLM_ENDPOINT='http://model.internal/api/generate'):
            response = self._post()
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    @patch('analyzer.views.get_analyzer')
    def test_report_is_unavailable_when_no_model_is_configured(self, mock_get_analyzer):
        self._mock_analyzer(mock_get_analyzer)
        self._make_paid()
        with override_settings(LLM_ENDPOINT=''):
            response = self._post()
        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)

    @override_settings(LLM_ENDPOINT='http://model.internal/api/generate', LLM_MODEL='local-model')
    @patch('analyzer.report.requests.post')
    @patch('analyzer.views.get_analyzer')
    def test_model_explanation_is_used_and_labelled(self, mock_get_analyzer, mock_post):
        self._mock_analyzer(mock_get_analyzer)
        self._make_paid()
        mock_post.return_value = MagicMock(
            json=MagicMock(return_value={'response': 'This gives the client all your work.'}),
        )
        mock_post.return_value.raise_for_status = MagicMock()

        response = self._post()

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['report_status'], 'ai_explanation_pending_lawyer_review')
        issue = response.data['flags'][0]['issues'][0]
        self.assertEqual(issue['explanation'], 'This gives the client all your work.')
        self.assertEqual(issue['explanation_source'], 'model')

    @override_settings(LLM_ENDPOINT='http://model.internal/api/generate', LLM_MODEL='local-model')
    @patch('analyzer.report.requests.post')
    @patch('analyzer.views.get_analyzer')
    def test_contract_text_only_goes_to_our_own_endpoint(self, mock_get_analyzer, mock_post):
        self._mock_analyzer(mock_get_analyzer)
        self._make_paid()
        mock_post.return_value = MagicMock(json=MagicMock(return_value={'response': 'ok'}))
        mock_post.return_value.raise_for_status = MagicMock()

        self._post()

        self.assertTrue(mock_post.called)
        for call in mock_post.call_args_list:
            self.assertEqual(call.args[0], 'http://model.internal/api/generate')

    @override_settings(LLM_ENDPOINT='http://model.internal/api/generate', LLM_MODEL='local-model')
    @patch('analyzer.report.requests.post')
    @patch('analyzer.views.get_analyzer')
    def test_model_failure_falls_back_to_the_library_wording(self, mock_get_analyzer, mock_post):
        import requests as requests_lib
        self._mock_analyzer(mock_get_analyzer)
        self._make_paid()
        mock_post.side_effect = requests_lib.ConnectionError('model down')

        response = self._post()

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        issue = response.data['flags'][0]['issues'][0]
        self.assertEqual(issue['explanation'], issue['reason'])
        self.assertEqual(issue['explanation_source'], 'library')

    @override_settings(LLM_ENDPOINT='http://model.internal/api/generate', LLM_MODEL='local-model')
    @patch('analyzer.report.requests.post')
    @patch('analyzer.views.get_analyzer')
    def test_empty_model_answer_falls_back(self, mock_get_analyzer, mock_post):
        self._mock_analyzer(mock_get_analyzer)
        self._make_paid()
        mock_post.return_value = MagicMock(json=MagicMock(return_value={'response': '   '}))
        mock_post.return_value.raise_for_status = MagicMock()

        response = self._post()

        self.assertEqual(response.data['flags'][0]['issues'][0]['explanation_source'], 'library')

    @override_settings(LLM_ENDPOINT='http://model.internal/api/generate', LLM_MODEL='local-model')
    @patch('analyzer.report.requests.post')
    @patch('analyzer.views.get_analyzer')
    def test_model_cannot_add_a_flag_the_library_did_not_produce(self, mock_get_analyzer, mock_post):
        self._mock_analyzer(mock_get_analyzer)
        self._make_paid()
        mock_post.return_value = MagicMock(
            json=MagicMock(return_value={'response': 'Also watch the non-compete clause.'}),
        )
        mock_post.return_value.raise_for_status = MagicMock()

        response = self._post()

        flagged_ids = [f['clause_id'] for f in response.data['flags']]
        self.assertEqual(flagged_ids, [0])  # the liability clause is not in scope, so never flagged
        self.assertEqual(len(response.data['flags'][0]['issues']), 1)

    @override_settings(LLM_ENDPOINT='http://model.internal/api/generate', LLM_MODEL='local-model')
    @patch('analyzer.report.requests.post')
    @patch('analyzer.views.get_analyzer')
    def test_long_model_answers_are_capped(self, mock_get_analyzer, mock_post):
        from analyzer.report import MAX_EXPLANATION_CHARS
        self._mock_analyzer(mock_get_analyzer)
        self._make_paid()
        mock_post.return_value = MagicMock(json=MagicMock(return_value={'response': 'x' * 5000}))
        mock_post.return_value.raise_for_status = MagicMock()

        response = self._post()

        self.assertEqual(len(response.data['flags'][0]['issues'][0]['explanation']), MAX_EXPLANATION_CHARS)

    @override_settings(LLM_ENDPOINT='http://model.internal/api/generate', LLM_MODEL='local-model')
    @patch('analyzer.report.requests.post')
    @patch('analyzer.views.get_analyzer')
    def test_report_stores_nothing(self, mock_get_analyzer, mock_post):
        self._mock_analyzer(mock_get_analyzer)
        self._make_paid()
        mock_post.return_value = MagicMock(json=MagicMock(return_value={'response': 'ok'}))
        mock_post.return_value.raise_for_status = MagicMock()

        self._post()

        self.assertEqual(AnalysisResult.objects.count(), 0)

class SegmentationTests(TestCase):
    """LEX-17: PDF-style text must become clause-sized chunks, losing nothing."""

    def seg(self, text, **kw):
        from .segmentation import segment_clauses
        return segment_clauses(text, **kw)

    @staticmethod
    def words(s):
        import re
        return re.findall(r"[A-Za-z0-9$]+", s)

    def test_empty_and_whitespace_give_no_chunks(self):
        self.assertEqual(self.seg(''), [])
        self.assertEqual(self.seg('  \n \n  '), [])

    def test_page_headers_and_bare_page_numbers_are_dropped(self):
        text = ("Page 1 \nThe Supplier shall deliver the goods within thirty days of the order date. \n"
                "Page 2 of 9 \n12\n- 3 -\nThe Customer shall pay each invoice within forty-five days of receipt. \n")
        chunks = self.seg(text)
        self.assertFalse(any('Page' in c for c in chunks))
        self.assertTrue(any('deliver the goods' in c for c in chunks))
        self.assertTrue(any('forty-five days' in c for c in chunks))

    def test_hard_wrapped_lines_are_rejoined_into_one_clause(self):
        text = ("1) The Supplier agrees to maintain the required insurance coverage for a period of \n"
                "three (3) years after the expiration of this Agreement and any extensions \n"
                "thereof, at its own expense. \n")
        (chunk,) = self.seg(text)
        self.assertIn('period of three (3) years after the expiration', chunk)
        self.assertNotIn('\n', chunk)

    def test_numbered_items_become_separate_clauses(self):
        text = ("3) Comprehensive General Liability Insurance in the minimum amount of one million dollars. \n \n"
                "4) Professional Liability Insurance in the minimum amount of one million dollars, if initialed. \n \n"
                "5) Workers Compensation Insurance as required by the laws of the State of California. \n")
        chunks = self.seg(text)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(chunks[1].startswith('4)'))

    def test_numbering_splits_even_without_blank_lines(self):
        text = ("3) Comprehensive General Liability Insurance in the minimum amount of one \nmillion dollars each occurrence \n"
                "4) Professional Liability Insurance in the minimum amount of one \nmillion dollars each claim \n")
        chunks = self.seg(text)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith('3)') and chunks[1].startswith('4)'))

    def test_heading_travels_with_the_clause_that_follows_it(self):
        text = ("B. Other Insurance Provisions \n"
                "1) If any insurance coverage required in this Agreement is provided on a claims made form, the \n"
                "Consultant agrees to maintain the required coverage for three years. \n")
        (chunk,) = self.seg(text)
        self.assertTrue(chunk.startswith('B. Other Insurance Provisions 1) If any insurance'))

    def test_a_clause_that_runs_across_a_page_break_stays_whole(self):
        text = ("1) The Consultant shall indemnify and hold harmless the Commission from any claim arising \n"
                "Page 5 \n"
                "out of the Consultant's negligence or willful misconduct under this Agreement. \n")
        (chunk,) = self.seg(text)
        self.assertIn('any claim arising out of the Consultant', chunk)

    def test_simple_one_clause_per_line_paste_is_preserved(self):
        text = ("The Supplier shall deliver the goods within thirty days of the order. \n"
                "The Customer shall pay each invoice within forty-five days of receipt. \n"
                "Either party may terminate this Agreement on thirty days written notice. \n")
        self.assertEqual(len(self.seg(text)), 3)

    def test_blank_line_separated_paragraphs_are_preserved(self):
        text = ("The Supplier shall deliver the goods within thirty days of the order, time being of the essence.\n\n"
                "The Customer shall pay each invoice within forty-five days of receipt of a valid invoice.")
        self.assertEqual(len(self.seg(text)), 2)

    def test_long_paragraph_is_split_at_sentences_within_the_limit(self):
        sentence = "The Consultant shall comply with all applicable laws and regulations of the jurisdiction. "
        chunks = self.seg(sentence * 40, max_chars=800)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(c) <= 840 for c in chunks), [len(c) for c in chunks])
        # packed greedily at sentence boundaries: no chunk starts mid-sentence
        self.assertTrue(all(c.startswith('The Consultant') for c in chunks))

    def test_one_endless_sentence_is_hard_split_never_truncated(self):
        text = ' '.join(f'word{i}' for i in range(600))  # ~4,000 chars, no punctuation at all
        chunks = self.seg(text, max_chars=800)
        self.assertTrue(all(len(c) <= 840 for c in chunks))
        self.assertEqual(self.words(' '.join(chunks)), self.words(text))

    def test_no_words_are_lost_on_a_multi_page_contract(self):
        import textwrap
        paragraphs = [
            f"{n}) The Consultant shall perform obligation number {n} under this Agreement in a professional "
            f"manner, and shall indemnify the Commission against any claim arising from that performance, "
            f"including reasonable attorney fees, costs and expenses incurred. Further obligations apply."
            for n in range(1, 41)
        ]
        pages, page_no = [], 1
        for i in range(0, len(paragraphs), 4):
            body = '\n \n'.join(textwrap.fill(p, 95) + ' ' for p in paragraphs[i:i + 4])
            pages.append(f"Page {page_no} \n{body}\n")
            page_no += 1
        text = '\n'.join(pages)

        chunks = self.seg(text)
        original = self.words(' '.join(p for p in paragraphs))
        self.assertEqual(self.words(' '.join(chunks)), original)
        # 10 pages used to give 10 page-sized "clauses"; now it gives one per numbered clause
        self.assertEqual(len(chunks), 40)
        self.assertLessEqual(max(len(c) for c in chunks), 800)

    def test_analyzer_extract_clauses_uses_the_segmenter_and_returns_everything(self):
        analyzer = LegalAnalyzer.__new__(LegalAnalyzer)
        text = '\n \n'.join(f"{n}) Clause number {n} states an obligation that is long enough to stand alone here." for n in range(1, 151))
        self.assertEqual(len(analyzer.extract_clauses(text)), 150)  # no silent 100-clause cap any more


class ClauseCapAndTextTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='alice', password='pw')
        self.client.force_authenticate(user=self.user)
        self.url = reverse('document-analyze')

    def _post(self, n_clauses):
        with patch('analyzer.views.get_analyzer') as mock_get:
            analyzer = mock_get.return_value
            analyzer.extract_clauses.return_value = [f'clause {i}' for i in range(n_clauses)]
            analyzer.analyze_clauses.side_effect = lambda clauses: {
                'total_clauses': len(clauses), 'risky_count': 0, 'risky_clauses': [], 'processing_time_ms': 1.0,
            }
            response = self.client.post(self.url, {'text': 'a long enough document', 'filename': 'a.pdf'}, format='json')
            return response, analyzer

    @override_settings(MAX_CLAUSES_PER_DOCUMENT=3)
    def test_long_document_is_capped_and_says_so(self):
        response, analyzer = self._post(5)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(response.data['clauses_truncated'])
        self.assertEqual(response.data['total_clauses'], 3)
        self.assertEqual(len(analyzer.analyze_clauses.call_args[0][0]), 3)

    @override_settings(MAX_CLAUSES_PER_DOCUMENT=3)
    def test_short_document_is_not_flagged_as_truncated(self):
        response, _ = self._post(3)
        self.assertFalse(response.data['clauses_truncated'])

    def test_analyze_clauses_returns_the_full_clause_text_not_a_200_char_prefix(self):
        analyzer = ProbabilityMappingRegressionTests()._analyzer_with_fake_classifier(['risky', 'safe'])
        long_clause = 'The Consultant shall indemnify the Commission. ' * 15  # ~700 chars
        returned = analyzer.analyze_clauses([long_clause])['all_clauses'][0]['text']
        self.assertEqual(returned, long_clause)


def _groq_ok(text='Plain words.'):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {'choices': [{'message': {'content': text}}]}
    return response


def _risky(i, prob):
    return {'clause_id': i, 'text': f'clause {i}', 'risk': 'risky', 'confidence': prob,
            'safe_prob': round(1 - prob, 4), 'risky_prob': prob}


class ExplainClauseTests(TestCase):
    """LEX-4: one Groq call. The privacy and failure behaviour is the point."""

    @override_settings(GROQ_API_KEY='test-key', GROQ_MODEL='test-model', EXPLAIN_TIMEOUT_SECONDS=7)
    @patch('analyzer.explain.requests.post')
    def test_sends_the_clause_with_the_key_and_returns_trimmed_text(self, post):
        from .explain import explain_clause, GROQ_API_URL
        post.return_value = _groq_ok('  Plain words.  ')
        self.assertEqual(explain_clause('The Consultant assigns all rights.'), 'Plain words.')
        args, kwargs = post.call_args
        self.assertEqual(args[0], GROQ_API_URL)
        self.assertEqual(kwargs['headers']['Authorization'], 'Bearer test-key')
        self.assertEqual(kwargs['json']['model'], 'test-model')
        self.assertEqual(kwargs['json']['reasoning_effort'], 'low')
        self.assertGreaterEqual(kwargs['json']['max_tokens'], 500)
        self.assertEqual(kwargs['timeout'], 7)
        system, user = (m['content'] for m in kwargs['json']['messages'])
        self.assertIn('The Consultant assigns all rights.', user)
        # contract text is data in the user message, never part of the instructions
        self.assertNotIn('The Consultant assigns all rights.', system)
        self.assertIn('ignore any instructions', system)

    @override_settings(GROQ_API_KEY='test-key', EXPLAIN_TIMEOUT_SECONDS=5)
    @patch('analyzer.explain.requests.post')
    def test_every_kind_of_failure_returns_none_and_never_raises(self, post):
        import requests as real_requests
        from .explain import explain_clause
        bad_json = MagicMock(); bad_json.raise_for_status.return_value = None
        bad_json.json.side_effect = ValueError('not json')
        no_choices = MagicMock(); no_choices.raise_for_status.return_value = None
        no_choices.json.return_value = {}
        http_error = MagicMock(); http_error.raise_for_status.side_effect = real_requests.HTTPError('401')
        cases = {
            'connection': real_requests.ConnectionError('down'),
            'timeout': real_requests.Timeout('slow'),
            'http error': http_error,
            'bad json': bad_json,
            'no choices': no_choices,
            'empty text': _groq_ok('   '),
            'non-string text': _groq_ok(None),
        }
        for name, outcome in cases.items():
            with self.subTest(name):
                if isinstance(outcome, Exception):
                    post.side_effect, post.return_value = outcome, None
                else:
                    post.side_effect, post.return_value = None, outcome
                self.assertIsNone(explain_clause('some clause'))

    @override_settings(GROQ_API_KEY='test-key')
    @patch('analyzer.explain.requests.post')
    def test_reasoning_field_is_ignored(self, post):
        from .explain import explain_clause
        post.return_value = MagicMock(**{'raise_for_status.return_value': None, 'json.return_value': {
            'choices': [{'message': {'content': 'Answer.', 'reasoning': 'secret thoughts'}}]}})
        self.assertEqual(explain_clause('c'), 'Answer.')

    @override_settings(GROQ_API_KEY='test-key', GROQ_MODEL='gone-model')
    @patch('analyzer.explain.requests.post')
    def test_model_not_found_is_logged_clearly_without_the_key(self, post):
        import requests as real_requests
        from .explain import explain_clause
        resp = MagicMock(status_code=404)
        resp.json.return_value = {'error': {'code': 'model_not_found', 'message': 'x'}}
        resp.raise_for_status.side_effect = real_requests.HTTPError('404', response=resp)
        post.return_value = resp
        with self.assertLogs('analyzer.explain', level='WARNING') as logs:
            self.assertIsNone(explain_clause('SECRET CLAUSE TEXT'))
        joined = ' '.join(logs.output)
        self.assertIn('model_not_found', joined)
        self.assertIn('gone-model', joined)
        self.assertNotIn('test-key', joined)
        self.assertNotIn('SECRET CLAUSE TEXT', joined)

    @override_settings(GROQ_API_KEY='test-key')
    @patch('analyzer.explain.requests.post')
    def test_long_answers_and_long_clauses_are_capped(self, post):
        from .explain import explain_clause, MAX_EXPLANATION_CHARS, MAX_CLAUSE_CHARS_SENT
        post.return_value = _groq_ok('x' * 5000)
        self.assertEqual(len(explain_clause('y' * 9000)), MAX_EXPLANATION_CHARS)
        sent = post.call_args.kwargs['json']['messages'][1]['content']
        self.assertLess(len(sent), MAX_CLAUSE_CHARS_SENT + 100)

    @override_settings(GROQ_API_KEY='test-key')
    @patch('analyzer.explain.requests.post')
    def test_the_key_and_clause_are_never_logged(self, post):
        import requests as real_requests
        from .explain import explain_clause
        post.side_effect = real_requests.ConnectionError('boom')
        with self.assertLogs('analyzer.explain', level='WARNING') as logs:
            explain_clause('SECRET CLAUSE TEXT')
        joined = ' '.join(logs.output)
        self.assertNotIn('test-key', joined)
        self.assertNotIn('SECRET CLAUSE TEXT', joined)


class AddExplanationsTests(TestCase):
    def _clauses(self, n=4):
        return [_risky(i, 0.99 - i * 0.05) for i in range(n)]

    @override_settings(GROQ_API_KEY='test-key')
    @patch('analyzer.explain.requests.post')
    def test_not_requested_sends_nothing_and_adds_nothing(self, post):
        from .explain import add_explanations
        clauses = self._clauses()
        self.assertEqual(add_explanations(clauses, requested=False), 'not_requested')
        post.assert_not_called()
        self.assertTrue(all('explanation' not in c for c in clauses))

    @override_settings(GROQ_API_KEY='')
    @patch('analyzer.explain.requests.post')
    def test_requested_without_a_key_is_unavailable_and_sends_nothing(self, post):
        from .explain import add_explanations
        self.assertEqual(add_explanations(self._clauses(), requested=True), 'unavailable')
        post.assert_not_called()

    @override_settings(GROQ_API_KEY='test-key')
    @patch('analyzer.explain.requests.post')
    def test_no_risky_clauses_means_nothing_to_explain(self, post):
        from .explain import add_explanations
        self.assertEqual(add_explanations([], requested=True), 'nothing_to_explain')
        post.assert_not_called()

    @override_settings(GROQ_API_KEY='test-key', EXPLAIN_MAX_CLAUSES=2)
    @patch('analyzer.explain.requests.post')
    def test_only_the_top_clauses_are_explained(self, post):
        from .explain import add_explanations
        post.return_value = _groq_ok('Because.')
        clauses = self._clauses(4)
        self.assertEqual(add_explanations(clauses, requested=True), 'ok')
        self.assertEqual(post.call_count, 2)
        self.assertEqual([c['explanation'] for c in clauses], ['Because.', 'Because.', None, None])
        self.assertEqual([c['explanation_source'] for c in clauses], ['groq', 'groq', None, None])

    @override_settings(GROQ_API_KEY='test-key', EXPLAIN_MAX_CLAUSES=3)
    @patch('analyzer.explain.requests.post')
    def test_one_failed_call_gives_a_partial_result_not_an_error(self, post):
        import requests as real_requests
        from .explain import add_explanations

        def fake_post(url, **kwargs):
            if 'clause 1' in kwargs['json']['messages'][1]['content']:
                raise real_requests.ConnectionError('down')
            return _groq_ok('Because.')

        post.side_effect = fake_post
        clauses = self._clauses(3)
        self.assertEqual(add_explanations(clauses, requested=True), 'partial')
        self.assertEqual([c['explanation'] for c in clauses], ['Because.', None, 'Because.'])

    @override_settings(GROQ_API_KEY='test-key')
    @patch('analyzer.explain.requests.post')
    def test_all_calls_failing_is_reported_as_failed(self, post):
        import requests as real_requests
        from .explain import add_explanations
        post.side_effect = real_requests.ConnectionError('down')
        clauses = self._clauses(2)
        self.assertEqual(add_explanations(clauses, requested=True), 'failed')
        self.assertTrue(all(c['explanation'] is None for c in clauses))


class ExplainEndpointTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='alice', password='pw')
        self.client.force_authenticate(user=self.user)
        self.url = reverse('document-analyze')
        self.analysis = {
            'total_clauses': 4, 'risky_count': 3, 'processing_time_ms': 1.0, 'all_clauses': [],
            'risky_clauses': [_risky(0, 0.7), _risky(1, 0.95), _risky(2, 0.8)],
        }

    def _post(self, body, **kwargs):
        with patch('analyzer.views.get_analyzer') as mock_get:
            analyzer = mock_get.return_value
            analyzer.extract_clauses.return_value = ['a', 'b', 'c', 'd']
            analyzer.analyze_clauses.return_value = self.analysis
            return self.client.post(self.url, {'text': 'x' * 40, **body}, **kwargs)

    @override_settings(GROQ_API_KEY='test-key')
    @patch('analyzer.explain.requests.post')
    def test_default_request_sends_nothing_to_groq(self, post):
        response = self._post({}, format='json')
        post.assert_not_called()
        self.assertEqual(response.data['explanations_status'], 'not_requested')
        self.assertTrue(all('explanation' not in c for c in response.data['risky_clauses']))

    @override_settings(GROQ_API_KEY='test-key')
    @patch('analyzer.explain.requests.post')
    def test_explain_true_adds_explanations_most_risky_first(self, post):
        post.return_value = _groq_ok('Because.')
        response = self._post({'explain': True}, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['explanations_status'], 'ok')
        shown = response.data['risky_clauses']
        self.assertEqual([c['clause_id'] for c in shown], [1, 2, 0])
        self.assertTrue(all(c['explanation'] == 'Because.' for c in shown))

    @override_settings(GROQ_API_KEY='test-key')
    @patch('analyzer.explain.requests.post')
    def test_the_api_key_never_appears_in_a_response(self, post):
        import json
        post.return_value = _groq_ok('Because.')
        response = self._post({'explain': True}, format='json')
        self.assertNotIn('test-key', json.dumps(response.data))

    @override_settings(GROQ_API_KEY='')
    @patch('analyzer.explain.requests.post')
    def test_explain_without_a_server_key_is_reported_unavailable(self, post):
        response = self._post({'explain': True}, format='json')
        post.assert_not_called()
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['explanations_status'], 'unavailable')

    @override_settings(GROQ_API_KEY='test-key')
    @patch('analyzer.explain.requests.post')
    def test_only_a_clear_true_turns_explanations_on(self, post):
        for value in (False, 'false', 'no', None, 0, '', 'maybe'):
            with self.subTest(explain=value):
                self._post({'explain': value}, format='json')
        post.assert_not_called()

    @override_settings(GROQ_API_KEY='test-key')
    @patch('analyzer.explain.requests.post')
    def test_a_form_string_true_works_as_the_upload_page_sends_it(self, post):
        post.return_value = _groq_ok('Because.')
        response = self._post({'explain': 'true'})  # multipart form, like /upload
        self.assertEqual(response.data['explanations_status'], 'ok')

    @override_settings(GROQ_API_KEY='test-key')
    @patch('analyzer.explain.requests.post')
    def test_saved_results_never_contain_the_explanations(self, post):
        post.return_value = _groq_ok('Because.')
        response = self._post({'explain': True, 'save': True}, format='json')
        self.assertEqual(response.data['explanations_status'], 'ok')
        saved = AnalysisResult.objects.get(document_id=response.data['document_id'])
        self.assertTrue(all('explanation' not in c for c in saved.analysis['risky_clauses']))

    @override_settings(GROQ_API_KEY='test-key')
    @patch('analyzer.explain.requests.post')
    def test_a_groq_outage_still_returns_the_classification(self, post):
        import requests as real_requests
        post.side_effect = real_requests.ConnectionError('down')
        response = self._post({'explain': True}, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['explanations_status'], 'failed')
        self.assertEqual(response.data['risky_count'], 3)
        self.assertEqual(len(response.data['risky_clauses']), 3)


class IndexExplainAvailabilityTests(TestCase):
    @override_settings(GROQ_API_KEY='')
    def test_page_knows_explanations_are_unavailable_without_a_key(self):
        self.assertFalse(self.client.get(reverse('index')).context['explain_available'])

    @override_settings(GROQ_API_KEY='test-key')
    def test_page_knows_explanations_are_available_with_a_key(self):
        self.assertTrue(self.client.get(reverse('index')).context['explain_available'])
