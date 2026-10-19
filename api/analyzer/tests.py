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
