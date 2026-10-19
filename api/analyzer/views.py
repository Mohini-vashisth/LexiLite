import logging
import uuid
import time
from datetime import timedelta
from django.conf import settings
from django.utils import timezone
from django.views.generic import TemplateView
from rest_framework import mixins, viewsets, status
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
from .document_parsers import extract_text_from_upload, UnsupportedFileType, DocumentParseError
from .models import AnalysisResult
from .serializers import AnalysisResultSerializer
from .inference import LegalAnalyzer
from .ip_review import review_ip_clauses
from .report import build_full_report, ReportUnavailable

logger = logging.getLogger(__name__)

_analyzer = None


def get_analyzer():
    """Lazy singleton — importing this module (e.g. via urls.py, on any
    manage.py command) must not force a model load. Only actually
    instantiate LegalAnalyzer the first time a request needs it."""
    global _analyzer
    if _analyzer is None:
        _analyzer = LegalAnalyzer()
    return _analyzer


MAX_FILENAME_LENGTH = 255  # AnalysisResult.filename column size


def _wants_save(value):
    """`save` arrives as a JSON boolean from /analyze and as a form string
    ('true'/'false') from /upload. Anything that isn't clearly true means
    don't save: storage is opt-in, so a missing or odd value must not
    create a record."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('true', '1', 'yes')


def _run_analysis(text, filename, owner, save=False):
    """
    Shared by both the JSON-text endpoint (analyze) and the file-upload
    endpoint (upload) — everything after "we have plain text" is identical
    regardless of where that text came from, so it lives in one place
    rather than being copy-pasted between the two views.

    By default nothing is written to the database: the contract is read,
    analysed and the result returned. A record is created only when the
    caller passes save=True, and only then is a document_id returned to use
    for later retrieval or deletion.

    Returns a DRF Response ready to hand back to the caller.
    """
    start_time = time.time()
    # LEX-9: filename is client-supplied; an over-long one would overflow the
    # column (500 on Postgres), and a non-string would blow up later.
    filename = str(filename)[:MAX_FILENAME_LENGTH]

    try:
        if not isinstance(text, str) or len(text) < 10:
            return Response(
                {'error': 'Document text required (min 10 characters)'},
                status=status.HTTP_400_BAD_REQUEST
            )

        clauses = get_analyzer().extract_clauses(text)

        if not clauses:
            return Response(
                {'error': 'No clauses extracted from document'},
                status=status.HTTP_400_BAD_REQUEST
            )

        analysis = get_analyzer().analyze_clauses(clauses)
        ip_review = review_ip_clauses(clauses)

        document_id = None
        if save:
            document_id = str(uuid.uuid4())
            AnalysisResult.objects.create(
                owner=owner,
                document_id=document_id,
                filename=filename,
                clauses=clauses,
                analysis=analysis,
                processing_time_ms=analysis['processing_time_ms'],
                model_used='sbert',
            )

        total_time_ms = (time.time() - start_time) * 1000
        logger.info(
            f"✓ Analysis complete: {len(clauses)} clauses, "
            f"{analysis['risky_count']} risky, saved={save}, {total_time_ms:.1f}ms"
        )

        response_data = {
            'saved': save,
            'filename': filename,
            'ip_review': ip_review,
            'total_clauses': analysis['total_clauses'],
            'risky_count': analysis['risky_count'],
            'risky_clauses': analysis['risky_clauses'][:10],  # Top 10
            'processing_time_ms': round(analysis['processing_time_ms'], 2),
            'api_response_time_ms': round(total_time_ms, 2),
        }
        if document_id is not None:
            response_data['document_id'] = document_id
        return Response(response_data, status=status.HTTP_201_CREATED)

    except Exception:
        # LEX-9: log the details server-side, but never echo str(e) to the
        # client — exception text can include file paths, SQL, or internals.
        logger.exception("Analysis error")
        return Response(
            {'error': 'Internal error while analyzing the document'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class DocumentAnalysisViewSet(
    mixins.RetrieveModelMixin,
    mixins.ListModelMixin,
    mixins.DestroyModelMixin,
    viewsets.GenericViewSet,
):
    """API endpoint for document analysis.

    Deliberately NOT a full ModelViewSet — a record is created only as a
    side effect of `analyze`/`upload` with save=true (which set `owner`
    server-side and populate `analysis` from a real model run), never
    directly. Without save=true, nothing is stored at all. A raw
    `POST /api/documents/` via the generic create action would bypass all
    of that: `owner` isn't in the serializer, so it'd fail with a raw
    IntegrityError (500) instead of anything sensible. Same reasoning
    against update — there's no legitimate reason to hand-edit a result.
    Delete IS supported (LEX-9 data retention): users can delete their own
    analyses, one at a time or all at once. get_queryset() scopes everything
    to request.user, so someone else's record is a 404, never deletable.
    """
    serializer_class = AnalysisResultSerializer

    def get_queryset(self):
        """Scoped to the requesting user (LEX-6) — this is what /recent/
        and DRF's other ModelViewSet actions (retrieve, list, etc.) read
        from, so a user can never see another user's analyses regardless
        of which action they hit.

        Also hides analyses past RETENTION_DAYS on read, so expiry does not
        depend on the purge command having run (see DATA_POLICY.md)."""
        queryset = AnalysisResult.objects.filter(owner=self.request.user)
        if settings.RETENTION_DAYS > 0:
            cutoff = timezone.now() - timedelta(days=settings.RETENTION_DAYS)
            queryset = queryset.filter(created_at__gte=cutoff)
        return queryset

    @action(detail=False, methods=['post'])
    def analyze(self, request):
        """
        Analyze a legal document for risky clauses.
        POST body: {"text": "document text", "filename": "optional.pdf", "save": false}
        Nothing is stored unless "save" is true.
        """
        text = request.data.get('text', '')
        filename = request.data.get('filename', 'unnamed_document')
        save = _wants_save(request.data.get('save', False))
        return _run_analysis(text, filename, owner=request.user, save=save)

    @action(detail=False, methods=['post'])
    def upload(self, request):
        """
        Analyze an uploaded PDF or DOCX file for risky clauses.
        POST multipart/form-data with a `file` field and optional `save=true`.
        Returns the same response shape as /analyze. Nothing is stored unless
        `save` is true.
        """
        uploaded_file = request.FILES.get('file')
        if uploaded_file is None:
            return Response(
                {'error': "No file provided — send it as multipart/form-data under the 'file' field"},
                status=status.HTTP_400_BAD_REQUEST
            )

        if uploaded_file.size > settings.MAX_UPLOAD_SIZE_BYTES:
            max_mb = settings.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
            return Response(
                {'error': f'File too large ({uploaded_file.size / (1024 * 1024):.1f}MB) — max {max_mb:.0f}MB'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            text = extract_text_from_upload(uploaded_file)
        except UnsupportedFileType as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except DocumentParseError as e:
            logger.warning(f"Failed to parse uploaded file '{uploaded_file.name}': {e}")
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

        save = _wants_save(request.data.get('save', False))
        return _run_analysis(text, uploaded_file.name, owner=request.user, save=save)

    @action(detail=False, methods=['post'])
    def report(self, request):
        """
        The paid full report: the IP review, with each flag explained in plain
        language by the self-hosted model. Paid users only.
        POST body: {"text": "document text"}. Nothing is stored.
        """
        if not request.user.groups.filter(name=settings.PAID_REPORT_GROUP).exists():
            return Response(
                {'error': 'The full report is available on the paid plan.'},
                status=status.HTTP_403_FORBIDDEN,
            )
        text = request.data.get('text', '')
        if not isinstance(text, str) or len(text) < 10:
            return Response(
                {'error': 'Document text required (min 10 characters)'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        clauses = get_analyzer().extract_clauses(text)
        if not clauses:
            return Response(
                {'error': 'No clauses extracted from document'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            report = build_full_report(review_ip_clauses(clauses))
        except ReportUnavailable as e:
            return Response({'error': str(e)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        return Response(report, status=status.HTTP_200_OK)

    @action(detail=False, methods=['delete'], url_path='delete-all')
    def delete_all(self, request):
        """Delete every analysis belonging to the caller (LEX-9: a user can
        remove all of their stored data in one call)."""
        deleted, _ = self.get_queryset().delete()
        logger.info(f"User {request.user.pk} deleted all their analyses ({deleted} rows)")
        return Response({'deleted': deleted})

    @action(detail=False, methods=['get'])
    def recent(self, request):
        """Get recent analysis results — the caller's own only, via
        get_queryset() (already scoped to request.user)."""
        # LEX-9: int() on user input crashed with a 500 on junk like
        # ?limit=abc, and an unbounded limit let one call pull the whole table.
        try:
            limit = int(request.query_params.get('limit', 10))
        except ValueError:
            return Response({'error': "'limit' must be an integer"}, status=status.HTTP_400_BAD_REQUEST)
        limit = max(1, min(limit, 100))
        results = self.get_queryset()[:limit]
        return Response({
            'count': len(results),
            'results': [
                {
                    'document_id': r.document_id,
                    'filename': r.filename,
                    'risky_count': r.analysis['risky_count'],
                    'created_at': r.created_at,
                }
                for r in results
            ]
        })


class HealthCheckView(APIView):
    """Health check endpoint - used by load balancers. Deliberately open —
    a load balancer/monitoring probe has no user to authenticate as, and
    the response reveals nothing sensitive (just up/down + model state)."""
    permission_classes = [AllowAny]

    def get(self, request):
        """Check API and model health"""
        try:
            from .inference import ModelCache
            models_loaded = ModelCache.is_loaded()

            return Response({
                'status': 'healthy' if models_loaded else 'degraded',
                'models_loaded': models_loaded,
                'timestamp': time.time(),
            }, status=status.HTTP_200_OK if models_loaded else status.HTTP_503_SERVICE_UNAVAILABLE)
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return Response(
                {'status': 'unhealthy', 'error': str(e)},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )


class MetricsView(APIView):
    """Metrics endpoint for monitoring performance. Deliberately open, same
    reasoning as HealthCheckView — aggregate operational numbers, nothing
    user-specific or sensitive."""
    permission_classes = [AllowAny]

    def get(self, request):
        """Get performance metrics. Must never trigger a model load as a
        side effect — an observability endpoint that can hang/slow-load the
        model on first hit is a correctness bug, not just a nuisance."""
        from .inference import ModelCache
        if not ModelCache.is_loaded():
            return Response({
                'total_requests': 0,
                'avg_latency_ms': 0,
                'max_latency_ms': 0,
                'message': 'No requests processed yet (models not loaded).',
            })

        metrics = get_analyzer().get_metrics()
        return Response({
            **metrics,
            'message': f"Average latency: {metrics['avg_latency_ms']:.1f}ms ({metrics['total_requests']} requests)"
        })


class IndexView(TemplateView):
    """The browser page. Adds the sample report, built by running the real
    IP review on the fixture contract, so the page's example can't drift
    from what the product outputs (see tests.SampleReportTests)."""
    template_name = 'index.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['sample_report'] = sample_report()
        return context


def sample_report():
    """Run the IP review on analyzer/fixtures/sample_contract.txt, split into
    paragraphs. Used for the page's example and checked by a test."""
    import os
    path = os.path.join(os.path.dirname(__file__), 'fixtures', 'sample_contract.txt')
    with open(path, encoding='utf-8') as f:
        clauses = [p.strip() for p in f.read().split('\n\n') if p.strip()]
    return review_ip_clauses(clauses)
