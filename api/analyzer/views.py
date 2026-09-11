import logging
import uuid
import time
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from django.core.cache import cache
from .models import AnalysisResult
from .serializers import AnalysisResultSerializer
from .inference import LegalAnalyzer

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


class DocumentAnalysisViewSet(viewsets.ModelViewSet):
    """API endpoint for document analysis"""
    queryset = AnalysisResult.objects.all()
    serializer_class = AnalysisResultSerializer

    @action(detail=False, methods=['post'])
    def analyze(self, request):
        """
        Analyze a legal document for risky clauses.
        POST body: {"text": "document text", "filename": "optional.pdf"}
        """
        start_time = time.time()

        try:
            text = request.data.get('text', '')
            filename = request.data.get('filename', 'unnamed_document')

            if not text or len(text) < 10:
                return Response(
                    {'error': 'Document text required (min 10 characters)'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            document_id = str(uuid.uuid4())
            logger.info(f"Analyzing document {document_id} ({len(text)} chars)")

            # Extract clauses
            clauses = get_analyzer().extract_clauses(text)

            if not clauses:
                return Response(
                    {'error': 'No clauses extracted from document'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Analyze
            analysis = get_analyzer().analyze_clauses(clauses)

            # Store result
            result = AnalysisResult.objects.create(
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
                f"{analysis['risky_count']} risky, {total_time_ms:.1f}ms"
            )

            return Response({
                'document_id': document_id,
                'filename': filename,
                'total_clauses': analysis['total_clauses'],
                'risky_count': analysis['risky_count'],
                'risky_clauses': analysis['risky_clauses'][:10],  # Top 10
                'processing_time_ms': round(analysis['processing_time_ms'], 2),
                'api_response_time_ms': round(total_time_ms, 2),
            }, status=status.HTTP_201_CREATED)

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['get'])
    def recent(self, request):
        """Get recent analysis results"""
        limit = int(request.query_params.get('limit', 10))
        results = AnalysisResult.objects.all()[:limit]
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
    """Health check endpoint - used by load balancers"""

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
    """Metrics endpoint for monitoring performance"""

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
