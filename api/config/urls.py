from django.urls import path, include
from django.views.generic import TemplateView
from rest_framework.routers import DefaultRouter
from analyzer.views import DocumentAnalysisViewSet, HealthCheckView, MetricsView

router = DefaultRouter()
router.register(r'documents', DocumentAnalysisViewSet, basename='document')

urlpatterns = [
    path('', TemplateView.as_view(template_name='index.html'), name='index'),
    path('api/', include(router.urls)),
    path('api/health/', HealthCheckView.as_view(), name='health'),
    path('api/metrics/', MetricsView.as_view(), name='metrics'),
]
