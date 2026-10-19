from django.urls import path, include
from rest_framework.routers import DefaultRouter
from analyzer.views import DocumentAnalysisViewSet, HealthCheckView, IndexView, MetricsView

router = DefaultRouter()
router.register(r'documents', DocumentAnalysisViewSet, basename='document')

urlpatterns = [
    path('', IndexView.as_view(), name='index'),
    path('api/', include(router.urls)),
    path('api/health/', HealthCheckView.as_view(), name='health'),
    path('api/metrics/', MetricsView.as_view(), name='metrics'),
    # DRF's ready-made login/logout views (with default templates — no
    # custom template work needed). Gives the browser frontend somewhere
    # to log in: session cookie set here is what SessionAuthentication
    # recognizes on subsequent API calls from the page.
    path('api-auth/', include('rest_framework.urls')),
]
