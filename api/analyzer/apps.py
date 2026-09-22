import os
from django.apps import AppConfig


class AnalyzerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'analyzer'

    def ready(self):
        """Registers signals (always) and preloads models only for the
        actual server process, not for management commands (migrate,
        makemigrations, shell, etc.) — those don't need torch/
        sentence-transformers loaded and it was hanging `migrate` on this
        machine (fork/mutex issue with the ML libs)."""
        from . import signals  # noqa: F401 — registers auto-create-token; must run on every startup, not just runserver

        import sys
        if os.environ.get('LEXILITE_SKIP_MODEL_PRELOAD') == '1':
            return  # diagnostic escape hatch — verify routing/DB without paying the model-load cost
        if 'runserver' not in sys.argv and os.environ.get('GUNICORN_WORKER') != '1':
            return

        from .inference import ModelCache
        try:
            ModelCache()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Could not preload models: {e}")
