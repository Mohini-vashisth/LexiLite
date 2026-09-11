import os
import time
import logging
import joblib
import numpy as np
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# sentence_transformers (and the torch it pulls in) is imported lazily, inside
# ModelCache._load_models, not here at module top level. This module is
# reachable just by importing analyzer.views -> analyzer.inference, which
# Django's URLconf system check does on every manage.py command (migrate,
# check, runserver, ...). A top-level `import torch` deadlocks on this
# machine's fork/mutex handling the moment the module loads — long before any
# model is actually needed — so `manage.py migrate` never returned. Deferring
# the import to first real use means commands that don't need inference never
# pay that cost.

class ModelCache:
    """Singleton for cached model instances to avoid reloading"""
    _instance = None
    _sbert_model = None
    _sbert_clf = None
    _executor = ThreadPoolExecutor(max_workers=4)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._load_models()
        self._initialized = True

    @classmethod
    def is_loaded(cls):
        """Report load state WITHOUT constructing/loading anything. A health
        check must never have the side effect of triggering a fresh model
        load — instantiating ModelCache() to check this would do exactly
        that on the first request after startup, turning a status probe
        into a multi-second (or, if the load fails, hanging) operation."""
        return cls._instance is not None and cls._instance._sbert_model is not None

    def _load_models(self):
        """Load models once at startup"""
        try:
            from sentence_transformers import SentenceTransformer  # see module docstring note above

            logger.info("Loading Sentence-BERT model...")
            self._sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✓ Sentence-BERT model loaded")

            # settings.TRAIN_MODEL_DIR is an absolute path resolved relative
            # to this codebase, not the process's cwd — using a bare
            # 'train_model/...' relative path here meant the classifier was
            # never found when the server was launched the normal way (from
            # inside api/), silently leaving self._sbert_clf as None and
            # making every /analyze call fail with "Models not loaded".
            from django.conf import settings
            clf_path = os.path.join(settings.TRAIN_MODEL_DIR, 'saved_model', 'clause_classifier.pkl')
            if os.path.exists(clf_path):
                self._sbert_clf = joblib.load(clf_path)
                logger.info(f"✓ SBERT classifier loaded from {clf_path}")
            else:
                logger.warning(f"SBERT classifier not found at {clf_path} — /analyze will fail until it exists")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def get_sbert_model(self):
        return self._sbert_model

    def get_sbert_classifier(self):
        return self._sbert_clf


class LegalAnalyzer:
    """High-performance legal document analyzer"""

    def __init__(self, model_choice='sbert'):
        self.model_cache = ModelCache()
        self.model_choice = model_choice
        self.metrics = {
            'total_requests': 0,
            'total_time_ms': 0,
            'avg_latency_ms': 0,
            'max_latency_ms': 0,
        }

    def extract_clauses(self, text):
        """Extract clauses from text by splitting on common legal delimiters"""
        delimiters = ['\n\n', '\n', '  ']
        clauses = []

        for delimiter in delimiters:
            if delimiter in text:
                clauses = text.split(delimiter)
                break

        # Filter empty clauses and strip whitespace
        clauses = [c.strip() for c in clauses if c.strip()]
        return clauses[:100]  # Limit to 100 clauses per document

    def analyze_clauses(self, clauses):
        """Analyze clauses for risk - optimized for latency"""
        start_time = time.time()

        sbert_model = self.model_cache.get_sbert_model()
        sbert_clf = self.model_cache.get_sbert_classifier()

        if not sbert_model or not sbert_clf:
            raise ValueError("Models not loaded")

        # Batch encode for better performance
        embeddings = sbert_model.encode(
            clauses,
            show_progress_bar=False,
            batch_size=32,
            convert_to_numpy=True
        )

        # Predict risk
        predictions = sbert_clf.predict(embeddings)
        probabilities = sbert_clf.predict_proba(embeddings)

        # predict_proba's columns are ordered by clf.classes_, NOT
        # guaranteed to be [safe, risky] — scikit-learn sorts class labels
        # alphabetically, and 'risky' < 'safe' alphabetically, so column 0
        # is actually P(risky) and column 1 is P(safe) for this classifier.
        # A previous version of this code hardcoded probs[0]=safe_prob,
        # probs[1]=risky_prob, which silently swapped the two: clauses
        # correctly labeled "risky" were reported with a 99%+ "safe_prob"
        # alongside them. Confirmed and fixed 2026-09-02 (LEX-11) by reading
        # the classifier's own class order instead of assuming one.
        class_labels = list(sbert_clf.classes_)
        safe_idx = class_labels.index('safe')
        risky_idx = class_labels.index('risky')

        # Build results
        results = []
        for i, (clause, pred, probs) in enumerate(zip(clauses, predictions, probabilities)):
            results.append({
                'clause_id': i,
                'text': clause[:200],  # Truncate for response
                'risk': pred,
                'confidence': float(np.max(probs)),
                'safe_prob': float(probs[safe_idx]),
                'risky_prob': float(probs[risky_idx]),
            })

        elapsed_ms = (time.time() - start_time) * 1000
        self._update_metrics(elapsed_ms)

        return {
            'risky_clauses': [r for r in results if r['risk'] == 'risky'],
            'all_clauses': results,
            'processing_time_ms': elapsed_ms,
            'total_clauses': len(clauses),
            'risky_count': sum(1 for r in results if r['risk'] == 'risky'),
        }

    def _update_metrics(self, latency_ms):
        """Track performance metrics"""
        self.metrics['total_requests'] += 1
        self.metrics['total_time_ms'] += latency_ms
        self.metrics['avg_latency_ms'] = self.metrics['total_time_ms'] / self.metrics['total_requests']
        self.metrics['max_latency_ms'] = max(self.metrics['max_latency_ms'], latency_ms)

    def get_metrics(self):
        """Return performance metrics"""
        return {
            'total_requests': self.metrics['total_requests'],
            'avg_latency_ms': round(self.metrics['avg_latency_ms'], 2),
            'max_latency_ms': round(self.metrics['max_latency_ms'], 2),
            'p50_latency_ms': 'See monitoring dashboard',
        }
