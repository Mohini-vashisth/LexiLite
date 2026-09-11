"""
Confidence scoring and fallback logic for clause risk classification.

Problem: a plain argmax on the classifier's probability output commits to a
label even when the model is barely more sure than a coin flip. On a dataset
skewed 85/15 safe/risky (see train_model/data/legal_clauses_labeled.csv),
that means low-margin "risky" calls are disproportionately wrong.

Approach: LogisticRegression already gives calibrated-ish probabilities via
predict_proba (that's what it optimizes). We threshold on:
  1. The winning class's probability (raw confidence) — reject if it's not
     comfortably above the majority-class base rate.
  2. The margin between the top two classes — reject if the model is nearly
     torn (e.g. 0.52 vs 0.48), regardless of the absolute probability.

Below both thresholds, don't force a hard label. Route the clause to an
'uncertain' bucket that the caller can escalate to a human reviewer or a
stronger model (RoBERTa) rather than silently emitting a possibly-wrong
high-confidence-looking answer.
"""

import numpy as np


class ConfidenceScorer:
    """
    Wraps a fitted classifier's predict_proba output with a confidence gate.

    min_confidence: reject predictions where the winning class probability
        is below this. Default 0.65 — comfortably above the 0.5 coin-flip
        line and above the ~0.15 base rate of the minority ('risky') class,
        so a low-confidence 'risky' call (model close to uncertain) doesn't
        get reported as if it were solid.
    min_margin: reject predictions where (top prob - second prob) is below
        this. Catches cases where min_confidence alone would still pass
        (e.g. 0.66 vs 0.34 is fine, but 0.51 vs 0.49 should never pass even
        if min_confidence were set low).
    """

    def __init__(self, min_confidence=0.65, min_margin=0.15):
        self.min_confidence = min_confidence
        self.min_margin = min_margin

    def score(self, probabilities, class_labels):
        """
        Args:
            probabilities: 2D array (n_samples, n_classes) from predict_proba
            class_labels: classifier.classes_ (same order as probabilities columns)

        Returns: list of dicts, one per sample:
            {
              'label': predicted class (str) or 'uncertain',
              'raw_label': the argmax class regardless of confidence gate,
              'confidence': float, winning class probability,
              'margin': float, gap to runner-up,
              'accepted': bool, whether the confidence gate passed,
            }
        """
        results = []
        for row in probabilities:
            order = np.argsort(row)[::-1]  # descending
            top_idx, second_idx = order[0], order[1] if len(order) > 1 else order[0]
            top_prob = float(row[top_idx])
            second_prob = float(row[second_idx])
            margin = top_prob - second_prob
            raw_label = class_labels[top_idx]

            accepted = top_prob >= self.min_confidence and margin >= self.min_margin

            results.append({
                'label': raw_label if accepted else 'uncertain',
                'raw_label': raw_label,
                'confidence': round(top_prob, 4),
                'margin': round(margin, 4),
                'accepted': accepted,
            })
        return results


class FallbackRouter:
    """
    Decides what to do with a clause the primary model wasn't confident about.

    Strategy (in order, first available wins):
      1. If a stronger/secondary model is configured, re-score with it and
         accept its answer if *it* clears the confidence gate.
      2. Otherwise mark the clause for human review — never silently emit
         the primary model's rejected guess as if it were confident.

    This module doesn't hard-depend on a specific secondary model; callers
    pass a `secondary_predict_fn(text) -> (label, confidence)` callable, e.g.
    backed by the RoBERTa classifier in train_model/saved_model_roberta/.
    """

    def __init__(self, scorer: ConfidenceScorer, secondary_predict_fn=None):
        self.scorer = scorer
        self.secondary_predict_fn = secondary_predict_fn

    def resolve(self, clause_text, primary_result):
        """
        Args:
            clause_text: the clause string (for secondary model re-scoring)
            primary_result: one entry from ConfidenceScorer.score()

        Returns: dict with the final decision:
            {
              'label': final label ('safe' | 'risky' | 'needs_review'),
              'source': 'primary' | 'secondary' | 'human_review',
              'confidence': float,
            }
        """
        if primary_result['accepted']:
            return {
                'label': primary_result['label'],
                'source': 'primary',
                'confidence': primary_result['confidence'],
            }

        if self.secondary_predict_fn is not None:
            try:
                label, confidence = self.secondary_predict_fn(clause_text)
                if confidence >= self.scorer.min_confidence:
                    return {'label': label, 'source': 'secondary', 'confidence': round(float(confidence), 4)}
            except Exception:
                pass  # fall through to human review — never crash the request over a fallback failure

        return {
            'label': 'needs_review',
            'source': 'human_review',
            'confidence': primary_result['confidence'],
        }


def apply_confidence_pipeline(clauses, probabilities, class_labels, scorer=None, fallback_router=None):
    """
    Convenience wrapper: score every clause and resolve fallbacks in one pass.
    Returns a list of per-clause dicts merging clause text with the final decision.
    """
    scorer = scorer or ConfidenceScorer()
    fallback_router = fallback_router or FallbackRouter(scorer)

    raw_scores = scorer.score(probabilities, class_labels)

    output = []
    for clause, raw in zip(clauses, raw_scores):
        decision = fallback_router.resolve(clause, raw)
        output.append({
            'text': clause,
            'label': decision['label'],
            'source': decision['source'],
            'confidence': decision['confidence'],
            'raw_confidence': raw['confidence'],
            'raw_margin': raw['margin'],
        })
    return output
