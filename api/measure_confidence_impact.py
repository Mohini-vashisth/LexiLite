#!/usr/bin/env python3
"""
Measures the actual effect of confidence gating (analyzer/confidence.py)
on the test split of train_model/data/legal_clauses_labeled.csv.

Defines "incorrect high-confidence output" precisely, since that's the
number that ends up on a resume and needs to survive scrutiny:

  BEFORE (no gating): among predictions where the raw argmax probability
  is >= min_confidence, what fraction are wrong vs. ground truth?

  AFTER (with gating): among predictions the ConfidenceScorer actually
  *accepts* (passes both the confidence AND margin thresholds), what
  fraction are wrong?

The claimed ~30% reduction is (before_error_rate - after_error_rate) /
before_error_rate on that "confidently answered" subset — NOT overall
accuracy, and NOT free: gating trades some coverage (more clauses land in
'uncertain') for a lower error rate among the ones it does answer. This
script reports both so the tradeoff is visible, not hidden.

Run this on a machine/container where `import torch` and
`import sentence_transformers` complete normally.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

from analyzer.confidence import ConfidenceScorer

DATA_PATH = os.path.join("..", "train_model", "data", "legal_clauses_labeled.csv")
MODEL_PATH = os.path.join("..", "train_model", "saved_model", "clause_classifier.pkl")


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH).rename(columns={"label": "risk"})
    df.drop_duplicates(subset=["clause"], inplace=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    X_texts = df["clause"].tolist()
    y_labels = df["risk"].tolist()
    _, X_test, _, y_test = train_test_split(X_texts, y_labels, test_size=0.2, random_state=42)

    print(f"Test set: {len(X_test)} clauses")

    print("Encoding with Sentence-BERT...")
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    X_test_vec = sbert.encode(X_test, show_progress_bar=True)

    clf = joblib.load(MODEL_PATH)
    probs = clf.predict_proba(X_test_vec)
    class_labels = clf.classes_

    scorer = ConfidenceScorer(min_confidence=0.65, min_margin=0.15)
    scored = scorer.score(probs, class_labels)

    # --- BEFORE: plain argmax gated only by min_confidence (no margin check) ---
    before_confident = [
        (s, truth) for s, truth in zip(scored, y_test)
        if s['confidence'] >= scorer.min_confidence
    ]
    before_wrong = sum(1 for s, truth in before_confident if s['raw_label'] != truth)
    before_rate = before_wrong / len(before_confident) if before_confident else 0.0

    # --- AFTER: full gate (confidence AND margin) ---
    after_confident = [(s, truth) for s, truth in zip(scored, y_test) if s['accepted']]
    after_wrong = sum(1 for s, truth in after_confident if s['label'] != truth)
    after_rate = after_wrong / len(after_confident) if after_confident else 0.0

    uncertain_count = len(scored) - len(after_confident)

    print("\n" + "=" * 60)
    print("CONFIDENCE GATING — MEASURED IMPACT")
    print("=" * 60)
    print(f"Total test clauses:                 {len(scored)}")
    print()
    print(f"BEFORE (confidence-only threshold):")
    print(f"  Confidently answered:              {len(before_confident)}")
    print(f"  Wrong among those:                 {before_wrong}")
    print(f"  Error rate:                        {before_rate:.1%}")
    print()
    print(f"AFTER (confidence + margin gate):")
    print(f"  Confidently answered:              {len(after_confident)}")
    print(f"  Wrong among those:                 {after_wrong}")
    print(f"  Error rate:                        {after_rate:.1%}")
    print(f"  Routed to 'uncertain' (no longer high-confidence): {uncertain_count} "
          f"({uncertain_count/len(scored):.1%} of test set)")
    print()

    if before_rate > 0:
        reduction = (before_rate - after_rate) / before_rate
        print(f"REDUCTION in high-confidence error rate: {reduction:.1%}")
    else:
        print("No baseline errors to compare against — check thresholds/data.")

    print("=" * 60)
    print("\nNote: this is a coverage/precision tradeoff, not a free win —")
    print("the 'uncertain' bucket needs a fallback (secondary model or human")
    print("review) or those clauses simply don't get classified.")


if __name__ == "__main__":
    main()
