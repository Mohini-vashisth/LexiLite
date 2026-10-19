#!/usr/bin/env python3
"""
Measures the effect of the confidence gate (analyzer/confidence.py) on the
test split of train_model/data/legal_clauses_labeled.csv, across a range of
min_confidence thresholds.

For each threshold it reports:
  - coverage: share of clauses the model answers (the rest go to 'uncertain')
  - error rate among answered clauses, vs. the no-gate baseline (threshold 0.5,
    i.e. plain argmax)
  - risky recall: share of truly risky clauses that are answered AND labeled
    risky. This is the cost to watch for a risk analyzer: raising the bar
    lowers errors but pushes real risky clauses into 'uncertain'.

Only min_confidence is swept. The margin gate was removed from the defaults
(LEX-13): for this binary classifier margin = 2p - 1, so it is just a
confidence threshold under another name.

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

    y = np.array(y_test)
    n = len(y)
    n_risky = int((y == "risky").sum())
    print("\n" + "=" * 78)
    print("CONFIDENCE GATE - MEASURED TRADEOFF (test set: %d clauses, %d risky)" % (n, n_risky))
    print("=" * 78)
    print(f"{'min_conf':>8} {'answered':>9} {'coverage':>9} {'error (answered)':>17} "
          f"{'uncertain':>10} {'risky recall':>13}")

    baseline_error = None
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        scored = ConfidenceScorer(min_confidence=threshold).score(probs, class_labels)
        accepted = np.array([r["accepted"] for r in scored])
        labels = np.array([r["raw_label"] for r in scored])
        answered = int(accepted.sum())
        error = float((labels[accepted] != y[accepted]).mean()) if answered else 0.0
        recall = float(((labels == "risky") & (y == "risky") & accepted).sum() / n_risky)
        if baseline_error is None:
            baseline_error = error
        marker = "  <- current default" if abs(threshold - 0.65) < 1e-9 else ""
        print(f"{threshold:8.2f} {answered:9d} {answered / n:8.1%} {error:17.2%} "
              f"{n - answered:10d} {recall:12.1%}{marker}")

    print("=" * 78)
    print(f"Baseline (no gate, plain argmax) error rate: {baseline_error:.2%}")
    print("Relative error reduction = (baseline - gated error) / baseline, among answered clauses.")
    print("This is a coverage/recall tradeoff, not a free win: 'uncertain' clauses need a")
    print("fallback (secondary model or human review) or they simply go unclassified.")


if __name__ == "__main__":
    main()
