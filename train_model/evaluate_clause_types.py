#!/usr/bin/env python3
"""
Trains and scores the embedding-based clause-type detector, and a keyword
baseline, on the dataset built by build_clause_type_dataset.py.

What is compared: for each of the eight clause types, one yes/no question per
chunk ("is this chunk a <type> clause?"). A chunk can have several types, so
each type gets its own classifier (multi-label), exactly as the dataset is
built.

Three things make the numbers fair:
  - Train and test are CUAD's own contract-level split: no contract is on both
    sides, so the detector is scored on contracts it has never seen.
  - Every score is printed beside the number of test chunks that really are
    that type. Rare types (Uncapped Liability, Liquidated Damages) have few, so
    their scores are rough.
  - The same sample of test chunks (all chunks that have any of the eight
    types, plus a random set of the rest) is written to disk, so the LLM can be
    scored on exactly the same chunks (see evaluate_clause_types_llm.py).

Usage (from the repo root, in an environment with sentence-transformers and
scikit-learn installed):
    python3 train_model/evaluate_clause_types.py
Output (git-ignored): train_model/data/generated/eval_sample.jsonl,
embedding_predictions.json and a printed table.
"""
import argparse
import json
import os
import random
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(HERE, "data", "generated")
DATASET_PATH = os.path.join(GENERATED_DIR, "clause_types.jsonl")
ENCODER_NAME = "all-MiniLM-L6-v2"

TYPES = [
    "IP Ownership Assignment",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Termination For Convenience",
    "Non-Compete",
    "Exclusivity",
    "Irrevocable Or Perpetual License",
]

# The keyword baseline: the few words a non-expert would search for to find each
# type. This is the floor a learned model has to beat; it is deliberately simple.
KEYWORDS = {
    "IP Ownership Assignment": r"\bassign\w*\b.*\b(intellectual property|work product|copyright|rights)\b|\bwork[s]? made for hire\b|\bshall (be|remain) the (sole and exclusive )?property of\b|\bowned by\b",
    "Uncapped Liability": r"\bunlimited\b|\bwithout limit(ation)?\b|\bnot be limited\b|\bnot subject to (any )?(limit|cap)",
    "Cap On Liability": r"\bin no event\b|\bshall not exceed\b|\bnot exceed\b.*\b(fees|amounts?|paid)\b|\blimitation of liability\b|\baggregate liability\b",
    "Liquidated Damages": r"\bliquidated damages\b|\btermination fee\b|\bpenalt(y|ies)\b",
    "Termination For Convenience": r"\bterminate\b.*\b(without cause|for convenience|for any reason|at any time)\b|\b(without cause|for convenience)\b.*\bterminat",
    "Non-Compete": r"\bnon-?compet\w*\b|\bcompete\b|\bcompeting\b",
    "Exclusivity": r"\bexclusive(ly)?\b|\bsole provider\b|\bonly (supplier|distributor)\b",
    "Irrevocable Or Perpetual License": r"\birrevocabl\w*\b|\bperpetual(ly)?\b",
}
KEYWORD_RES = {t: re.compile(p, re.IGNORECASE | re.DOTALL) for t, p in KEYWORDS.items()}


def load_rows(path=DATASET_PATH):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def keyword_predict(text):
    """The set of types whose keywords appear in `text`."""
    return {t for t, rx in KEYWORD_RES.items() if rx.search(text)}


def make_sample(test_rows, negatives_per_positive=1.0, seed=13):
    """Every test chunk with at least one of the eight types, plus a random set
    of the others (about `negatives_per_positive` per positive chunk). A sample
    is used because running the LLM over every test chunk is slow and costly;
    the embedding scores are reported on both the sample and the full test set."""
    rng = random.Random(seed)
    positives = [r for r in test_rows if set(r["types"]) & set(TYPES)]
    others = [r for r in test_rows if not set(r["types"]) & set(TYPES)]
    chosen = rng.sample(others, min(len(others), int(len(positives) * negatives_per_positive)))
    sample = positives + chosen
    rng.shuffle(sample)
    return sample


def sample_weights(sample_rows, test_rows):
    """1 for every chunk that has one of the eight types; for the others, how
    many chunks of the full test set each sampled chunk stands for."""
    typed = lambda r: bool(set(r["types"]) & set(TYPES))
    others_full = sum(1 for r in test_rows if not typed(r))
    others_sample = sum(1 for r in sample_rows if not typed(r))
    factor = others_full / max(others_sample, 1)
    return [1.0 if typed(r) else factor for r in sample_rows]


def score(rows, predicted_sets, types=TYPES, weights=None):
    """Per-type precision, recall, F1 and the number of real positives, plus the
    macro average (every type counts equally, however rare).

    `weights` (one per row, default 1) lets a sample stand in for the full test
    set: the sample keeps every chunk with a type but only some of the others,
    so each kept "other" chunk is counted as several. Without it, precision on
    the sample is far too high. The positives count stays the real count."""
    weights = weights or [1.0] * len(rows)
    result = {}
    for t in types:
        tp = sum(w for r, p, w in zip(rows, predicted_sets, weights) if t in r["types"] and t in p)
        fp = sum(w for r, p, w in zip(rows, predicted_sets, weights) if t not in r["types"] and t in p)
        fn = sum(w for r, p, w in zip(rows, predicted_sets, weights) if t in r["types"] and t not in p)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[t] = {"precision": precision, "recall": recall, "f1": f1,
                     "positives": sum(1 for r in rows if t in r["types"]), "predicted": tp + fp}
    result["macro_f1"] = float(np.mean([result[t]["f1"] for t in types]))
    return result


def format_table(title, scores_by_method, types=TYPES):
    lines = [title, f"{'type':34} {'n':>5}  " + "  ".join(f"{m:>20}" for m in scores_by_method)]
    lines.append(f"{'':34} {'':>5}  " + "  ".join(f"{'P / R / F1':>20}" for _ in scores_by_method))
    for t in types:
        n = next(iter(scores_by_method.values()))[t]["positives"]
        cells = []
        for s in scores_by_method.values():
            c = s[t]
            cells.append(f"{c['precision']:.2f} / {c['recall']:.2f} / {c['f1']:.2f}".rjust(20))
        lines.append(f"{t:34} {n:>5}  " + "  ".join(cells))
    lines.append(f"{'macro F1':34} {'':>5}  " + "  ".join(f"{s['macro_f1']:>20.3f}" for s in scores_by_method.values()))
    return "\n".join(lines)


def embed(texts, cache_path):
    """Sentence-BERT vectors (all-MiniLM-L6-v2, frozen), cached on disk."""
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if len(cached) == len(texts):
            return cached
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(ENCODER_NAME)
    vectors = encoder.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    np.save(cache_path, vectors)
    return vectors


def train_detectors(train_vectors, train_rows, c_value):
    """One logistic regression per type. class_weight='balanced' because every
    type is rare (most chunks are not any of the eight)."""
    from sklearn.linear_model import LogisticRegression
    detectors = {}
    for t in TYPES:
        y = np.array([t in r["types"] for r in train_rows])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=c_value)
        clf.fit(train_vectors, y)
        detectors[t] = clf
    return detectors


def best_threshold(y_true, probabilities):
    """The probability cutoff with the best F1 on `y_true` (used on training
    data only)."""
    best, best_f1 = 0.5, -1.0
    for cutoff in np.concatenate([np.arange(0.05, 0.95, 0.025), [0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999]]):
        pred = probabilities >= cutoff
        tp = int(np.sum(pred & y_true))
        fp = int(np.sum(pred & ~y_true))
        fn = int(np.sum(~pred & y_true))
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best_f1:
            best, best_f1 = float(cutoff), f1
    return best


def tune_thresholds(train_vectors, train_rows, c_value, folds=5):
    """Pick each type's cutoff from out-of-fold predictions on the TRAINING
    contracts (folds never split a contract), so nothing is tuned on the test
    set."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    groups = np.array([r["contract"] for r in train_rows])
    thresholds = {}
    for t in TYPES:
        y = np.array([t in r["types"] for r in train_rows])
        oof = np.zeros(len(y))
        for fit_idx, val_idx in GroupKFold(n_splits=folds).split(train_vectors, y, groups):
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=c_value)
            clf.fit(train_vectors[fit_idx], y[fit_idx])
            oof[val_idx] = clf.predict_proba(train_vectors[val_idx])[:, 1]
        thresholds[t] = best_threshold(y, oof)
    return thresholds


def predict_sets(detectors, vectors, thresholds=None):
    thresholds = thresholds or {t: 0.5 for t in TYPES}
    flags = {t: detectors[t].predict_proba(vectors)[:, 1] >= thresholds[t] for t in TYPES}
    return [{t for t in TYPES if flags[t][i]} for i in range(len(vectors))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--out-dir", default=GENERATED_DIR)
    parser.add_argument("--cache-dir", default=None, help="where to keep embedding caches (default: out-dir)")
    parser.add_argument("--c", type=float, default=1.0, help="logistic regression C (default 1.0, no tuning on test)")
    args = parser.parse_args(argv)
    cache_dir = args.cache_dir or args.out_dir
    os.makedirs(cache_dir, exist_ok=True)

    rows = load_rows(args.dataset)
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    print(f"train chunks {len(train)} | test chunks {len(test)}")

    train_vectors = embed([r["text"] for r in train], os.path.join(cache_dir, "train_vectors.npy"))
    test_vectors = embed([r["text"] for r in test], os.path.join(cache_dir, "test_vectors.npy"))
    detectors = train_detectors(train_vectors, train, args.c)
    thresholds = tune_thresholds(train_vectors, train, args.c)
    print("cutoffs chosen on training contracts:", {t: round(v, 3) for t, v in thresholds.items()})
    embedding_pred = predict_sets(detectors, test_vectors, thresholds)
    untuned_pred = predict_sets(detectors, test_vectors)
    keyword_pred = [keyword_predict(r["text"]) for r in test]

    sample = make_sample(test)
    sample_ids = {(r["contract"], r["chunk_index"]) for r in sample}
    index = [i for i, r in enumerate(test) if (r["contract"], r["chunk_index"]) in sample_ids]
    sample_rows = [test[i] for i in index]

    full = {"keywords": score(test, keyword_pred), "embed (0.5 cutoff)": score(test, untuned_pred),
            "embed (tuned)": score(test, embedding_pred)}
    weights = sample_weights(sample_rows, test)
    on_sample = {"keywords": score(sample_rows, [keyword_pred[i] for i in index], weights=weights),
                 "embed (0.5 cutoff)": score(sample_rows, [untuned_pred[i] for i in index], weights=weights),
                 "embed (tuned)": score(sample_rows, [embedding_pred[i] for i in index], weights=weights)}
    report = "\n\n".join([
        format_table(f"FULL TEST SET ({len(test)} chunks from the held-out contracts)", full),
        format_table(f"EVAL SAMPLE ({len(sample_rows)} chunks, others reweighted to stand for the full test set; should land near the table above)", on_sample),
    ])
    print("\n" + report)

    with open(os.path.join(args.out_dir, "eval_sample.jsonl"), "w", encoding="utf-8") as f:
        for r in sample_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    predictions = {f"{r['contract']}::{r['chunk_index']}": sorted(embedding_pred[i])
                   for i, r in zip(index, sample_rows)}
    with open(os.path.join(args.out_dir, "embedding_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    with open(os.path.join(args.out_dir, "eval_report_embeddings.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
