#!/usr/bin/env python3
"""
Builds the clause-type training data from CUAD.

Why this exists: the old dataset (legal_clauses_labeled.csv) labelled a clause
"risky" if it contained one of seven keywords, and threw away which of CUAD's 41
expert-annotated clause types each passage was, and which contract it came from.
This script keeps what CUAD actually annotated.

What one row is: one CHUNK, made by LexiLite's own splitter
(api/analyzer/segmentation.py, imported unchanged) from a CUAD contract, so the
model trains on exactly what it will see at run time. Each row lists every
clause type whose highlighted passage overlaps the chunk (a chunk can carry
several types, so this is a multi-label task). A chunk with an empty list is a
negative for every type.

How passages are matched to chunks: by character position, using each passage's
`answer_start` offset in the contract text, not by matching text (a short passage
such as a state name matches unrelated chunks). Chunks are located in the
original text by their first and last few words, in order, because the splitter
joins lines and drops page headers, so a chunk is not always an exact copy.
Chunks that cannot be located are dropped and counted in the report.

The split is CUAD's own contract-level split (train_separate_questions.json and
test.json): no contract appears on both sides.

Usage (from the repo root):
    python3 train_model/build_clause_type_dataset.py
    python3 train_model/build_clause_type_dataset.py --limit 20     # quick look
Output (git-ignored): train_model/data/generated/clause_types.jsonl and report.txt
"""
import argparse
import collections
import importlib.util
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DEFAULT_CUAD_DIR = os.path.join(REPO_ROOT, "cuad", "data")
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "data", "generated")
SEGMENTATION_PATH = os.path.join(REPO_ROOT, "api", "analyzer", "segmentation.py")

CATEGORY_RE = re.compile(r'related to "([^"]+)"')

# The eight types of the first playbook, spelled as CUAD's questions spell them.
FOCUS_TYPES = [
    "IP Ownership Assignment",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Termination For Convenience",
    "Non-Compete",
    "Exclusivity",
    "Irrevocable Or Perpetual License",
]

ANCHOR_WORDS = 6        # words of a chunk's start and end used to find it in the original
LOCATE_SLACK = 200      # characters of tolerance for removed page headers and joins
OVERLAP_THRESHOLD = 0.5  # a chunk gets a type if the passage covers half the chunk, or vice versa


def canonical_category(name):
    """CUAD's raw file spells some types oddly ("Ip Ownership Assignment").
    Map any case variant of a focus type to the spelling used in FOCUS_TYPES;
    other names are returned unchanged."""
    for focus in FOCUS_TYPES:
        if name.lower() == focus.lower():
            return focus
    return name


def load_segmenter():
    """Import segment_clauses straight from its file, so no Django is needed."""
    spec = importlib.util.spec_from_file_location("lexilite_segmentation", SEGMENTATION_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.segment_clauses


def normalize_with_map(text):
    """Lower-case `text` and collapse whitespace runs to single spaces.
    Returns (normalized, index_map) where index_map[i] is the index in the
    original text of normalized character i."""
    chars, index_map = [], []
    pending_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            pending_space = bool(chars)
            continue
        if pending_space:
            chars.append(" ")
            index_map.append(i - 1)
            pending_space = False
        for lowered in ch.lower():
            chars.append(lowered)
            index_map.append(i)
    return "".join(chars), index_map


def _find_anchor(words, from_end, haystack, low, high, min_words):
    """Find the start (or end) of a chunk in `haystack` by its first (or last)
    few words. If the full anchor is not found, try shorter ones: the splitter
    drops page headers, so an anchor that straddles a removed header never
    appears contiguously in the original. Returns (position, phrase) or (-1, "")."""
    for k in range(min(ANCHOR_WORDS, len(words)), min_words - 1, -1):
        phrase = " ".join(words[-k:] if from_end else words[:k])
        at = haystack.find(phrase, low) if high is None else haystack.find(phrase, low, high)
        if at != -1:
            return at, phrase
    return -1, ""


def locate_chunks(chunks, context):
    """Find each chunk's (start, end) character range in `context`, or None if
    it cannot be found. Chunks are searched in order, so repeated phrases are
    resolved by position."""
    normalized, index_map = normalize_with_map(context)
    cursor = 0
    ranges = []
    for chunk in chunks:
        words = chunk.lower().split()
        if not words:
            ranges.append(None)
            continue
        expected = len(" ".join(words))
        start, _ = _find_anchor(words, False, normalized, cursor, None, min(3, len(words)))
        if start == -1:
            ranges.append(None)
            continue
        window_start = max(start, start + expected - LOCATE_SLACK - 80)
        window_end = start + expected + LOCATE_SLACK + 80
        tail_at, tail = _find_anchor(words, True, normalized, window_start, window_end, min(2, len(words)))
        end = tail_at + len(tail) if tail_at != -1 else start + expected
        end = min(end, len(normalized))
        if end <= start:
            ranges.append(None)
            continue
        ranges.append((index_map[start], index_map[end - 1] + 1))
        cursor = end
    return ranges


def extract_spans(item):
    """Return (spans, mismatches). spans are (category, start, end) in the
    contract text; mismatches counts passages whose offset does not point at
    the passage's own text (a data-integrity check)."""
    context = item["paragraphs"][0]["context"]
    spans, mismatches = [], 0
    for qa in item["paragraphs"][0]["qas"]:
        match = CATEGORY_RE.search(qa["question"])
        category = canonical_category(match.group(1)) if match else None
        if category is None:
            continue
        for answer in qa["answers"]:
            text, start = answer["text"], answer["answer_start"]
            if not text.strip():
                continue
            end = start + len(text)
            if context[start:end] != text:
                mismatches += 1
            spans.append((category, start, end))
    return spans, mismatches


def label_chunks(ranges, spans):
    """For each chunk range, the sorted list of types whose passages overlap it.
    Also returns the indexes of the passages that labelled at least one chunk."""
    labels, used = [], set()
    for rng in ranges:
        found = set()
        if rng is not None:
            chunk_start, chunk_end = rng
            for idx, (category, span_start, span_end) in enumerate(spans):
                overlap = min(chunk_end, span_end) - max(chunk_start, span_start)
                if overlap <= 0:
                    continue
                if (overlap / (chunk_end - chunk_start) >= OVERLAP_THRESHOLD
                        or overlap / (span_end - span_start) >= OVERLAP_THRESHOLD):
                    found.add(category)
                    used.add(idx)
        labels.append(sorted(found))
    return labels, used


def assign_split(title, test_titles):
    return "test" if title in test_titles else "train"


def process_contract(item, test_titles, segment_clauses):
    """Rows for one contract plus counts for the report."""
    title = item["title"]
    context = item["paragraphs"][0]["context"]
    chunks = segment_clauses(context)
    ranges = locate_chunks(chunks, context)
    spans, mismatches = extract_spans(item)
    labels, used = label_chunks(ranges, spans)
    split = assign_split(title, test_titles)

    rows = []
    for index, (chunk, rng, types) in enumerate(zip(chunks, ranges, labels)):
        if rng is None:
            continue
        rows.append({
            "contract": title, "split": split, "chunk_index": index,
            "start": rng[0], "end": rng[1], "text": chunk, "types": types,
        })
    stats = {
        "chunks": len(chunks), "kept": len(rows),
        "passages": len(spans), "mismatches": mismatches, "covered": len(used),
    }
    return rows, stats


def load_titles(path):
    with open(path, encoding="utf-8") as f:
        return {item["title"] for item in json.load(f)["data"]}


def build_report(rows, totals, contracts_by_split, seed):
    rng = random.Random(seed)
    lines = []
    add = lines.append
    add("CLAUSE-TYPE DATASET REPORT")
    add(f"contracts: train {contracts_by_split['train']}, test {contracts_by_split['test']}")
    dropped = totals["chunks"] - totals["kept"]
    add(f"chunks made: {totals['chunks']} | kept: {totals['kept']} | dropped (could not be located): "
        f"{dropped} ({dropped / max(totals['chunks'], 1):.1%})")
    add(f"annotated passages: {totals['passages']} | offset does not match its own text: "
        f"{totals['mismatches']} | passages that labelled at least one kept chunk: {totals['covered']} "
        f"({totals['covered'] / max(totals['passages'], 1):.1%})")
    no_type = sum(1 for r in rows if not r["types"])
    multi = sum(1 for r in rows if len(r["types"]) >= 2)
    labelled = sum(1 for r in rows if r["types"])
    add(f"chunks with no type at all: {no_type} ({no_type / max(len(rows), 1):.1%}) | "
        f"labelled chunks with two or more types: {multi} ({multi / max(labelled, 1):.1%})")
    add("")
    add(f"{'type':36} {'train chunks':>13} {'test chunks':>12} {'train contracts':>16} {'test contracts':>15}")
    positives = collections.defaultdict(list)
    for row in rows:
        for t in row["types"]:
            positives[t].append(row)
    for t in FOCUS_TYPES:
        pos = positives.get(t, [])
        tr = [r for r in pos if r["split"] == "train"]
        te = [r for r in pos if r["split"] == "test"]
        add(f"{t:36} {len(tr):>13} {len(te):>12} {len({r['contract'] for r in tr}):>16} "
            f"{len({r['contract'] for r in te}):>15}")
    add("")
    add("Spot check: three random chunks per focus type (read them; do they look like that type?)")
    for t in FOCUS_TYPES:
        pos = positives.get(t, [])
        add(f"\n[{t}]")
        for row in rng.sample(pos, min(3, len(pos))):
            add(f"  - ({row['contract'][:40]}) {row['text'][:200]!r}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cuad-dir", default=DEFAULT_CUAD_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="only the first N contracts (quick look)")
    parser.add_argument("--seed", type=int, default=13, help="seed for the spot-check sample")
    args = parser.parse_args(argv)

    with open(os.path.join(args.cuad_dir, "CUADv1.json"), encoding="utf-8") as f:
        contracts = json.load(f)["data"]
    if args.limit:
        contracts = contracts[:args.limit]
    test_titles = load_titles(os.path.join(args.cuad_dir, "test.json"))
    segment_clauses = load_segmenter()

    rows = []
    totals = collections.Counter()
    contracts_by_split = collections.Counter()
    for item in contracts:
        item_rows, stats = process_contract(item, test_titles, segment_clauses)
        rows.extend(item_rows)
        totals.update(stats)
        contracts_by_split[assign_split(item["title"], test_titles)] += 1

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "clause_types.jsonl"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = build_report(rows, totals, contracts_by_split, args.seed)
    with open(os.path.join(args.out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nwrote {len(rows)} rows to {os.path.join(args.out_dir, 'clause_types.jsonl')}")


if __name__ == "__main__":
    main()
