"""
Tests for the pure parts of evaluate_clause_types_filter.py (no model, no
network). Run from the repo root:

    python3 -m unittest discover -s train_model -p "test_evaluate_clause_types_filter.py"
"""
import importlib.util
import os
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("flt", os.path.join(HERE, "evaluate_clause_types_filter.py"))
flt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flt)

TYPES = flt.base.TYPES


def train_row(text, *types, contract="TRAIN"):
    return {"contract": contract, "chunk_index": 0, "text": text, "types": list(types), "split": "train"}


class RecallCutoffTests(unittest.TestCase):
    def test_returns_the_highest_cutoff_that_still_catches_enough_positives(self):
        y = np.array([True] * 10 + [False] * 10)
        probabilities = np.array([0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5] + [0.4] * 10)
        cutoff = flt.recall_cutoff(y, probabilities, target=0.9)
        caught = np.sum((probabilities >= cutoff) & y)
        self.assertGreaterEqual(caught, 9)                       # at least 90% of the 10 real ones
        self.assertEqual(np.sum((probabilities >= cutoff + 0.011) & y) < 9, True)  # and no higher cutoff would

    def test_when_nothing_scores_high_it_falls_back_to_the_lowest_cutoff(self):
        y = np.array([True, False])
        self.assertEqual(flt.recall_cutoff(y, np.array([0.0, 0.0])), 0.01)


class ExampleTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        for i in range(6):
            for t in TYPES:
                self.rows.append(train_row(f"positive {t} {i}", t))
        self.rows += [train_row(f"plain {i}") for i in range(20)]
        self.rows.append(train_row("x" * 900, TYPES[0]))     # too long to use as an example
        self.oof = {t: np.zeros(len(self.rows)) for t in TYPES}
        self.lookalike = len(self.rows) - 5                   # a chunk with no type that scored highest
        for t in TYPES:
            self.oof[t][self.lookalike] = 0.9

    def test_each_type_gets_two_positives_and_one_near_miss(self):
        examples = flt.pick_examples(self.rows, self.oof)
        for t in TYPES:
            self.assertEqual(len(examples[t]["positives"]), 2)
            self.assertTrue(all(f"positive {t}" in p for p in examples[t]["positives"]))
            self.assertEqual(examples[t]["near_miss"], self.rows[self.lookalike]["text"])

    def test_picks_are_repeatable_and_skip_long_chunks(self):
        first = flt.pick_examples(self.rows, self.oof)
        second = flt.pick_examples(self.rows, self.oof)
        self.assertEqual(first, second)
        for t in TYPES:
            self.assertFalse(any(len(p) > flt.EXAMPLE_MAX_CHARS for p in first[t]["positives"]))


class PromptTests(unittest.TestCase):
    EXAMPLES = {t: {"positives": [f"yes-{t}-1", f"yes-{t}-2"], "near_miss": f"no-{t}"} for t in TYPES}

    def test_only_the_candidate_types_appear(self):
        prompt = flt.build_system_prompt(["Non-Compete"])
        self.assertIn("Non-Compete", prompt)
        self.assertNotIn("Exclusivity", prompt)

    def test_examples_appear_only_for_the_few_shot_variant_and_only_for_candidates(self):
        zero = flt.build_system_prompt(["Non-Compete"])
        few = flt.build_system_prompt(["Non-Compete"], self.EXAMPLES)
        self.assertNotIn("yes-Non-Compete-1", zero)
        self.assertIn("yes-Non-Compete-1", few)
        self.assertIn("no-Non-Compete", few)
        self.assertNotIn("yes-Exclusivity-1", few)


class WinRuleTests(unittest.TestCase):
    def scores(self, recall, macro_f1):
        result = {t: {"recall": recall} for t in TYPES}
        result["macro_f1"] = macro_f1
        return result

    def test_passes_only_when_all_three_conditions_hold(self):
        checks = flt.check_win_rule(self.scores(0.9, 0.55), share=0.10)
        self.assertTrue(all(ok for ok, _ in checks.values()))

    def test_each_condition_can_fail_on_its_own(self):
        low_recall = flt.check_win_rule(self.scores(0.5, 0.55), share=0.10)
        low_f1 = flt.check_win_rule(self.scores(0.9, 0.45), share=0.10)
        too_much = flt.check_win_rule(self.scores(0.9, 0.55), share=0.30)
        for checks in (low_recall, low_f1, too_much):
            self.assertEqual(sum(1 for ok, _ in checks.values() if not ok), 1)

    def test_share_sent_counts_flagged_sampled_chunks_by_their_weight(self):
        sample = [{"contract": "C", "chunk_index": 0}, {"contract": "C", "chunk_index": 1}]
        candidates = {"C::0": ["Non-Compete"], "C::1": []}
        self.assertAlmostEqual(flt.share_sent(sample, [1.0, 50.0], candidates, test_size=100), 0.01)
        candidates["C::1"] = ["Exclusivity"]
        self.assertAlmostEqual(flt.share_sent(sample, [1.0, 50.0], candidates, test_size=100), 0.51)


if __name__ == "__main__":
    unittest.main()
