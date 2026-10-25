"""
Tests for the scoring code in evaluate_clause_types.py and the reply parsing in
evaluate_clause_types_llm.py (no model, no network). Run from the repo root:

    python3 -m unittest discover -s train_model -p "test_evaluate_clause_types.py"

Needs numpy and requests (both already used by the evaluation scripts).
"""
import importlib.util
import os
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ev = load("evaluate_clause_types", "evaluate_clause_types.py")
llm = load("evaluate_clause_types_llm", "evaluate_clause_types_llm.py")

LIABILITY = "Cap On Liability"
NONCOMPETE = "Non-Compete"


def row(*types, contract="C"):
    return {"contract": contract, "chunk_index": 0, "text": "x", "types": list(types)}


class ScoreTests(unittest.TestCase):
    def test_precision_recall_and_f1_for_one_type(self):
        rows = [row(LIABILITY), row(LIABILITY), row(), row()]
        predicted = [{LIABILITY}, set(), {LIABILITY}, set()]
        result = ev.score(rows, predicted)[LIABILITY]
        self.assertEqual((result["precision"], result["recall"]), (0.5, 0.5))
        self.assertEqual(result["f1"], 0.5)
        self.assertEqual(result["positives"], 2)

    def test_a_type_with_no_predictions_scores_zero_without_dividing_by_zero(self):
        result = ev.score([row(LIABILITY)], [set()])[LIABILITY]
        self.assertEqual((result["precision"], result["recall"], result["f1"]), (0.0, 0.0, 0.0))

    def test_macro_f1_counts_every_type_equally(self):
        rows = [row(LIABILITY), row(NONCOMPETE)]
        perfect_on_one = [{LIABILITY}, set()]
        macro = ev.score(rows, perfect_on_one)["macro_f1"]
        self.assertAlmostEqual(macro, 1 / len(ev.TYPES))

    def test_weights_make_a_false_alarm_on_a_sampled_chunk_count_for_many(self):
        rows = [row(LIABILITY), row()]            # one real positive, one sampled "other"
        predicted = [{LIABILITY}, {LIABILITY}]    # the "other" is a false alarm
        plain = ev.score(rows, predicted)[LIABILITY]
        weighted = ev.score(rows, predicted, weights=[1.0, 9.0])[LIABILITY]
        self.assertEqual(plain["precision"], 0.5)
        self.assertAlmostEqual(weighted["precision"], 0.1)  # 1 true hit against 9 false ones
        self.assertEqual(weighted["positives"], 1)          # the real count is not weighted


class SampleTests(unittest.TestCase):
    def setUp(self):
        self.test_rows = [row(LIABILITY) for _ in range(5)] + [row() for _ in range(95)]

    def test_sample_keeps_every_positive_and_some_others(self):
        sample = ev.make_sample(self.test_rows, negatives_per_positive=1.0)
        self.assertEqual(sum(1 for r in sample if r["types"]), 5)
        self.assertEqual(sum(1 for r in sample if not r["types"]), 5)

    def test_weights_scale_the_others_back_to_the_full_set(self):
        sample = ev.make_sample(self.test_rows, negatives_per_positive=1.0)
        weights = ev.sample_weights(sample, self.test_rows)
        others = sum(w for r, w in zip(sample, weights) if not r["types"])
        self.assertAlmostEqual(others, 95)
        self.assertTrue(all(w == 1.0 for r, w in zip(sample, weights) if r["types"]))


class ThresholdTests(unittest.TestCase):
    def test_best_threshold_separates_two_groups(self):
        y = np.array([True, True, False, False])
        probabilities = np.array([0.9, 0.8, 0.3, 0.2])
        cutoff = ev.best_threshold(y, probabilities)
        self.assertGreater(cutoff, 0.3)
        self.assertLessEqual(cutoff, 0.8)

    def test_keyword_baseline_finds_a_plain_non_compete(self):
        self.assertIn(NONCOMPETE, ev.keyword_predict("The Consultant shall not compete with the Client."))
        self.assertEqual(ev.keyword_predict("This Agreement is governed by the laws of Delaware."), set())


class LlmReplyTests(unittest.TestCase):
    def test_valid_types_are_kept_and_unknown_names_dropped(self):
        reply = '{"types": ["Non-Compete", "Governing Law", "Non-Compete"]}'
        self.assertEqual(llm.parse_types(reply), [NONCOMPETE])

    def test_an_empty_list_means_no_type(self):
        self.assertEqual(llm.parse_types('{"types": []}'), [])

    def test_a_reply_that_is_not_json_is_a_failure_not_an_empty_answer(self):
        self.assertIsNone(llm.parse_types("I think this is a non-compete."))
        self.assertIsNone(llm.parse_types('{"types": "Non-Compete"}'))


if __name__ == "__main__":
    unittest.main()
