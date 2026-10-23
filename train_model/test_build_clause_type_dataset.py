"""
Tests for build_clause_type_dataset.py, on an invented mini-contract (no CUAD
file needed). Run locally from the repo root:

    python3 -m unittest discover -s train_model -p "test_build_clause_type_dataset.py"
"""
import importlib.util
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("builder", os.path.join(HERE, "build_clause_type_dataset.py"))
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

CONTRACT = (
    "SERVICES AGREEMENT\n \n"
    "1. SERVICES. The Consultant shall provide the services described in Exhibit A to the Client\n"
    "and shall deliver them on the dates listed in the schedule.\n \n"
    "2. OWNERSHIP. All work product created under this Agreement shall be owned by the Client\n"
    "upon creation, and the Consultant hereby assigns all rights in it to the\n"
    "Page 2 \n"
    "Client without further payment.\n \n"
    "3. LIABILITY. The Consultant's liability under this Agreement is unlimited for any\n"
    "breach of its confidentiality obligations.\n"
)
OWNERSHIP_PASSAGE = (
    "All work product created under this Agreement shall be owned by the Client\n"
    "upon creation, and the Consultant hereby assigns all rights in it to the\n"
    "Page 2 \n"
    "Client without further payment."
)
LIABILITY_PASSAGE = "is unlimited for any\nbreach of its confidentiality obligations."


def question(category):
    return (f'Highlight the parts (if any) of this contract related to "{category}" '
            f'that should be reviewed by a lawyer.')


def item(title="MINI", qas=None):
    return {"title": title, "paragraphs": [{"context": CONTRACT, "qas": qas or []}]}


class NormalizeTests(unittest.TestCase):
    def test_collapses_whitespace_and_lowercases(self):
        norm, _ = builder.normalize_with_map("  Hello \n\n  WORLD  ")
        self.assertEqual(norm, "hello world")

    def test_every_character_maps_back_to_the_same_letter(self):
        text = "Ab  c\nD"
        norm, index_map = builder.normalize_with_map(text)
        for i, ch in enumerate(norm):
            if ch != " ":
                self.assertEqual(text[index_map[i]].lower(), ch)


class LocateTests(unittest.TestCase):
    def setUp(self):
        self.segment = builder.load_segmenter()
        self.chunks = self.segment(CONTRACT)
        self.ranges = builder.locate_chunks(self.chunks, CONTRACT)

    def test_all_chunks_are_found_in_order_without_overlap(self):
        self.assertEqual(len(self.chunks), 3)
        self.assertTrue(all(r is not None for r in self.ranges))
        for (a_start, a_end), (b_start, b_end) in zip(self.ranges, self.ranges[1:]):
            self.assertLess(a_start, a_end)
            self.assertLessEqual(a_end, b_start)

    def test_a_chunk_after_a_removed_page_header_still_covers_the_original_text(self):
        start, end = self.ranges[1]
        original = CONTRACT[start:end]
        self.assertTrue(original.startswith("2. OWNERSHIP"))
        self.assertTrue(original.rstrip().endswith("without further payment."))
        self.assertIn("Page 2", original)  # removed from the chunk, present in the original

    def test_a_chunk_that_is_not_in_the_text_is_reported_as_unlocated(self):
        ranges = builder.locate_chunks(["this sentence does not appear anywhere in it"], CONTRACT)
        self.assertEqual(ranges, [None])


class LabelTests(unittest.TestCase):
    RANGES = [(0, 100), (100, 200), (200, 300)]

    def test_a_passage_inside_a_chunk_labels_only_that_chunk(self):
        labels, used = builder.label_chunks(self.RANGES, [("Cap On Liability", 110, 190)])
        self.assertEqual(labels, [[], ["Cap On Liability"], []])
        self.assertEqual(used, {0})

    def test_a_passage_spanning_two_chunks_labels_both(self):
        # 70 of its 140 characters fall in each chunk: half of the passage, so both qualify
        labels, _ = builder.label_chunks(self.RANGES, [("Non-Compete", 30, 170)])
        self.assertEqual(labels, [["Non-Compete"], ["Non-Compete"], []])

    def test_a_passage_straddling_a_boundary_unevenly_labels_only_the_larger_side(self):
        # 40 of 100 characters in chunk 0, 60 in chunk 1: only chunk 1 reaches half
        labels, _ = builder.label_chunks(self.RANGES, [("Non-Compete", 60, 160)])
        self.assertEqual(labels, [[], ["Non-Compete"], []])

    def test_a_tiny_overlap_does_not_label(self):
        labels, used = builder.label_chunks(self.RANGES, [("Exclusivity", 95, 130)])
        # overlap with chunk 0 is 5 characters: under half of the chunk and of the passage
        self.assertEqual(labels[0], [])
        self.assertEqual(labels[1], ["Exclusivity"])

    def test_a_chunk_can_carry_two_types(self):
        labels, _ = builder.label_chunks(self.RANGES, [("Cap On Liability", 110, 190), ("Uncapped Liability", 120, 180)])
        self.assertEqual(labels[1], ["Cap On Liability", "Uncapped Liability"])

    def test_an_unlocated_chunk_gets_no_labels(self):
        labels, _ = builder.label_chunks([None], [("Exclusivity", 0, 50)])
        self.assertEqual(labels, [[]])


class CategoryNameTests(unittest.TestCase):
    def test_cuads_odd_spelling_of_a_focus_type_is_normalised(self):
        # The raw CUAD file says "Ip Ownership Assignment"; without this the type had zero rows.
        self.assertEqual(builder.canonical_category("Ip Ownership Assignment"), "IP Ownership Assignment")
        self.assertEqual(builder.canonical_category("cap on liability"), "Cap On Liability")

    def test_other_names_are_left_alone(self):
        self.assertEqual(builder.canonical_category("Governing Law"), "Governing Law")

    def test_every_focus_type_is_found_when_cuad_spells_it_in_any_case(self):
        for focus in builder.FOCUS_TYPES:
            self.assertEqual(builder.canonical_category(focus.title()), focus)
            self.assertEqual(builder.canonical_category(focus.upper()), focus)

    def test_a_span_for_the_oddly_spelled_type_is_extracted_under_the_normal_name(self):
        qas = [{"question": question("Ip Ownership Assignment"),
                "answers": [{"text": OWNERSHIP_PASSAGE, "answer_start": CONTRACT.index("All work product")}]}]
        spans, _ = builder.extract_spans(item(qas=qas))
        self.assertEqual(spans[0][0], "IP Ownership Assignment")


class ProcessContractTests(unittest.TestCase):
    def setUp(self):
        self.segment = builder.load_segmenter()
        qas = [
            {"question": question("IP Ownership Assignment"),
             "answers": [{"text": OWNERSHIP_PASSAGE, "answer_start": CONTRACT.index("All work product")}]},
            {"question": question("Uncapped Liability"),
             "answers": [{"text": LIABILITY_PASSAGE, "answer_start": CONTRACT.index("is unlimited")}]},
            {"question": question("Governing Law"), "answers": []},
        ]
        self.item = item(qas=qas)

    def test_each_chunk_gets_the_right_types(self):
        rows, stats = builder.process_contract(self.item, set(), self.segment)
        by_index = {r["chunk_index"]: r["types"] for r in rows}
        self.assertEqual(by_index, {0: [], 1: ["IP Ownership Assignment"], 2: ["Uncapped Liability"]})
        self.assertEqual(stats["passages"], 2)
        self.assertEqual(stats["mismatches"], 0)
        self.assertEqual(stats["covered"], 2)

    def test_an_offset_that_does_not_point_at_the_text_is_counted(self):
        self.item["paragraphs"][0]["qas"][0]["answers"][0]["answer_start"] += 3
        _, stats = builder.process_contract(self.item, set(), self.segment)
        self.assertEqual(stats["mismatches"], 1)

    def test_split_follows_the_official_test_titles(self):
        self.assertEqual(builder.assign_split("MINI", {"MINI"}), "test")
        self.assertEqual(builder.assign_split("MINI", {"OTHER"}), "train")
        rows, _ = builder.process_contract(self.item, {"MINI"}, self.segment)
        self.assertTrue(all(r["split"] == "test" for r in rows))

    def test_rows_carry_the_original_offsets(self):
        rows, _ = builder.process_contract(self.item, set(), self.segment)
        row = next(r for r in rows if r["chunk_index"] == 1)
        self.assertTrue(CONTRACT[row["start"]:row["end"]].startswith("2. OWNERSHIP"))


if __name__ == "__main__":
    unittest.main()
