"""Synthetic-only complete annotation and uncertainty-loss regression tests."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from gold_ner import SCHEMA, digest, load_bundle, prepare, validate_row, validate_partitions
from prepare_ner_gold import publish
from train_ner import align


def row(source, text, entities=None, masks=None, complete=True, unit="u"):
    return {"schema": SCHEMA, "source_id": source, "unit_id": unit, "text": text,
            "text_sha256": digest(text), "annotation_complete": complete,
            "entities": entities or [], "masked_spans": masks or []}


def corpus():
    return [row("a", "Jane", [{"start": 0, "end": 4, "label": "PERSON"}]),
            row("a", "Technical port 8126", unit="negative"),
            row("b", "Rina", [{"start": 0, "end": 4, "label": "PERSON"}]),
            row("c", "Lena", [{"start": 0, "end": 4, "label": "PERSON"}])]


class Tokenizer:
    def __init__(self, offsets): self.offsets = offsets
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(self.offsets))), "offset_mapping": self.offsets.copy()}


class GoldTests(unittest.TestCase):
    def test_complete_source_hash_and_codepoint_contract(self):
        value = row("a", "\U0001d11e Doë", [{"start": 2, "end": 5, "label": "PERSON"}])
        self.assertEqual(validate_row(value), value)
        for key, wrong in [("text_sha256", "0" * 64), ("annotation_complete", 1)]:
            changed = copy.deepcopy(value); changed[key] = wrong
            with self.assertRaises(ValueError): validate_row(changed)

    def test_bounds_types_unknown_and_overlap_rejected(self):
        for entity in [{"start": True, "end": 2, "label": "PERSON"},
                       {"start": 0, "end": 999, "label": "PERSON"},
                       {"start": 0, "end": 1, "label": "UNCERTAIN"}]:
            with self.assertRaises(ValueError): validate_row(row("a", "word", [entity]))
        value = row("a", "Jane", [{"start": 0, "end": 4, "label": "PERSON"}], [{"start": 3, "end": 4}])
        with self.assertRaises(ValueError): validate_row(value)

    def test_incomplete_rows_never_become_o(self):
        value = row("a", "Jane", complete=False)
        with self.assertRaises(ValueError): validate_row(value)
        with self.assertRaises(ValueError): align([value], Tokenizer([(0, 4)]), {"O": 0}, 8)
        bundle = prepare(corpus() + [row("a", "Unknown", complete=False, unit="pending")],
                         {"a": "train", "b": "selection", "c": "holdout"}, ["PERSON"])
        self.assertEqual(bundle["excluded_incomplete"], 1)
        self.assertEqual(len(bundle["train"]), 2)

    def test_unknown_token_overlap_not_just_first_character(self):
        value = row("a", "abc", masks=[{"start": 2, "end": 3}])
        result = align([value], Tokenizer([(0, 0), (0, 3), (3, 3)]), {"O": 0}, 8)
        self.assertEqual(result, [{"input_ids": [0, 1, 2], "labels": [-100, -100, -100]}])

    def test_known_negative_and_entity_labels(self):
        rows = [row("a", "Jane", [{"start": 0, "end": 4, "label": "PERSON"}]), row("b", "port")]
        result = align(rows, Tokenizer([(0, 0), (0, 2), (2, 4)]),
                       {"O": 0, "B-PERSON": 1, "I-PERSON": 2}, 8)
        self.assertEqual(result, [{"input_ids": [0, 1, 2], "labels": [-100, 1, 2]},
                                  {"input_ids": [0, 1, 2], "labels": [-100, 0, 0]}])

    def test_gold_crossing_boundary_token_is_not_false_o(self):
        value = row("a", "xJane", [{"start": 1, "end": 5, "label": "PERSON"}])
        result = align([value], Tokenizer([(0, 5)]), {"O": 0, "B-PERSON": 1, "I-PERSON": 2}, 8)
        self.assertEqual(result, [{"input_ids": [0], "labels": [-100]}])

    def test_gold_does_not_silently_truncate(self):
        with self.assertRaises(ValueError): align([row("a", "abcd")], Tokenizer([(0, 1)] * 5), {"O": 0}, 4)

    def test_unknown_types_never_fall_back_to_o(self):
        value = row("a", "Jane", [{"start": 0, "end": 4, "label": "PERSON"}])
        with self.assertRaises(ValueError): align([value], Tokenizer([(0, 4)]), {"O": 0}, 8)

    def test_source_and_exact_copy_leakage_rejected(self):
        rows = corpus(); parts = {"train": rows[:2], "selection": [rows[2]], "holdout": [rows[3]]}
        changed = copy.deepcopy(parts); changed["selection"][0]["source_id"] = "a"
        with self.assertRaises(ValueError): validate_partitions(changed)
        changed = copy.deepcopy(parts); changed["selection"][0]["text"] = "Jane";changed["selection"][0]["text_sha256"] = digest("Jane")
        with self.assertRaises(ValueError): validate_partitions(changed)

    def test_training_requires_positive_and_confirmed_negative(self):
        for values in [corpus()[1:], [corpus()[0], *corpus()[2:]]]:
            with self.assertRaises(ValueError): prepare(values, {"a": "train", "b": "selection", "c": "holdout"}, ["PERSON"])

    def test_taxonomy_and_frozen_source_map_required(self):
        for splits, labels in [({}, ["PERSON"]), ({"a": "train", "b": "selection", "c": "holdout"}, ["PIN"])]:
            with self.assertRaises(ValueError): prepare(corpus(), splits, labels)

    def test_bundle_roundtrip_checksum_private_no_clobber(self):
        bundle = prepare(corpus(), {"a": "train", "b": "selection", "c": "holdout"}, ["PERSON"])
        self.assertEqual(bundle, prepare(corpus(), {"a": "train", "b": "selection", "c": "holdout"}, ["PERSON"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.json";publish(path, bundle)
            self.assertEqual(load_bundle(path), bundle)
            with self.assertRaises(FileExistsError): publish(path, {"bad": True})
            self.assertEqual(load_bundle(path), bundle)
            if __import__('os').name != 'nt': self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            bad = copy.deepcopy(bundle);bad["train"][0]["text"] = "changed"
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError): load_bundle(path)

    def test_duplicate_json_fields_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.json";path.write_text('{"schema":"x","schema":"y"}')
            with self.assertRaises(ValueError): load_bundle(path)


if __name__ == "__main__": unittest.main()
