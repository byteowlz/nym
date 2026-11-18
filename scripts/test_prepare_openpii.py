#!/usr/bin/env python3
"""Offline corpus-boundary tests: uv run scripts/test_prepare_openpii.py."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from prepare_openpii import (ALIASES, INPUT_HASHES, annotations, convert, digest,
                             normalized, prepare, sha256, signatures)


def row(uid, prefix, value="Alex", split="train"):
    return {"source_text": prefix + value, "privacy_mask": [
        {"label": "GIVENNAME", "start": len(prefix), "end": len(prefix + value), "value": value}],
        "source_dataset": "synthetic-fixture", "region": "ZZ", "uid": uid,
        "language": "en", "split": split}


class OpenPIITests(unittest.TestCase):
    def setUp(self):
        self.mapping = {"O": 0}
        for label in sorted(set(ALIASES.values())):
            for prefix in ("B", "I"):
                self.mapping[f"{prefix}-{label}"] = len(self.mapping)

    def test_unicode_offsets_and_known_alias(self):
        source = row(1, "名前: ", "田中")
        self.assertEqual(convert(source, annotations(source), self.mapping)["entities"],
                         [{"start": 4, "end": 6, "label": "GIVEN_NAME"}])

    def test_rejects_unknown_instead_of_promoting_it_to_o(self):
        source = row(1, "Title: ", "Dr")
        source["privacy_mask"][0]["label"] = "TITLE"
        with self.assertRaisesRegex(ValueError, "unsupported"):
            convert(source, annotations(source), self.mapping)

    def test_rejects_missing_taxonomy_and_empty_annotations(self):
        source = row(1, "Name: ")
        with self.assertRaisesRegex(ValueError, "frozen taxonomy"):
            convert(source, annotations(source), {"O": 0})
        with self.assertRaisesRegex(ValueError, "unreviewed empty"):
            convert(source, [], self.mapping)

    def test_wrong_value_overlap_and_noninteger_offsets_fail(self):
        for kind in ("value", "overlap", "offset"):
            source = row(1, "Name: ")
            if kind == "value":
                source["privacy_mask"][0]["value"] = "different"
            elif kind == "overlap":
                source["privacy_mask"] *= 2
            else:
                source["privacy_mask"][0]["start"] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                annotations(source)

    def test_delexicalized_context_and_normalized_text_signatures(self):
        a, b = row(1, "Name: ", "Alex"), row(2, "Name: ", "Erika")
        shared = signatures(a, annotations(a)) & signatures(b, annotations(b))
        self.assertEqual(len(shared), 1)
        self.assertTrue(next(iter(shared)).startswith(b"c"))
        self.assertEqual(normalized("  ＡLEX\nExample "), "alex example")
        self.assertEqual(digest(normalized("Ａlex")), digest(normalized("Alex")))

    def test_whole_validation_blocks_training_and_qa_is_reproducible(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            validation = [row(i, f"Validation document {i}: ", split="validation") for i in range(30)]
            training = [row(100 + i, f"Train document {i}: ") for i in range(10)]
            training.append(row(500, "Validation document 3: ", "Erika"))
            training.append(row(4, "Changed text but same source identity: "))
            for split, rows in (("train", training), ("validation", validation)):
                (root / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            hashes = {name: sha256(root / name) for name in INPUT_HASHES}
            with patch.dict(INPUT_HASHES, hashes), redirect_stdout(io.StringIO()):
                first = prepare(root, root / "first", self.mapping, 10, 5)
                second = prepare(root, root / "second", self.mapping, 10, 5)
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    prepare(root, root / "first", self.mapping, 10, 5)
            self.assertEqual(first, second)
            self.assertEqual(first["counts"]["train_validation_overlap_excluded"], 2)
            self.assertEqual(first["cross_partition_signature_overlap"], 0)
            self.assertEqual({k: v["rows"] for k, v in first["partitions"].items()},
                             {"train": 10, "selection": 5, "holdout": 5})
            for split in ("train", "selection", "holdout"):
                self.assertEqual((root / "first" / f"{split}.jsonl").read_bytes(),
                                 (root / "second" / f"{split}.jsonl").read_bytes())

    def test_wrong_source_checksum_fails_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "train.jsonl").write_text("not the pinned source")
            with self.assertRaisesRegex(ValueError, "pinned upstream"):
                prepare(root, root / "out", self.mapping, 1, 1)
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
