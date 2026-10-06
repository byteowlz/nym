"""Offline contracts only; the test's 40-type map is NOT a measured checkpoint."""
import ast
from collections import defaultdict
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import agent_trace_corpus as corpus

# Explicit synthetic 81-label config to exercise preservation/validation. The
# production CLI REQUIRES the actual local checkpoint map and never assumes
# these particular two omissions match a deployed model.
TYPES = sorted(set(corpus.VALUES) - {"BUILDING_NUMBER", "SECONDARY_ADDRESS"})
LABELS = ["O"] + [f"{p}-{label}" for label in TYPES for p in ("B", "I")]
CONFIG = {"id2label": {str(i): label for i, label in enumerate(LABELS)},
          "label2id": {label: i for i, label in enumerate(LABELS)}}


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.rows = corpus.generate(LABELS, replicas=2)

    def test_vocabulary_matches_original_without_faker_dependency(self):
        tree = ast.parse(Path(corpus.__file__).with_name("labels.py").read_text())
        generators = next(node.value for node in tree.body if isinstance(node, ast.AnnAssign)
                          and getattr(node.target, "id", "") == "GENERATORS")
        self.assertEqual(set(corpus.VALUES), {ast.literal_eval(key) for key in generators.keys})

    def test_train_rows_cover_each_frozen_class(self):
        train = [row for row in self.rows if row["split"] == "train"]
        self.assertEqual({span["label"] for row in train for span in row["entities"]}, set(TYPES))
        self.assertEqual(corpus.load_label_map(CONFIG), LABELS)

    def test_non_alphabetical_frozen_ids_preserved_including_o(self):
        labels = LABELS[17:] + LABELS[:17]
        config = {"id2label": {str(i): label for i, label in enumerate(labels)},
                  "label2id": {label: i for i, label in enumerate(labels)}}
        self.assertEqual(corpus.load_label_map(config), labels)
        self.assertNotEqual(config["label2id"]["O"], 0)
        rows = corpus.generate(labels, replicas=1)
        self.assertEqual({span["label"] for row in rows for span in row["entities"]}, set(TYPES))

    def test_config_selects_actual_types_not_assumed_omissions(self):
        types = sorted(set(corpus.VALUES) - {"GENDER", "FAX_NUMBER"})
        labels = ["O"] + [f"{prefix}-{label}" for label in types for prefix in ("B", "I")]
        config = {"id2label": {str(i): label for i, label in enumerate(labels)}}
        self.assertEqual(corpus.load_label_map(config), labels)
        rows = corpus.generate(labels, replicas=1)
        self.assertEqual({span["label"] for row in rows for span in row["entities"]}, set(types))

    def test_generation_does_not_read_old_benchmarks_or_other_files(self):
        from unittest.mock import patch

        with patch("builtins.open", side_effect=AssertionError("generation must be self-contained")), patch.object(Path, "read_text", side_effect=AssertionError("no fixture reads")):
            self.assertEqual(corpus.generate(LABELS, replicas=2), self.rows)

    def test_rejects_new_head_duplicates_and_inconsistent_map(self):
        for mutation in ("length", "duplicate", "reverse"):
            config = copy.deepcopy(CONFIG)
            if mutation == "length":
                config["id2label"]["81"] = "B-NEW"
            elif mutation == "duplicate":
                config["id2label"]["1"] = config["id2label"]["2"]
            else:
                config["label2id"]["O"] = 10
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                corpus.load_label_map(config)

    def test_deterministic_family_disjointness_and_transports(self):
        self.assertEqual(self.rows, corpus.generate(LABELS, replicas=2))
        families, transports = defaultdict(set), defaultdict(set)
        for row in self.rows:
            families[row["scenario_family"]].add(row["split"])
            transports[row["split"]].add(row["transport"])
        self.assertTrue(all(len(splits) == 1 for splits in families.values()))
        self.assertEqual(dict(transports), {split: set(corpus.TRANSPORTS) for split in ("train", "selection", "holdout")})
        self.assertNotEqual(corpus.family_splits(47), corpus.family_splits(48))
        self.assertEqual(len({row["text"] for row in self.rows}), len(self.rows))

    def test_annotation_integrity_and_full_o(self):
        for row in self.rows:
            corpus.validate_row(row, LABELS)
            self.assertEqual(row["supervision"], "full")
            self.assertEqual(row["source"], corpus.SOURCE)
            if row["variant"] == "negative":
                self.assertEqual(row["entities"], [])
            for entity in row["entities"] + row["benign"]:
                self.assertEqual(row["text"].encode()[entity["byte_start"]:entity["byte_end"]].decode(), entity["value"])

    def test_same_literal_benign_and_private_occurrences(self):
        rows = [row for row in self.rows if row["variant"] == "mixed" and row["benign"]]
        for row in rows:
            for negative in row["benign"]:
                positives = [span for span in row["entities"] if span["value"] == negative["value"]]
                self.assertTrue(positives, row["scenario_family"])
                self.assertTrue(all(span["start"] != negative["start"] for span in positives))

    def test_negation_does_not_hide_actual_credentials(self):
        for family in ("allocator", "socket-port", "retry-budget", "curl-auth", "dotenv", "secret-rotation"):
            rows = [row for row in self.rows if row["scenario_family"] == family and row["variant"] == "negated-with-secret"]
            self.assertTrue(rows)
            self.assertTrue(all(row["entities"] for row in rows))

    def test_unicode_repeated_slots_have_independent_char_and_byte_offsets(self):
        text, entities, benign = corpus.render("λ [[GIVEN_NAME]] / [[O:GIVEN_NAME]] / [[GIVEN_NAME]]", corpus.VALUES)
        self.assertEqual([text[span["start"]:span["end"]] for span in entities + benign], ["Zoë"] * 3)
        self.assertEqual(len({span["byte_start"] for span in entities + benign}), 3)
        self.assertTrue(all(span["byte_start"] > span["start"] for span in entities + benign))

    def test_detects_tampered_offsets_and_overlap(self):
        row = copy.deepcopy(next(row for row in self.rows if row["entities"]))
        row["entities"][0]["byte_start"] += 1
        with self.assertRaises(ValueError):
            corpus.validate_row(row, LABELS)
        row = copy.deepcopy(next(row for row in self.rows if row["entities"]))
        row["entities"].append(dict(row["entities"][0]))
        with self.assertRaises(ValueError):
            corpus.validate_row(row, LABELS)

    def test_training_alignment_supervises_negatives_as_o(self):
        path = Path(corpus.__file__).parents[1] / "train_ner.py"
        spec = importlib.util.spec_from_file_location("offline_align", path)
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)

        def tokenizer(text, **kwargs):
            return {"offset_mapping": [(0, 0)] + [(i, i + 1) for i in range(len(text))],
                    "input_ids": list(range(len(text) + 1))}

        negative = next(row for row in self.rows if row["variant"] == "negative")
        aligned = trainer.align([negative], tokenizer, CONFIG["label2id"], 512)
        self.assertEqual(aligned[0]["labels"], [-100] + [0] * len(negative["text"]))

    def test_cli_emits_three_splits_and_preserved_map(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "config.json"
            labels = LABELS[17:] + LABELS[:17]
            supplied = {"id2label": {str(i): label for i, label in enumerate(labels)},
                        "label2id": {label: i for i, label in enumerate(labels)}}
            config.write_text(json.dumps(supplied))
            out = Path(temp) / "corpus"
            proc = subprocess.run([sys.executable, corpus.__file__, "--label-config", str(config), "--output-dir", str(out), "--replicas", "1"], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["id2label"], supplied["id2label"])
            self.assertEqual(manifest["label2id"], supplied["label2id"])
            for split in ("train", "selection", "holdout"):
                rows = [json.loads(line) for line in (out / f"agent-trace.{split}.jsonl").read_text().splitlines()]
                self.assertTrue(rows)
                self.assertEqual({row["split"] for row in rows}, {split})


if __name__ == "__main__":
    unittest.main()
