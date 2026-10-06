#!/usr/bin/env python3
"""Offline frozen-taxonomy safety tests: uv run --no-project scripts/test_train_ner.py.

No model download or optimizer step. Real checkpoint continuation is additionally
proved by the private bounded training study, not by these isolated mocks.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from train_ner import (build_labels, load_training_model, validate_checkpoint_labels,
                       validate_labels)


class FrozenLabelsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # Deliberately nonalphabetic ordering detects silent ID reordering.
        self.labels = ["O"] + [f"{p}-TYPE_{i:02d}" for i in reversed(range(40)) for p in ("B", "I")]
        self.mapping = {label: i for i, label in enumerate(self.labels)}
        self.config = {"id2label": {str(i): label for i, label in enumerate(self.labels)},
                       "label2id": self.mapping}
        self.path = Path(self.temp.name) / "config.json"
        self.path.write_text(json.dumps(self.config))
        self.rows = [{"text": "synthetic", "entities": [{"start": 0, "end": 9, "label": "TYPE_03"}]}]

    def test_subset_preserves_full_81_ids_and_config_bytes(self):
        before = self.path.read_bytes()
        self.assertEqual(build_labels(self.rows, self.path), (self.labels, self.mapping))
        self.assertEqual(build_labels([], self.path), (self.labels, self.mapping))
        self.assertEqual(self.path.read_bytes(), before)

    def test_default_still_derives_sorted_subset_without_config(self):
        self.assertEqual(build_labels(self.rows),
                         (["O", "B-TYPE_03", "I-TYPE_03"], {"O": 0, "B-TYPE_03": 1, "I-TYPE_03": 2}))

    def test_unmapped_train_and_evaluation_labels_fail(self):
        bad = [{"entities": [{"label": "UNMAPPED"}]}]
        with self.assertRaisesRegex(ValueError, "unmapped"):
            build_labels(bad, self.path)
        with self.assertRaisesRegex(ValueError, "unmapped"):
            validate_labels(bad, self.mapping)

    def test_malformed_maps_fail(self):
        for config in ({}, {"id2label": {"1": "O"}},
                       {"id2label": {"0": "O", "1": "B-X"}, "label2id": {"O": 0, "B-X": 1}},
                       {"id2label": {"0": "O", "1": "O"}, "label2id": {"O": 1}},
                       {"id2label": {"0": 0}, "label2id": {}},
                       {**self.config, "label2id": {"O": 0}}):
            with self.subTest(config=config):
                self.path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    build_labels([], self.path)

    def test_same_size_changed_checkpoint_ids_fail(self):
        changed = dict(self.mapping)
        a, b = self.labels[1:3]
        changed[a], changed[b] = changed[b], changed[a]
        config = SimpleNamespace(label2id=changed, id2label={i: l for l, i in changed.items()})
        with self.assertRaisesRegex(ValueError, "refusing head reset"):
            validate_checkpoint_labels(config, self.mapping)

    def test_frozen_loader_rejects_missing_or_mismatched_tensors(self):
        model = SimpleNamespace(config=SimpleNamespace(label2id=self.mapping,
                                                       id2label={i: l for l, i in self.mapping.items()}))
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            class FakeModel:
                @staticmethod
                def from_pretrained(path, **kwargs):
                    self.assertFalse(kwargs["ignore_mismatched_sizes"])
                    self.assertTrue(kwargs["output_loading_info"])
                    return model, {key: ["classifier.weight"]}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "exact learned tensors"):
                load_training_model(FakeModel, "unused", self.labels, self.mapping, frozen=True)

    def test_frozen_loader_returns_original_head_without_reinitialization(self):
        model = SimpleNamespace(config=SimpleNamespace(label2id=self.mapping,
                                                       id2label={i: l for l, i in self.mapping.items()}))
        class FakeModel:
            @staticmethod
            def from_pretrained(path, **kwargs):
                self.assertNotIn("label2id", kwargs)
                self.assertNotIn("num_labels", kwargs)
                self.assertFalse(kwargs["ignore_mismatched_sizes"])
                return model, {}
        self.assertIs(load_training_model(FakeModel, "unused", self.labels, self.mapping, frozen=True), model)

    def test_frozen_loader_cannot_hide_same_size_id_permutation(self):
        changed = dict(self.mapping)
        a, b = self.labels[1:3]
        changed[a], changed[b] = changed[b], changed[a]
        model = SimpleNamespace(config=SimpleNamespace(label2id=changed,
                                                       id2label={i: l for l, i in changed.items()}))
        class FakeModel:
            @staticmethod
            def from_pretrained(path, **kwargs):
                self.assertNotIn("label2id", kwargs)
                return model, {}
        with self.assertRaisesRegex(ValueError, "refusing head reset"):
            load_training_model(FakeModel, "unused", self.labels, self.mapping, frozen=True)

    def test_default_loader_retains_original_mismatch_behavior(self):
        sentinel = object()
        class FakeModel:
            @staticmethod
            def from_pretrained(path, **kwargs):
                self.assertTrue(kwargs["ignore_mismatched_sizes"])
                self.assertNotIn("output_loading_info", kwargs)
                return sentinel
        self.assertIs(load_training_model(FakeModel, "unused", self.labels, self.mapping), sentinel)


if __name__ == "__main__":
    unittest.main()
