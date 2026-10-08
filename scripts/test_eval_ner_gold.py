#!/usr/bin/env python3
"""Synthetic-only full-value privacy/utility gate regressions."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eval_ner_gold import Native, gate, main, score, sha
from gold_ner import prepare
from json_safe import publish


def row(text, label=None, span=None, masks=None, source="a"):
    return {"schema": "nym.ner.gold.v1", "source_id": source, "unit_id": "one",
            "text": text, "text_sha256": sha(text.encode()), "annotation_complete": True,
            "entities": [] if label is None else [dict(start=span[0], end=span[1], label=label)],
            "masked_spans": masks or []}


def finding(text, start, end, pattern="ner_person"):
    return {"start": start, "end": end, "matched_text": text.encode()[start:end].decode(),
            "category": "other", "pattern_name": pattern}


class GoldEvalTests(unittest.TestCase):
    def test_unicode_complete_removal_and_negative(self):
        positive = row("🦉 Alice", "PERSON", (2, 7))
        negative = row("public port 8126", source="b")
        result = score([positive, negative], [[finding(positive["text"], 5, 10)], []], ["🦉 <PII>", negative["text"]])
        self.assertEqual(result, {"units": 2, "false_positive_findings": 0, "collateral_bytes": 0,
            "confirmed_negatives": 1, "preserved_negatives": 1, "masked_bytes": 0,
            "unknown_intersecting_findings": 0, "classes": {"PERSON": {"support": 1, "full_removed": 1}}})

    def test_partial_not_full_despite_known_name(self):
        positive = row("Alice", "PERSON", (0, 5))
        result = score([positive], [[finding("Alice", 0, 3)]], ["<PII>ce"])
        self.assertEqual(result["classes"], {"PERSON": {"support": 1, "full_removed": 0}})

    def test_unknown_not_false_negative_or_confirmed_negative(self):
        unknown = row("Alice port", masks=[{"start": 0, "end": 5}])
        result = score([unknown], [[finding(unknown["text"], 0, 5)]], ["<PII> port"])
        self.assertEqual(result["classes"], {})
        self.assertEqual((result["masked_bytes"], result["unknown_intersecting_findings"],
                          result["false_positive_findings"], result["confirmed_negatives"]), (5, 1, 0, 0))

    def test_collateral_bytes_and_false_findings(self):
        positive = row("Alice port", "PERSON", (0, 5))
        result = score([positive], [[finding(positive["text"], 0, 10)]], ["<PII>"])
        self.assertEqual((result["false_positive_findings"], result["collateral_bytes"]), (1, 5))

    def test_unreconciled_outputs_fail(self):
        positive = row("Alice", "PERSON", (0, 5))
        with self.assertRaises(ValueError):
            score([positive], [[]], ["changed outside findings"])
        with self.assertRaises(ValueError):
            score([positive], [], [])

    def test_nonmodel_control_must_be_empty_even_if_names_collide(self):
        with tempfile.TemporaryDirectory() as directory:
            native = Native.__new__(Native)
            native.config = Path(directory) / "config.toml"
            native.disabled = ["email"]
            native.invoke = lambda *args: b'[{"pattern_name":"email"}]'
            with patch("eval_ner_gold.cached_model", return_value=(Path(directory), {})):
                with self.assertRaisesRegex(ValueError, "not isolated"):
                    native.evaluate([row("synthetic")], directory)

    def test_recall_cannot_be_hidden_by_fewer_false_positives(self):
        base = score([row("Alice", "PERSON", (0, 5)), row("port", source="b")],
                     [[finding("Alice", 0, 5)], [finding("port", 0, 4)]], ["<PII>", "<PII>"])
        candidate = score([row("Alice", "PERSON", (0, 5)), row("port", source="b")], [[], []], ["Alice", "port"])
        verdict = gate(base, candidate, ["PERSON", "PIN"])
        self.assertEqual(verdict, {"eligible_for_further_gates": False, "regressed_classes": ["PERSON"],
            "incomplete_supported_classes": ["PERSON"], "unmeasured_classes": ["PIN"], "utility_improved": True,
            "status": "policy_blocked", "not_promotion_authorization": True})

    def test_eligible_is_not_promotion_and_requires_utility_improvement(self):
        rows = [row("Alice", "PERSON", (0, 5)), row("port", source="b")]
        baseline = score(rows, [[finding("Alice", 0, 5)], [finding("port", 0, 4)]], ["<PII>", "<PII>"])
        candidate = score(rows, [[finding("Alice", 0, 5)], []], ["<PII>", "port"])
        self.assertTrue(gate(baseline, candidate, ["PERSON"])["eligible_for_further_gates"])
        self.assertFalse(gate(candidate, candidate, ["PERSON"])["eligible_for_further_gates"])

    def test_failed_native_run_is_operational_and_preserves_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts = {"train": [row("Alice", "PERSON", (0, 5)), row("public port", source="b")],
                     "selection": [row("Carol", "PERSON", (0, 5), source="c")],
                     "holdout": [row("Doris", "PERSON", (0, 5), source="d")]}
            bundle = prepare([r for rows in parts.values() for r in rows],
                {"a": "train", "b": "train", "c": "selection", "d": "holdout"}, ["PERSON"])
            publish(root / "bundle.json", bundle)
            binary = root / "failed-nym"
            binary.write_text("#!/usr/bin/env bash\necho 'PRIVATE CHILD ERROR' >&2\nexit 1\n")
            binary.chmod(0o700)
            output = root / "report.json";output.write_text("preserve")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(["--gold-bundle", str(root / "bundle.json"), "--binary", str(binary),
                    "--baseline-model", str(root), "--candidate-model", str(root), "--output", str(output)])
            self.assertEqual(status, 1)
            self.assertEqual(output.read_text(), "preserve")
            self.assertNotIn("PRIVATE CHILD ERROR", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
