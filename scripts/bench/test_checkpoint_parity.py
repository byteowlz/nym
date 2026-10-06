#!/usr/bin/env python3
"""Synthetic numerical/contract regressions; run with uv and numpy, no models."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import checkpoint_parity as parity

EXPORTER = Path(__file__).resolve().parents[1] / "export_onnx.py"
spec = importlib.util.spec_from_file_location("export_onnx", EXPORTER)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def taxonomy():
    labels = {"0": "O"}
    for i in range(40):
        labels[str(2 * i + 1)] = f"B-TYPE_{i}"
        labels[str(2 * i + 2)] = f"I-TYPE_{i}"
    return {"id2label": labels, "label2id": {v: int(k) for k, v in labels.items()}}


class ComparisonTests(unittest.TestCase):
    def test_equal_full_array_passes(self):
        logits = np.array([[[2., 1., 0.], [0., 1., 2.]], [[1., 2., 0.], [2., 0., 1.]]])
        mask = np.array([[1, 0], [1, 1]])
        result = parity.compare_arrays(logits, logits.copy(), mask, 1e-3, 1e-4)
        self.assertEqual(result, {
            "max_logit_error": 0., "max_probability_error": 0.,
            "all": {"positions": 4, "argmax_disagreements": 0, "decision_disagreements": 0},
            "attended": {"positions": 3, "argmax_disagreements": 0, "decision_disagreements": 0},
            "masked": {"positions": 1, "argmax_disagreements": 0, "decision_disagreements": 0},
            "passed": True,
        })

    def test_threshold_crossing_fails_despite_same_argmax_and_loose_tolerance(self):
        # Same winning non-O class, but confidence crosses 0.5 in a three-class model.
        expected = np.array([[[0., np.log(2.01), 0.]]])
        actual = np.array([[[0., np.log(1.99), 0.]]])
        result = parity.compare_arrays(expected, actual, np.ones((1, 1)), 1., 1.)
        self.assertEqual(result["all"]["argmax_disagreements"], 0)
        self.assertEqual(result["all"]["decision_disagreements"], 1)
        self.assertFalse(result["passed"])

    def test_masked_second_row_disagreement_is_not_hidden(self):
        expected = np.array([[[2., 0.]], [[2., 0.]]])
        actual = np.array([[[2., 0.]], [[0., 2.]]])
        result = parity.compare_arrays(expected, actual, np.array([[1], [0]]), 10., 1.)
        self.assertEqual(result["attended"]["decision_disagreements"], 0)
        self.assertEqual(result["masked"]["decision_disagreements"], 1)
        self.assertFalse(result["passed"])

    def test_nan_shape_and_probability_failure(self):
        logits = np.array([[[2., 0.]]])
        for actual in (np.full_like(logits, np.nan), np.zeros((1, 2, 2))):
            with self.assertRaises(ValueError):
                parity.compare_arrays(logits, actual, np.ones((1, 1)), 1., 1.)
        changed = logits * 2
        result = parity.compare_arrays(logits, changed, np.ones((1, 1)), 10., 1e-4)
        self.assertFalse(result["passed"])
        self.assertEqual(result["all"]["argmax_disagreements"], 0)

    def test_stable_softmax(self):
        np.testing.assert_allclose(parity.softmax(np.array([[[10000., 10001.]]])),
                                   parity.softmax(np.array([[[0., 1.]]])))


class ContractTests(unittest.TestCase):
    def test_full_frozen_taxonomy(self):
        original = taxonomy()
        parity.validate_taxonomy(original, original)
        changed = json.loads(json.dumps(original))
        changed["id2label"]["1"], changed["id2label"]["3"] = changed["id2label"]["3"], changed["id2label"]["1"]
        with self.assertRaises(ValueError):
            parity.validate_taxonomy(changed, original)
        with self.assertRaises(ValueError):
            parity.validate_taxonomy({"id2label": {"0": "O"}, "label2id": {"O": 0}}, original)

    def test_dynamic_inputs_actual_vocab_and_mixed_padding(self):
        tokenizer = types.SimpleNamespace(no_truncation=lambda: None, no_padding=lambda: None,
                                          encode=lambda text: types.SimpleNamespace(ids=[1, 6, 2]))
        cases = parity.input_cases(tokenizer, 198804, 0, ["synthetic"])
        self.assertEqual([cases[f"random_{n}"][0].shape for n in parity.LENGTHS],
                         [(1, n) for n in parity.LENGTHS])
        for n in parity.LENGTHS:
            self.assertEqual(cases[f"random_{n}"][0][0, -1], 198803)
        for side in ("left", "right"):
            ids, mask = cases[f"padded_{side}"]
            self.assertEqual(mask.sum(axis=1).tolist(), [73, 301])
            self.assertTrue((ids[mask == 0] == 0).all())
            self.assertFalse(np.array_equal(ids[0], ids[1]))
            self.assertEqual(mask[:, 0].tolist(), [0, 0] if side == "left" else [1, 1])

    def test_tokenizer_truncation_only_change_not_vocabulary_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "tokenizer.json"
            raw = {"truncation": {"max_length": 256}, "padding": None, "model": {"vocab": {"a": 0}}}
            path.write_text(json.dumps(raw))
            original = parity.tokenizer_fingerprint(root)
            raw["truncation"] = None
            path.write_text(json.dumps(raw))
            self.assertEqual(parity.tokenizer_fingerprint(root), original)
            raw["model"]["vocab"]["a"] = 1
            path.write_text(json.dumps(raw))
            self.assertNotEqual(parity.tokenizer_fingerprint(root), original)

    def test_private_artifacts_reject_git(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            with self.assertRaises(ValueError):
                parity.private(root / "ignored" / "proof.json")

    def test_imports_do_not_load_either_backend(self):
        code = "import checkpoint_parity,sys; assert 'torch' not in sys.modules; assert 'onnxruntime' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parent, check=True)

    def test_exporter_detects_second_padded_row_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "references.npz"
            dump = {}
            for n in exporter.VERIFY_LENS:
                dump[f"ids_{n}"] = np.ones(n, dtype=np.int64)
                dump[f"logits_{n}"] = np.zeros((n, 3), dtype=np.float32)
            for side in ("left", "right"):
                dump[f"ids_{side}"] = np.ones((2, 384), dtype=np.int64)
                dump[f"mask_{side}"] = np.ones((2, 384), dtype=np.int64)
                dump[f"logits_{side}"] = np.zeros((2, 384, 3), dtype=np.float32)
            np.savez(path, **dump)
            class Session:
                def __init__(self, *args, **kwargs):
                    pass
                def run(self, outputs, feed):
                    logits = np.zeros((*feed["input_ids"].shape, 3), dtype=np.float32)
                    if logits.shape[0] == 2:
                        logits[1, -1, -1] = 1.
                    return [logits]
            fake = types.SimpleNamespace(InferenceSession=Session)
            with patch.dict(sys.modules, {"onnxruntime": fake}), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    exporter.verify_stage(str(path), "unused.onnx")
            self.assertEqual(caught.exception.code, 1)


class OutputReconciliationTests(unittest.TestCase):
    def test_supported_placeholder_does_not_make_overlapping_findings_provable(self):
        from run_trace_bench import reconcile

        source = "touch -t 203107230723 artifact"
        output = "touch -<PII> artifact"
        self.assertFalse(reconcile(source, output, [(7, 21), (11, 21)]))
        # The outer interval alone explains this exact output. The production
        # oracle is deliberately NOT changed to union or ignore nested findings.
        self.assertTrue(reconcile(source, output, [(7, 21)]))
        self.assertFalse(reconcile(source, "touch -<PIN203107230723> artifact", [(7, 21)]))

    @unittest.skipUnless(os.environ.get("NYM_PARITY_TEST_BINARY"), "set an immutable binary for native full-value proof")
    def test_native_complete_output_explains_overlap_rejection_without_oracle_relaxation(self):
        from run_trace_bench import Runner, score_case

        fixtures = [
            ("touch -t 203107230723 artifact", "touch -<PII> artifact", [
                {"category": "identity", "confidence": "low", "end": 21,
                 "matched_text": "t 203107230723", "path": "samples[0].text",
                 "pattern_name": "eu_id", "start": 7},
                {"category": "contact", "confidence": "high", "end": 21,
                 "matched_text": "3107230723", "path": "samples[0].text",
                 "pattern_name": "phone_us", "start": 11},
            ]),
            ("insurance ID 9118599561.", "insurance <PII>.", [
                {"category": "identity", "confidence": "low", "end": 23,
                 "matched_text": "ID 9118599561", "path": "samples[0].text",
                 "pattern_name": "eu_id", "start": 10},
                {"category": "contact", "confidence": "high", "end": 23,
                 "matched_text": "9118599561", "path": "samples[0].text",
                 "pattern_name": "phone_us", "start": 13},
            ]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            runner = Runner(os.environ["NYM_PARITY_TEST_BINARY"], directory)
            for source, transformed, expected_findings in fixtures:
                with self.subTest(source=source):
                    trace = {"samples": [{"text": source}]}
                    payload = json.dumps(trace)
                    findings = json.loads(runner.invoke("detect", payload, "json"))
                    output = json.loads(runner.invoke("anon", payload, "json"))
                    self.assertEqual(findings, expected_findings)
                    self.assertEqual(output, {"samples": [{"text": transformed}]})
                    case = {"id": "overlap-proof", "split": "holdout", "trace": trace,
                            "gold": [], "benign": []}
                    result = score_case(case, {"samples[0].text": findings}, output)
                    self.assertEqual(result["output_mismatches"], 1)
                    self.assertEqual(result["structure_errors"], 0)


if __name__ == "__main__":
    unittest.main()
