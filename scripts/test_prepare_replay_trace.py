#!/usr/bin/env python3
"""Offline unit tests for 40-class coverage-preserving replay + trace assembly."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from prepare_replay_trace import (
    SplitContext, build_replay_trace, generate_payments, norm_hash, split_and_validate,
)

TYPES = ("GIVEN_NAME", "SURNAME", "EMAIL", "CREDIT_DEBIT_CARD", "CVV", "DATE", "IBAN", "SWIFT_BIC", "API_KEY")
LABEL2ID = {"O": 0, **{f"{p}-{t}": 2 * i + (1 if p == "B" else 2) for i, t in enumerate(TYPES) for p in ("B", "I")}}


class FakeTokenizer:
    def __call__(self, text, truncation=False, return_offsets_mapping=False):
        offsets = [(i, i + 1) for i in range(len(text))]
        out = {"input_ids": [1] + list(range(2, len(text) + 2)) + [2]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(0, 0), *offsets, (0, 0)]
        return out


def make_row(prefix: str, value: str, label: str, lang: str = "en") -> dict:
    text = f"{prefix}{value}\n"
    return {
        "text": text,
        "entities": [{"start": len(prefix), "end": len(prefix) + len(value), "label": label}],
        "language": lang,
    }


class ReplayTraceSplitTests(unittest.TestCase):
    def test_payment_generator_produces_valid_non_overlapping_spans(self):
        ctx = SplitContext(FakeTokenizer(), LABEL2ID, 512, set())
        rows = generate_payments(25, 11, False, ctx.blocked)
        self.assertEqual(len(rows), 25)
        for row in rows:
            pieces = split_and_validate(row, "payment_synth", ctx)
            self.assertEqual(len(pieces), 1)
            self.assertTrue(pieces[0]["entities"])

    def test_long_record_splits_without_truncating_entities(self):
        ctx = SplitContext(FakeTokenizer(), LABEL2ID, 20, set())
        row = {
            "text": "Line one: Alpha\nLine two: Beta\n",
            "entities": [
                {"start": 10, "end": 15, "label": "GIVEN_NAME"},
                {"start": 26, "end": 30, "label": "SURNAME"},
            ],
            "language": "en",
        }
        pieces = split_and_validate(row, "struct", ctx)
        self.assertEqual(len(pieces), 2)
        self.assertEqual("".join(p["text"] for p in pieces), row["text"])
        self.assertEqual([e["label"] for p in pieces for e in p["entities"]], ["GIVEN_NAME", "SURNAME"])


class ReplayTraceBuildTests(unittest.TestCase):
    def test_build_replay_trace_enforces_coverage_and_reserved_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "config.json"
            cfg.write_text(json.dumps({"label2id": LABEL2ID}), encoding="utf-8")
            trace_dir = root / "trace"
            openpii_dir = root / "openpii"
            trace_dir.mkdir()
            openpii_dir.mkdir()
            for split in ("train", "val", "test"):
                rows = [make_row(f"{split} {lang} {idx}: ", f"val_{idx}", label, lang)
                        for lang in ("en", "de") for idx, label in enumerate(TYPES)]
                (trace_dir / f"trace.{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            for name in ("optimizer", "selection", "holdout"):
                rows = [make_row(f"openpii {name} {i}: ", f"op_{i}", "EMAIL") for i in range(6)]
                (openpii_dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            blocked_row = make_row("struct blocked: ", "secret_val", "API_KEY")
            struct_rows = [blocked_row] + [make_row(f"struct {i}: ", f"st_{i}", TYPES[i % len(TYPES)]) for i in range(10)]
            for name, tag in (("struct.jsonl", "st"), ("gemma.jsonl", "gm"), ("real.jsonl", "rl")):
                pool = struct_rows if name == "struct.jsonl" else [
                    make_row(f"{tag} {i}: ", f"v_{i}", "GIVEN_NAME") for i in range(8)
                ]
                (root / name).write_text("".join(json.dumps(r) + "\n" for r in pool))
            reserved = root / "reserved.json"
            reserved.write_text(json.dumps([norm_hash(blocked_row["text"])]), encoding="utf-8")
            args = SimpleNamespace(
                struct_train=root / "struct.jsonl", gemma_train=root / "gemma.jsonl",
                real_train=root / "real.jsonl", trace_dir=trace_dir, openpii_dir=openpii_dir,
                reserved_hashes=reserved, label_config=cfg, output=root / "out",
                struct_count=4, wiki_each=2, openpii_train=2, openpii_eval=2,
                payment_train=3, payment_eval=2, max_tokens=128, seed=7,
            )
            report = build_replay_trace(args, FakeTokenizer())
            self.assertEqual(report["reserved_hash_overlaps"], 0)
            self.assertEqual(report["cross_partition_overlaps"], 0)
            for part in ("train", "selection", "holdout"):
                self.assertEqual(report["partitions"][part]["classes"], len(TYPES))
                lines = (root / "out" / f"{part}.jsonl").read_text(encoding="utf-8").splitlines()
                texts = [norm_hash(json.loads(l)["text"]) for l in lines if l.strip()]
                self.assertNotIn(norm_hash(blocked_row["text"]), texts)


if __name__ == "__main__":
    unittest.main()
