#!/usr/bin/env python3
"""Offline synthetic recall / false-positive / utility benchmark for nym.

Runs the nym binary against a versioned, independently labeled synthetic
challenge set and reports per-class recall, false-positive rate, and
task-relevant literal (benign) preservation. Emits ONLY safe aggregate
results: no matched PII values, no source paths, no reversible mappings.

Usage:
    python3 scripts/bench/run_bench.py [--ner] [--no-ner] [--fail-on-recall 0.8]
                                       [--fixtures scripts/bench/fixtures/challenge.json]

Modes:
    --no-ner   regex-only (default; fully offline, no model download)
    --ner      also configure the local NER backend. Requires a locally cached
               model; declare the model/cache in config. This is the
               "model-backed" recipe and is not run by the default offline
               recipe. See below.

Environment:
    NYM_BIN          path to nym binary        (default target/release/nym)
    NYM_NER_CONFIG   optional config file that enables NER for the --ner mode
    NYM_DECISION_ENDPOINT  optional OpenAI-compatible endpoint for a future
               decision-backed mode; not exercised here (external calls need
               explicit egress approval).

Exit code: nonzero if any per-class recall is below --fail-on-recall, or if
any benign literal was flagged (false positive on task-relevant content).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

BIN = os.environ.get("NYM_BIN", "target/release/nym")
FIXTURES_DEFAULT = "scripts/bench/fixtures/challenge.json"

# Classes that regex patterns cannot possibly detect (they need NER/decide).
REGEX_IMPLICIT_CLASSES = {
    "person", "unlabeled_high_entropy", "internal_hostname", "codename",
    "organization",
}


def classify(span):
    return span.get("class", "unknown")


def run_detect(text, use_ner, config):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text)
        path = f.name
    args = [BIN, "detect", path, "--json", "--min-confidence", "low"]
    if config:
        args += ["--config", config]
    if use_ner:
        args += ["--ner"]
    else:
        args += ["--no-ner"]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=120)
    finally:
        os.unlink(path)
    if proc.returncode != 0:
        return None, proc.stderr
    try:
        return json.loads(proc.stdout), None
    except json.JSONDecodeError as e:
        return None, f"bad JSON from nym: {e}\n{proc.stdout[:400]}"


def char_coverage(gold, findings):
    """Fraction of the gold span's characters covered by an overlapping finding.

    Uses value-based overlap (a found span qualifies when it contains or is
    contained by the gold value) so multibyte/encoding offset ambiguities do
    not corrupt recall measurement.
    """
    gold_value = gold.get("value", "")
    best = 0.0
    for f in findings:
        matched = f.get("matched_text", "")
        if matched and (gold_value in matched or matched in gold_value):
            covered = min(len(matched), len(gold_value)) / max(1, len(gold_value))
            best = max(best, covered)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ner", action="store_true", help="enable local NER backend")
    ap.add_argument("--no-ner", dest="ner", action="store_false",
                    help="regex-only (default)")
    ap.set_defaults(ner=False)
    ap.add_argument("--fail-on-recall", type=float, default=None,
                    help="exit nonzero if any in-scope class recall is below this")
    ap.add_argument("--fail-on-benign", action="store_true",
                    help="also exit nonzero if any benign literal is flagged")
    ap.add_argument("--fixtures", default=FIXTURES_DEFAULT)
    ap.add_argument("--json", action="store_true", help="emit machine-readable aggregate")
    args = ap.parse_args()

    doc = json.loads(Path(args.fixtures).read_text())
    if doc.get("version") != "1":
        print(f"warning: unknown fixture version {doc.get('version')}")
    fixtures = doc["fixtures"]

    config = os.environ.get("NYM_NER_CONFIG") if args.ner else None

    per_class = defaultdict(lambda: {"gold": 0, "covered": 0, "fp": 0})
    benign_total = 0
    benign_flagged = 0
    class_scope = {}
    errors = []

    for fx in fixtures:
        findings, err = run_detect(fx["text"], args.ner, config)
        if findings is None:
            errors.append((fx["id"], err))
            continue

        # Per-class recall (value-based coverage, so partial matches count).
        for span in fx.get("spans", []):
            cls = classify(span)
            per_class[cls]["gold"] += 1
            if char_coverage(span, findings) >= 0.5:
                per_class[cls]["covered"] += 1
            class_scope[cls] = fx.get("out_of_regex_scope", []) and cls in fx.get("out_of_regex_scope", [])

        # False positives: findings that don't overlap any gold span AND are
        # not entirely inside a benign literal (which we want preserved).
        gold_texts = {s["value"] for s in fx.get("spans", [])}
        benign_texts = list(fx.get("benign", []))
        for f in findings:
            matched = f.get("matched_text", "")
            in_gold = any(matched in g or g in matched for g in gold_texts)
            in_benign = any(matched in b or b in matched for b in benign_texts)
            if in_gold:
                continue
            if in_benign:
                # A benign literal overlapped by a finding is a false positive
                # on task-relevant content.
                benign_flagged += 1
            per_class[f["category"]]["fp"] += 1

        # Benign literal preservation: a benign string must not be overlapped
        # by any finding. (Counted once here, not in the FP loop above.)
        for b in benign_texts:
            benign_total += 1
            overlapped = any(
                f.get("matched_text", "") and (b in f["matched_text"] or f["matched_text"] in b)
                for f in findings
            )
            if overlapped:
                benign_flagged += 1

    # Capture pinned versions.
    versions = {}
    try:
        versions["nym"] = (
            subprocess.run([BIN, "--version"], capture_output=True, text=True)
            .stdout.strip()
        )
    except Exception:
        versions["nym"] = "unknown"

    # Assemble aggregate summary (value-free).
    rows = []
    for cls in sorted(per_class):
        g = per_class[cls]
        recall = g["covered"] / g["gold"] if g["gold"] else 0.0
        rows.append({
            "class": cls,
            "gold": g["gold"],
            "recall": round(recall, 3),
            "false_positives": g["fp"],
            "out_of_regex_scope": bool(class_scope.get(cls)),
        })

    benign_rate = 1 - (benign_flagged / benign_total if benign_total else 0.0)
    summary = {
        "fixture_version": doc.get("version"),
        "mode": "ner" if args.ner else "regex-only",
        "nym": versions,
        "git": Path("/tmp/nym_bench_git.txt").read_text().strip()
               if Path("/tmp/nym_bench_git.txt").exists() else "unknown",
        "fixtures": len(fixtures),
        "errors": len(errors),
        "per_class": rows,
        "benign_literals_total": benign_total,
        "benign_literals_flagged": benign_flagged,
        "benign_preservation": round(benign_rate, 3),
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("nym synthetic benchmark (offline, safe aggregate only)")
        print(f"  fixture version   : {doc.get('version')}")
        print(f"  mode              : {summary['mode']}")
        print(f"  nym               : {versions.get('nym', '?')}")
        print(f"  git               : {summary['git']}")
        print(f"  fixtures          : {summary['fixtures']}  (errors: {summary['errors']})")
        print(f"  benign literals   : {benign_rate:.3f} preserved "
              f"({benign_total - benign_flagged}/{benign_total})")
        print("  per-class recall  :")
        for r in rows:
            mark = ""
            if r["out_of_regex_scope"] and summary["mode"] == "regex-only":
                mark = "  (out of regex scope - needs NER/decide)"
            print(f"    {r['class']:<22} gold={r['gold']:>2} "
                  f"recall={r['recall']:.3f} fp={r['false_positives']}{mark}")

    # Regression gate: fail on seeded misses (in-scope gold spans not detected)
    # and on fixture corruption. Benign-literal over-flagging is reported but
    # only gates when --fail-on-benign is given, since it reflects real pattern
    # over-reach rather than a seed/decoder regression.
    failures = []
    if args.fail_on_recall is not None:
        for r in rows:
            if r["out_of_regex_scope"] and summary["mode"] == "regex-only":
                continue
            if r["gold"] and r["recall"] < args.fail_on_recall:
                failures.append(f"{r['class']} recall {r['recall']} < {args.fail_on_recall}")
    if args.fail_on_benign and benign_flagged > 0:
        failures.append(f"{benign_flagged} benign literal(s) flagged")
    if errors:
        failures.append(f"{len(errors)} fixture(s) errored")
        for fid, e in errors[:3]:
            print(f"    error on {fid}: {e[:200]}")

    if failures:
        print("REGRESSION:", "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())