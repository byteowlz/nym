#!/usr/bin/env python3
"""Offline synthetic benchmark; emit only value/path/mapping-free aggregates.

Run via uv run --no-project scripts/bench/run_bench.py. Recall gates require
full UTF-8 byte-span coverage. Partial coverage is reported separately.
NYM_BIN overrides the executable; NYM_NER_CONFIG supplies an explicit local
NER configuration. No model is downloaded by the default regex-only recipe.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
BIN = os.environ.get("NYM_BIN", str(ROOT / "target/bench/release/nym"))
FIXTURES_DEFAULT = str(ROOT / "scripts/bench/fixtures/challenge.json")
REGEX_IMPLICIT_CLASSES = {
    "person", "unlabeled_high_entropy", "internal_hostname", "codename", "organization",
}
CATEGORIES = {"contact", "identity", "financial", "network", "authentication", "social", "other"}


def literal_ranges(text, value):
    """Resolve every occurrence to a half-open UTF-8 byte range."""
    if not isinstance(value, str) or not value:
        raise ValueError("empty or invalid annotation")
    raw, needle = text.encode("utf-8"), value.encode("utf-8")
    ranges = []
    start = raw.find(needle)
    while start >= 0:
        ranges.append((start, start + len(needle)))
        start = raw.find(needle, start + 1)
    if not ranges:
        raise ValueError("absent annotation")
    return ranges


def overlaps(left, right):
    return left[0] < right[1] and right[0] < left[1]


def validate_fixture(fixture):
    """Reject invalid labels before detection; deduplicate annotations, not occurrences."""
    if not isinstance(fixture, dict) or not isinstance(fixture.get("text"), str):
        raise ValueError("invalid fixture text")
    if any(not isinstance(fixture.get(key, []), list) for key in ("spans", "benign", "out_of_regex_scope")):
        raise ValueError("invalid annotation list")
    text = fixture["text"]
    gold = {}
    for span in fixture.get("spans", []):
        if not isinstance(span, dict) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", str(span.get("class", ""))):
            raise ValueError("invalid annotation class")
        value, cls = span.get("value"), span["class"]
        ranges = literal_ranges(text, value)
        gold[(cls, value)] = ranges
    benign = {value: literal_ranges(text, value) for value in fixture.get("benign", [])}
    gold_ranges = [span for ranges in gold.values() for span in ranges]
    if any(overlaps(g, b) for g in gold_ranges for ranges in benign.values() for b in ranges):
        raise ValueError("conflicting gold and benign annotations")
    scope = fixture.get("out_of_regex_scope", [])
    if any(cls not in {key[0] for key in gold} for cls in scope):
        raise ValueError("invalid scope annotation")
    return gold, benign


def finding_ranges(text, findings):
    """Validate nym offsets against the exact input bytes (including UTF-8 boundaries)."""
    if not isinstance(findings, list):
        raise ValueError("invalid findings")
    raw = text.encode("utf-8")
    ranges = []
    for finding in findings:
        if not isinstance(finding, dict):
            raise ValueError("invalid finding")
        start, end = finding.get("start"), finding.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(raw):
            raise ValueError("invalid finding offset")
        try:
            matched = raw[start:end].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("invalid UTF-8 boundary") from error
        if finding.get("matched_text") != matched or finding.get("category") not in CATEGORIES:
            raise ValueError("invalid finding content")
        ranges.append((start, end))
    return ranges


def byte_coverage(gold, findings):
    """Union intersecting ranges so duplicates/overlaps never inflate coverage."""
    cursor, covered = gold[0], 0
    for start, end in sorted(findings):
        start, end = max(start, cursor, gold[0]), min(end, gold[1])
        if end > start:
            covered += end - start
            cursor = end
    return covered


def score_fixture(fixture, findings, anonymized):
    gold, benign = validate_fixture(fixture)
    ranges = finding_ranges(fixture["text"], findings)
    classes = defaultdict(lambda: {"gold": 0, "full": 0, "partial": 0, "fp": 0})
    gold_ranges = []
    for (cls, _value), spans in gold.items():
        for span in spans:
            gold_ranges.append(span)
            covered = byte_coverage(span, ranges)
            row = classes[cls]
            row["gold"] += 1
            row["full"] += covered == span[1] - span[0]
            row["partial"] += 0 < covered < span[1] - span[0]
    for finding, span in zip(findings, ranges):
        if not any(overlaps(span, gold_span) for gold_span in gold_ranges):
            classes[finding["category"]]["fp"] += 1
    # A distinct benign annotation counts once, even with repeated occurrences or
    # multiple findings. Actual preservation requires every literal occurrence
    # to survive in anon stdout; it is not inferred from detection success.
    flagged = sum(any(overlaps(b, f) for b in spans for f in ranges) for spans in benign.values())
    preserved = sum(len(re.findall(f"(?={re.escape(value)})", anonymized)) >= len(spans)
                    for value, spans in benign.items())
    return {"classes": dict(classes), "benign": {"total": len(benign), "flagged": flagged, "preserved": preserved}}


def run_nym(text, command, use_ner, config):
    args = [BIN, command, "--format", "text", "--min-confidence", "low", "--ruleset", "all", "--quiet", "--no-color"]
    args += ["--ner" if use_ner else "--no-ner"]
    if config:
        args += ["--config", config]
    if command == "detect":
        args += ["--json"]
    else:
        args += ["--strategy", "placeholder", "--seed", "42", "--context", "synthetic-bench", "--tag", "bench"]
    try:
        proc = subprocess.run(args, input=text.encode("utf-8"), capture_output=True, timeout=120)
        if proc.returncode:
            return None, "nym invocation failed"
        return proc.stdout.decode("utf-8"), None
    except (OSError, subprocess.TimeoutExpired, UnicodeDecodeError):
        return None, "nym invocation failed"


def run_detect(text, use_ner, config):
    output, error = run_nym(text, "detect", use_ner, config)
    if error:
        return None, error
    try:
        return json.loads(output), None
    except json.JSONDecodeError:
        return None, "invalid detection JSON"


def run_anon(text, use_ner, config):
    return run_nym(text, "anon", use_ner, config)


def sha256_file(path):
    try:
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        return None


def git_output(*args):
    try:
        proc = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10)
        return proc.stdout.strip() if proc.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def provenance(fixture_path, doc, binary, fixture_bytes=None):
    """Read this invocation's checkout and content hashes, never shared temp state."""
    status = git_output("status", "--porcelain")
    return {
        "git": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "binary_sha256": sha256_file(binary),
        "fixture_sha256": (hashlib.sha256(fixture_bytes).hexdigest()
                           if fixture_bytes is not None else sha256_file(fixture_path)),
        "fixture_hashes": [hashlib.sha256(json.dumps(fx, sort_keys=True, ensure_ascii=False,
                                                   separators=(",", ":")).encode("utf-8")).hexdigest()
                           for fx in doc["fixtures"]],
    }


def evaluate(fixtures, use_ner, config):
    classes = defaultdict(lambda: {"gold": 0, "full": 0, "partial": 0, "fp": 0, "in_scope_gold": 0, "in_scope_full": 0})
    benign = {"total": 0, "flagged": 0, "preserved": 0}
    errors = 0
    for fixture in fixtures:
        findings, detect_error = run_detect(fixture["text"], use_ner, config)
        anonymized, anon_error = run_anon(fixture["text"], use_ner, config)
        if detect_error or anon_error:
            errors += 1
            continue
        try:
            score = score_fixture(fixture, findings, anonymized)
        except (ValueError, TypeError):
            errors += 1
            continue
        scope = set(fixture.get("out_of_regex_scope", [])) | REGEX_IMPLICIT_CLASSES
        for cls, counts in score["classes"].items():
            row = classes[cls]
            for key, count in counts.items():
                row[key] += count
            if use_ner or cls not in scope:
                row["in_scope_gold"] += counts["gold"]
                row["in_scope_full"] += counts["full"]
        for key, count in score["benign"].items():
            benign[key] += count
    return classes, benign, errors


def class_rows(classes):
    rows = []
    for cls, counts in sorted(classes.items()):
        gold, in_scope = counts["gold"], counts["in_scope_gold"]
        rows.append({
            "class": cls, "gold": gold, "full_spans": counts["full"], "partial_spans": counts["partial"],
            "recall": counts["full"] / gold if gold else 0.0,
            "partial_recall": counts["partial"] / gold if gold else 0.0,
            "false_positives": counts["fp"], "in_scope_gold": in_scope,
            "in_scope_recall": counts["in_scope_full"] / in_scope if in_scope else 0.0,
            "out_of_regex_scope": bool(gold and not in_scope),
        })
    return rows


def report(summary, as_json):
    if as_json:
        print(json.dumps(summary, indent=2))
        return
    print("nym synthetic benchmark (offline, safe aggregate only)")
    for key in ("fixture_version", "mode", "nym", "git", "git_dirty", "binary_sha256", "fixture_sha256", "fixtures", "errors"):
        print(f"  {key:<20}: {summary[key]}")
    print(f"  benign actual        : {summary['benign_preservation']:.3f} "
          f"({summary['benign_literals_preserved']}/{summary['benign_literals_total']})")
    print(f"  benign flagged       : {summary['benign_literals_flagged']}")
    for row in summary["per_class"]:
        scope = " (out of regex scope)" if row["out_of_regex_scope"] else ""
        print(f"    {row['class']:<22} gold={row['gold']:>2} full={row['recall']:.3f} "
              f"partial={row['partial_recall']:.3f} fp={row['false_positives']}{scope}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ner", action="store_true", help="enable a locally cached NER backend")
    parser.add_argument("--no-ner", dest="ner", action="store_false")
    parser.set_defaults(ner=False)
    parser.add_argument("--fail-on-recall", type=float, help="gate full byte-span recall for in-scope gold occurrences")
    parser.add_argument("--fail-on-benign", action="store_true", help="fail on detection overlap or actual loss of a benign literal")
    parser.add_argument("--fixtures", default=FIXTURES_DEFAULT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.fail_on_recall is not None and not 0 <= args.fail_on_recall <= 1:
        parser.error("recall threshold must be between 0 and 1")
    try:
        fixture_bytes = Path(args.fixtures).read_bytes()
        doc = json.loads(fixture_bytes.decode("utf-8"))
        if not isinstance(doc, dict) or doc.get("version") != "1" or not isinstance(doc.get("fixtures"), list) or not doc["fixtures"]:
            raise ValueError("invalid fixture document")
        for fixture in doc["fixtures"]:
            validate_fixture(fixture)
    except (OSError, ValueError, TypeError):
        print(json.dumps({"errors": 1, "error": "fixture validation failed"}))
        return 1

    binary = shutil.which(BIN) or BIN
    pinned = provenance(args.fixtures, doc, binary, fixture_bytes)
    version, version_error = run_version(binary)
    config = os.environ.get("NYM_NER_CONFIG") if args.ner else None
    classes, benign, errors = evaluate(doc["fixtures"], args.ner, config)
    errors += bool(version_error) + (pinned["binary_sha256"] is None)
    total = benign["total"]
    summary = {
        "fixture_version": doc["version"], "mode": "ner" if args.ner else "regex-only", "nym": {"nym": version},
        **pinned, "fixtures": len(doc["fixtures"]), "errors": errors, "per_class": class_rows(classes),
        "benign_literals_total": total, "benign_literals_flagged": benign["flagged"],
        "benign_literals_preserved": benign["preserved"],
        "benign_preservation": benign["preserved"] / total if total else 1.0,
        "benign_detection_preservation": 1 - benign["flagged"] / total if total else 1.0,
    }
    report(summary, args.json)
    missed_gate = args.fail_on_recall is not None and any(
        row["in_scope_gold"] and row["in_scope_recall"] < args.fail_on_recall for row in summary["per_class"])
    benign_gate = args.fail_on_benign and (benign["flagged"] or benign["preserved"] < total)
    if errors or missed_gate or benign_gate:
        print("REGRESSION: validation/execution errors or aggregate gate failed", file=sys.stderr)
        return 1
    return 0


def run_version(binary):
    try:
        proc = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=10)
        version = proc.stdout.strip()
        # Plain release versions, or build.rs provenance stamps:
        # nym 0.3.0+g<rev>.<state> (features=a,b;target=<triple>)
        if proc.returncode == 0 and re.fullmatch(
                r"nym [0-9A-Za-z.+-]+(?: \(features=[0-9a-z,_-]+;target=[0-9A-Za-z_.-]+\))?", version):
            return version, None
    except (OSError, subprocess.TimeoutExpired):
        return "unknown", "version unavailable"
    return "unknown", "version unavailable"


if __name__ == "__main__":
    sys.exit(main())
