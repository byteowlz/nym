#!/usr/bin/env python3
"""Local model-only gold selection gate. No lexicons, downloads or activation."""
import argparse
import hashlib
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from gold_ner import load_bundle
from json_safe import publish


def sha(raw):
    return hashlib.sha256(raw).hexdigest()

sys.path.insert(0, str(Path(__file__).resolve().parent / "bench"))
from run_bench import byte_coverage, finding_ranges
from run_trace_bench import cached_model, reconcile


def byte_span(text, span):
    return len(text[:span["start"]].encode()), len(text[:span["end"]].encode())


def union(ranges):
    result = []
    for start, end in sorted(ranges):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def score(rows, detections, outputs):
    if len(rows) != len(detections) or len(rows) != len(outputs):
        raise ValueError("incomplete native gold results")
    classes = defaultdict(lambda: {"support": 0, "full_removed": 0})
    totals = {"units": len(rows), "false_positive_findings": 0,
              "collateral_bytes": 0, "confirmed_negatives": 0, "preserved_negatives": 0,
              "masked_bytes": 0, "unknown_intersecting_findings": 0}
    for row, findings, output in zip(rows, detections, outputs):
        text = row["text"]
        ranges = finding_ranges(text, findings)
        if not reconcile(text, output, ranges):
            raise ValueError("native output does not reconcile with original source")
        known = [byte_span(text, e) for e in row["entities"]]
        masks = [byte_span(text, m) for m in row["masked_spans"]]
        protected = union(known + masks)
        for entity, span in zip(row["entities"], known):
            metric = classes[entity["label"]]
            metric["support"] += 1
            metric["full_removed"] += byte_coverage(span, ranges) == span[1] - span[0]
        for span in ranges:
            collateral = span[1] - span[0] - byte_coverage(span, protected)
            totals["false_positive_findings"] += collateral > 0
            totals["unknown_intersecting_findings"] += byte_coverage(span, masks) > 0
        totals["collateral_bytes"] += sum(end - start - byte_coverage((start, end), protected)
                                           for start, end in union(ranges))
        totals["masked_bytes"] += sum(end - start for start, end in masks)
        if not known and not masks:
            totals["confirmed_negatives"] += 1
            totals["preserved_negatives"] += output == text
    return {**totals, "classes": dict(sorted(classes.items()))}


def gate(baseline, candidate, taxonomy):
    if baseline["units"] != candidate["units"] or baseline["classes"].keys() != candidate["classes"].keys():
        raise ValueError("incomparable gold evaluation support")
    regressions, incomplete = [], []
    for label, original in baseline["classes"].items():
        proposed = candidate["classes"][label]
        if proposed["support"] != original["support"]:
            raise ValueError("incomparable gold class support")
        if proposed["full_removed"] < original["full_removed"]:
            regressions.append(label)
        if proposed["full_removed"] != proposed["support"]:
            incomplete.append(label)
    unmeasured = sorted(set(taxonomy) - candidate["classes"].keys())
    utility_improved = (candidate["collateral_bytes"] < baseline["collateral_bytes"]
                        and candidate["false_positive_findings"] <= baseline["false_positive_findings"]
                        and candidate["preserved_negatives"] >= baseline["preserved_negatives"])
    eligible = bool(candidate["classes"]) and not regressions and not incomplete and utility_improved
    return {"eligible_for_further_gates": eligible, "regressed_classes": regressions,
            "incomplete_supported_classes": incomplete, "unmeasured_classes": unmeasured,
            "utility_improved": utility_improved,
            "status": "selection_eligible" if eligible else "policy_blocked",
            "not_promotion_authorization": True}


class Native:
    def __init__(self, binary, directory, timeout):
        self.binary, self.directory, self.timeout = str(binary), Path(directory), timeout
        self.config = self.directory / "config.toml"
        self.config.write_text("[ner]\nenabled=false\n[decision]\nenabled=false\n")
        self.env = {key: value for key, value in os.environ.items() if not key.startswith("NYM_")}
        self.env.update(HOME=str(directory), XDG_CONFIG_HOME=str(directory),
                        XDG_DATA_HOME=str(directory), XDG_STATE_HOME=str(directory),
                        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        registry = json.loads(self.invoke(["--json", "patterns"]))
        if not isinstance(registry, list) or not registry or any(
                not isinstance(p, dict) or not isinstance(p.get("name"), str) for p in registry):
            raise ValueError("invalid native pattern registry")
        self.disabled = [p["name"] for p in registry]

    def invoke(self, args, payload=None):
        result = subprocess.run([self.binary, "--config", str(self.config), "--quiet", "--no-color", *args],
                                input=payload, capture_output=True, timeout=self.timeout,
                                env=self.env, cwd=self.directory)
        if result.returncode:
            raise ValueError("native gold evaluation incomplete")
        return result.stdout

    def evaluate(self, rows, model):
        model, before = cached_model(model)
        self.config.write_text("[detection]\ndisabled_patterns=" + json.dumps(self.disabled) + "\n"
            + "[ner]\nenabled=true\nbackend='tokens'\nprovider='cpu'\nthreshold=0.5\nrecall_first=false\ntoken_model="
            + json.dumps(str(model)) + "\n[decision]\nenabled=false\n")
        payload = b"".join((json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n").encode() for r in rows)
        opts = ["--format", "jsonl", "--include-path", "text", "--ner"]
        # Native NER names can equal regex names; a prefix is not provenance.
        control = json.loads(self.invoke(["detect", "--json", *opts[:-1], "--no-ner"], payload))
        if control != []:
            raise ValueError("non-model detectors are not isolated")
        findings = json.loads(self.invoke(["detect", "--json", *opts], payload))
        if not isinstance(findings, list):
            raise ValueError("invalid native findings")
        by_path = {f"record[{i + 1}].text": [] for i in range(len(rows))}
        for finding in findings:
            if not isinstance(finding, dict) or finding.get("path") not in by_path:
                raise ValueError("unreconciled native findings")
            by_path[finding["path"]].append(finding)
        raw_output = self.invoke(["anon", *opts, "--strategy", "placeholder"], payload)
        outputs = [json.loads(line) for line in raw_output.splitlines()]
        if any(not isinstance(row, dict) or set(row) != {"text"} or not isinstance(row["text"], str) for row in outputs):
            raise ValueError("invalid native anonymized output")
        _, after = cached_model(model)
        if before != after:
            raise ValueError("model assets changed during evaluation")
        return score(rows, list(by_path.values()), [r["text"] for r in outputs]), before


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-bundle", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args(argv)
    try:
        if args.timeout <= 0:
            raise ValueError("invalid evaluation timeout")
        bundle = load_bundle(args.gold_bundle)
        binary = args.binary.resolve(strict=True)
        binary_sha = sha(binary.read_bytes())
        with tempfile.TemporaryDirectory(prefix="nym-gold-eval-") as directory:
            native = Native(binary, directory, args.timeout)
            baseline, baseline_assets = native.evaluate(bundle["selection"], args.baseline_model)
            candidate, candidate_assets = native.evaluate(bundle["selection"], args.candidate_model)
        if sha(binary.read_bytes()) != binary_sha:
            raise ValueError("native binary changed during evaluation")
        verdict = gate(baseline, candidate, bundle["label_types"])
        report = {"schema": "nym.ner.gold.eval.v1", "component": "model_only", "threshold": 0.5,
                  "split": "selection", "nonmodel_control_empty": True, "bundle_sha256": bundle["payload_sha256"], "binary_sha256": binary_sha,
                  "baseline_assets": baseline_assets, "candidate_assets": candidate_assets,
                  "baseline": baseline, "candidate": candidate, "verdict": verdict}
        publish(args.output, report)
        print(json.dumps({"status": verdict["status"], "units": candidate["units"]}))
        return 0 if verdict["eligible_for_further_gates"] else 2
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError):
        print("gold evaluation failed; no completed accuracy result", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
