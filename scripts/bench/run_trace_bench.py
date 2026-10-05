#!/usr/bin/env python3
"""Synthetic pi-shaped precision/utility gate. Parsed JSON is the intended path.

No private input, downloads or persistent config writes. See TRACE_BENCHMARK.md.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from run_bench import byte_coverage, finding_ranges, git_output, run_version, sha256_file

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).with_name("fixtures") / "agent_traces.json"
# UUID matching is retained for identity metadata; only schema-specific trace
# IDs are excluded. No blanket metadata exclusion or value-based safe list.
EXCLUDED = "date,time,ipv4,ipv6,unix_path,social_url"
RECORD_TYPES = {"message", "session", "model_change", "thinking_level_change", "compaction", "custom"}
# Independent placeholder contract: arbitrary <TOKEN> could itself leak a PIN
# or key. A new native placeholder needs explicit review, not fuzzy acceptance.
PLACEHOLDERS = ("EMAIL", "PHONE", "SSN", "CREDIT_CARD", "IPV4", "IPV6", "MAC", "UUID", "JWT",
                "API_KEY", "IBAN", "PASSPORT", "HANDLE", "SOCIAL_URL", "DATE", "TIME", "USERNAME",
                "HOME_DIR", "PATH", "PII")
PLACEHOLDER_PATTERN = "(?:" + "|".join(f"<{name}>" for name in PLACEHOLDERS) + ")"


def leaves(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key):
                raise ValueError("unsupported fixture key")
            yield from leaves(item, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, f"{path}[{index}]")
    else:
        yield path, value


def structural_paths(trace):
    """Exempt only recognized trace shapes, not arbitrary fields named id."""
    paths = set()
    session = trace.get("session", {})
    if isinstance(session, dict) and session.get("type") == "session":
        paths.add("session.id")
    records = trace.get("records", [])
    if not isinstance(records, list):
        raise ValueError("invalid trace records")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("type") not in RECORD_TYPES:
            continue
        prefix = f"records[{index}]"
        paths.update(f"{prefix}.{key}" for key in ("id", "parentId", "timestamp") if key in record)
        message = record.get("message", {})
        if not isinstance(message, dict):
            continue
        if message.get("role") == "toolResult":
            paths.add(f"{prefix}.message.toolCallId")
        content = message.get("content", [])
        if isinstance(content, list):
            paths.update(f"{prefix}.message.content[{block}].id" for block, item in enumerate(content)
                         if isinstance(item, dict) and item.get("type") == "toolCall")
    return paths & dict(leaves(trace)).keys()


def scan_paths(trace):
    structural = structural_paths(trace)
    return {path: value for path, value in leaves(trace)
            if isinstance(value, str) and path not in structural}


def validate_relationships(trace):
    session = trace.get("session", {})
    seen = {session["id"]} if isinstance(session, dict) and "id" in session else set()
    calls = set()
    for record in trace.get("records", []):
        if not isinstance(record, dict) or record.get("type") not in RECORD_TYPES:
            continue
        uid, parent = record.get("id"), record.get("parentId")
        if not isinstance(uid, str) or uid in seen or (parent is not None and parent not in seen):
            raise ValueError("invalid trace id relationship")
        seen.add(uid)
        message = record.get("message", {})
        if not isinstance(message, dict):
            continue
        if message.get("role") == "toolResult" and message.get("toolCallId") not in calls:
            raise ValueError("invalid tool relationship")
        content = message.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "toolCall":
                    uid = item.get("id")
                    if not isinstance(uid, str) or uid in calls:
                        raise ValueError("invalid tool call id")
                    calls.add(uid)


def validate_case(case):
    if (not isinstance(case, dict) or not re.fullmatch(r"[a-z0-9-]+", str(case.get("id", "")))
            or case.get("split") not in {"selection", "holdout"}
            or not isinstance(case.get("trace"), dict)):
        raise ValueError("invalid case")
    strings = scan_paths(case["trace"])
    validate_relationships(case["trace"])
    seen = defaultdict(list)
    for kind in ("gold", "benign"):
        if not isinstance(case.get(kind), list):
            raise ValueError("invalid annotations")
        for row in case[kind]:
            if not isinstance(row, dict) or row.get("path") not in strings:
                raise ValueError("invalid annotation path")
            text = strings[row["path"]]
            start, end, value = row.get("start"), row.get("end"), row.get("value")
            if (type(start) is not int or type(end) is not int or not isinstance(value, str)
                    or not value or not 0 <= start < end <= len(text.encode())):
                raise ValueError("invalid annotation range")
            try:
                if text.encode()[start:end].decode() != value:
                    raise ValueError("annotation bytes disagree")
            except UnicodeDecodeError as error:
                raise ValueError("annotation splits UTF-8") from error
            if any(start < right and left < end for left, right in seen[row["path"]]):
                raise ValueError("duplicate or overlapping annotation")
            seen[row["path"]].append((start, end))
            if kind == "gold" and (not value.strip() or row.get("scope") not in {"regex", "ner", "policy"}
                                   or not re.fullmatch(r"[a-z][a-z0-9_]+", str(row.get("class", "")))):
                raise ValueError("invalid scope/class")


def validate_document(doc):
    if (not isinstance(doc, dict) or doc.get("version") != "agent-traces-v1"
            or doc.get("origin") != "handwritten-synthetic-only"
            or not isinstance(doc.get("cases"), list) or not doc["cases"]):
        raise ValueError("invalid document")
    ids, scopes = set(), {}
    for case in doc["cases"]:
        validate_case(case)
        if case["id"] in ids:
            raise ValueError("duplicate case id")
        ids.add(case["id"])
        for row in case["gold"]:
            if scopes.setdefault(row["class"], row["scope"]) != row["scope"]:
                raise ValueError("inconsistent class scope")
    if {case["split"] for case in doc["cases"]} != {"selection", "holdout"}:
        raise ValueError("missing split")


def required_ranges(text, start, end):
    """All UTF-8 bytes of non-whitespace characters, including letters/marks."""
    offset = start
    result = []
    for char in text.encode()[start:end].decode():
        size = len(char.encode())
        if not char.isspace():
            result.append((offset, offset + size))
        offset += size
    return result


def reconcile(text, output, ranges):
    """Prove output is unchanged chunks plus placeholder-only replacements.

    No fuzzy alignment or substring-only leak test. Only independently approved
    placeholders are accepted; arbitrary tokens could embed a secret. Overlapping
    findings cannot establish this proof and fail closed.
    Duplicate identical ranges are deduplicated. Whitespace gaps remain visible.
    """
    if not isinstance(output, str):
        return False
    raw, cursor, parts = text.encode(), 0, []
    for start, end in sorted(set(ranges)):
        if start < cursor:
            return False
        parts.extend([re.escape(raw[cursor:start].decode()), PLACEHOLDER_PATTERN])
        cursor = end
    parts.append(re.escape(raw[cursor:].decode()))
    return re.fullmatch("".join(parts), output) is not None


def structure_errors(before, after):
    if type(before) is not type(after):
        return 1
    if isinstance(before, dict):
        if before.keys() != after.keys():
            return 1
        return sum(structure_errors(before[key], after[key]) for key in before)
    if isinstance(before, list):
        if len(before) != len(after):
            return 1
        return sum(structure_errors(left, right) for left, right in zip(before, after))
    return int(not isinstance(before, str) and before != after)


def score_case(case, detections, output):
    validate_case(case)
    strings = scan_paths(case["trace"])
    if set(detections) - set(strings):
        raise ValueError("unexpected detection path")
    after = dict(leaves(output))
    classes = {}
    counts = {"findings": 0, "false_positive_findings": 0, "detected_bytes": 0,
              "gold_bytes": 0, "false_positive_bytes": 0}
    benign = {"occurrences": 0, "flagged": 0, "preserved": 0}
    occurrences, finding_offsets, benign_occurrences = [], [], []
    mismatches = 0
    for field, (path, text) in enumerate(strings.items()):
        findings = detections.get(path, [])
        ranges = finding_ranges(text, findings)
        finding_offsets.extend({"field": field, "start": start, "end": end,
                                "category": finding["category"]}
                               for finding, (start, end) in zip(findings, ranges))
        proven = reconcile(text, after.get(path), ranges)
        mismatches += not proven
        gold = [row for row in case["gold"] if row["path"] == path]
        counts["findings"] += len(ranges)
        counts["false_positive_findings"] += sum(
            not any(start < row["end"] and row["start"] < end for row in gold)
            for start, end in ranges)
        detected_bytes = byte_coverage((0, len(text.encode())), ranges)
        gold_bytes = 0
        for row in gold:
            span = row["start"], row["end"]
            covered = byte_coverage(span, ranges)
            gold_bytes += covered
            full = covered == span[1] - span[0]
            nonwhite = all(byte_coverage(part, ranges) == part[1] - part[0]
                           for part in required_ranges(text, *span))
            metric = classes.setdefault(row["class"], {"gold": 0, "full_bytes": 0,
                "nonwhitespace": 0, "partial": 0, "transformed_full_bytes": 0,
                "transformed_nonwhitespace": 0, "scope": row["scope"]})
            if metric["scope"] != row["scope"]:
                raise ValueError("inconsistent class scope")
            metric["gold"] += 1
            metric["full_bytes"] += full
            metric["nonwhitespace"] += nonwhite
            metric["partial"] += 0 < covered < span[1] - span[0]
            metric["transformed_full_bytes"] += full and proven
            metric["transformed_nonwhitespace"] += nonwhite and proven
            occurrences.append({"field": field, "start": span[0], "end": span[1],
                                "class": row["class"], "scope": row["scope"],
                                "covered_bytes": covered, "full_bytes": full,
                                "nonwhitespace": nonwhite, "output_reconciled": proven,
                                "transformed_covered_bytes": covered if proven else 0})
        counts["detected_bytes"] += detected_bytes
        counts["gold_bytes"] += gold_bytes
        counts["false_positive_bytes"] += detected_bytes - gold_bytes
        for row in case["benign"]:
            if row["path"] != path:
                continue
            overlap = byte_coverage((row["start"], row["end"]), ranges) > 0
            benign["occurrences"] += 1
            benign["flagged"] += overlap
            # Actual-output positional proof: literal counts would falsely
            # penalize a safe 4096 when an identical planted PIN is removed,
            # or let another occurrence conceal loss. Fail closed on mismatch.
            preserved = proven and not overlap
            benign["preserved"] += preserved
            benign_occurrences.append({"field": field, "start": row["start"], "end": row["end"],
                                       "flagged": overlap, "preserved": preserved})
    errors = structure_errors(case["trace"], output)
    structural = structural_paths(case["trace"])
    errors += sum(after.get(path) != value for path, value in leaves(case["trace"])
                  if path in structural)
    return {"case": case["id"], "split": case["split"], "classes": classes,
            "detection": counts, "benign": benign, "structure_errors": errors,
            "output_mismatches": mismatches, "gold_occurrences": occurrences,
            "finding_offsets": finding_offsets, "benign_occurrences": benign_occurrences,
            "coverage": {"scanned_strings": len(strings),
                         "structural_exclusions": len(structural),
                         "nonstrings": sum(not isinstance(value, str)
                                           for _, value in leaves(case["trace"]))}}


def gate(rows, ner=False):
    if not rows:
        return False
    for row in rows:
        if (row["structure_errors"] or row["output_mismatches"]
                or row["detection"]["false_positive_bytes"]
                or row["benign"]["flagged"]
                or row["benign"]["preserved"] != row["benign"]["occurrences"]):
            return False
        for cls, metric in row["classes"].items():
            if metric["scope"] == "regex" or (ner and metric["scope"] == "ner"):
                # Only person coverage has a whitespace-qualified gate; report
                # strict full-byte coverage independently, never round it up.
                key = "transformed_nonwhitespace" if cls == "person" else "transformed_full_bytes"
                if metric[key] != metric["gold"]:
                    return False
    return True


def cached_model(value):
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except OSError as error:
        raise ValueError("cached local directory required") from error
    models = ("model_int8.onnx", "model.onnx", "onnx/model.onnx", "onnx/model_quantized.onnx",
              "onnx/model_q4f16.onnx", "onnx/model_q4.onnx", "onnx/model_int8.onnx")
    files = [path / "config.json", path / "tokenizer.json"]
    weight = next((path / name for name in models if (path / name).is_file()), None)
    if not path.is_dir() or not all(file.is_file() for file in files) or weight is None:
        raise ValueError("cached token checkpoint files required")
    return path, {"config": sha256_file(files[0]), "tokenizer": sha256_file(files[1]),
                  "weights": sha256_file(weight)}


class Runner:
    def __init__(self, binary, directory, model=None, threshold=0.5, recall_first=False, broad=False):
        self.binary = binary
        self.directory = Path(directory)
        self.ner = model is not None
        self.broad = broad
        self.config = self.directory / "config.toml"
        config = "[ner]\nenabled = false\n[decision]\nenabled = false\n"
        if model is not None:
            config = (f"[ner]\nenabled = true\nbackend = 'tokens'\nprovider = 'cpu'\n"
                      f"token_model = {json.dumps(str(model))}\nthreshold = {threshold}\n"
                      f"recall_first = {str(recall_first).lower()}\n[decision]\nenabled = false\n")
        self.config.write_text(config)
        self.env = {key: value for key, value in os.environ.items() if not key.startswith("NYM_")}
        self.env.update({"HOME": directory, "XDG_CONFIG_HOME": directory, "XDG_DATA_HOME": directory,
                         "XDG_STATE_HOME": directory, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})

    def invoke(self, command, text, fmt="text", exclusions=()):
        args = [self.binary, command, "--config", str(self.config), "--format", fmt,
                "--ruleset", "all", "--min-confidence", "low", "--quiet", "--no-color",
                "--ner" if self.ner else "--no-ner"]
        if not self.broad:
            args += ["--exclude", EXCLUDED]
        for path in exclusions:
            args += ["--exclude-path", path]
        if command == "detect":
            args += ["--json"]
        else:
            args += ["--strategy", "placeholder", "--seed", "42", "--tag", "trace-bench",
                     "--context", "synthetic-trace-bench"]
        proc = subprocess.run(args, input=text.encode(), capture_output=True, timeout=120,
                              cwd=self.directory, env=self.env)
        if proc.returncode:
            # Never surface child output, paths or source values in reports.
            raise ValueError("benchmark child failed")
        return proc.stdout.decode()

    def evaluate(self, case):
        text = json.dumps(case["trace"], ensure_ascii=False)
        exclusions = sorted(structural_paths(case["trace"]))
        findings = json.loads(self.invoke("detect", text, "json", exclusions))
        if not isinstance(findings, list):
            raise ValueError("invalid native JSON findings")
        detections = defaultdict(list)
        for finding in findings:
            if not isinstance(finding, dict) or not isinstance(finding.get("path"), str):
                raise ValueError("missing native finding path")
            detections[finding["path"]].append(finding)
        output = json.loads(self.invoke("anon", text, "json", exclusions))
        return score_case(case, detections, output)


def totals(rows):
    detection = defaultdict(int)
    benign = defaultdict(int)
    classes = {}
    for row in rows:
        for key, value in row["detection"].items():
            detection[key] += value
        for key, value in row["benign"].items():
            benign[key] += value
        for cls, value in row["classes"].items():
            metric = classes.setdefault(cls, {"scope": value["scope"]})
            for key, count in value.items():
                if key != "scope":
                    metric[key] = metric.get(key, 0) + count
    return {"detection": dict(detection), "benign": dict(benign), "classes": classes,
            "byte_precision": (detection["gold_bytes"] / detection["detected_bytes"]
                               if detection["detected_bytes"] else None),
            "output_mismatches": sum(row["output_mismatches"] for row in rows),
            "structure_errors": sum(row["structure_errors"] for row in rows)}


def raw_case(case):
    """Serialize with independently mapped leaf offsets, including JSON escapes.

    Raw offsets describe serialized UTF-8, not the decoded string coordinates.
    This is a control, never a substitute for parsed native anon utility.
    """
    parts, origins = [], {}
    size = 0

    def emit(text):
        nonlocal size
        parts.append(text)
        size += len(text.encode())

    def walk(value, path=""):
        if isinstance(value, dict):
            emit("{")
            for index, (key, item) in enumerate(value.items()):
                if index:
                    emit(",")
                emit(json.dumps(key) + ":")
                walk(item, f"{path}.{key}" if path else key)
            emit("}")
        elif isinstance(value, list):
            emit("[")
            for index, item in enumerate(value):
                if index:
                    emit(",")
                walk(item, f"{path}[{index}]")
            emit("]")
        else:
            if isinstance(value, str):
                origins[path] = size + 1
            emit(json.dumps(value, ensure_ascii=False))
    walk(case["trace"])
    strings = dict(leaves(case["trace"]))
    labels = {"gold": [], "benign": []}
    for kind in labels:
        for row in case[kind]:
            text = strings[row["path"]].encode()
            prefix = json.dumps(text[:row["start"]].decode(), ensure_ascii=False)[1:-1]
            escaped = json.dumps(row["value"], ensure_ascii=False)[1:-1]
            start = origins[row["path"]] + len(prefix.encode())
            labels[kind].append({**row, "path": "serialized", "start": start,
                                 "end": start + len(escaped.encode()), "value": escaped})
    return {"id": case["id"], "split": case["split"], "trace": {"serialized": "".join(parts)}, **labels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=os.environ.get("NYM_BIN", str(ROOT / "target/bench/release/nym")))
    parser.add_argument("--fixtures", type=Path, default=FIXTURE)
    parser.add_argument("--split", choices=("selection", "holdout", "all"), default="holdout")
    parser.add_argument("--cached-token-model", help="existing local directory only; never a repo ID")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--recall-first", action="store_true")
    parser.add_argument("--raw-control", action="store_true")
    parser.add_argument("--broad-policy-control", action="store_true")
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("threshold must be finite and in [0,1]")
    if args.recall_first and not args.cached_token_model:
        parser.error("recall-first requires a cached token model")
    if args.gate and (args.raw_control or args.broad_policy_control):
        parser.error("diagnostic controls are not regression gates")
    try:
        fixture_bytes = args.fixtures.read_bytes()
        doc = json.loads(fixture_bytes)
        validate_document(doc)
        model, model_hashes = cached_model(args.cached_token_model) if args.cached_token_model else (None, None)
        binary = str(Path(shutil.which(args.binary) or args.binary).resolve(strict=True))
        binary_hash = sha256_file(binary)
        version, error = run_version(binary)
        if error:
            raise ValueError("version unavailable")
        cases = [case for case in doc["cases"] if args.split == "all" or case["split"] == args.split]
        rows = []
        errors = 0
        with tempfile.TemporaryDirectory(prefix="nym-trace-bench-") as directory:
            runner = Runner(binary, directory, model, args.threshold, args.recall_first, args.broad_policy_control)
            for case in cases:
                try:
                    if args.raw_control:
                        control = raw_case(case)
                        text = control["trace"]["serialized"]
                        findings = json.loads(runner.invoke("detect", text))
                        # CLI text output adds one line terminator; remove exactly
                        # that framing byte, never strip source whitespace.
                        output = runner.invoke("anon", text)
                        if output.endswith("\n"):
                            output = output[:-1]
                        rows.append(score_case(control, {"serialized": findings}, {"serialized": output}))
                    else:
                        rows.append(runner.evaluate(case))
                except (OSError, ValueError, TypeError, subprocess.TimeoutExpired):
                    errors += 1
        if sha256_file(binary) != binary_hash:
            raise ValueError("binary changed during measurement")
        passed = not errors and gate(rows, ner=model is not None)
        status = git_output("status", "--porcelain")
        report = {"benchmark": doc["version"], "synthetic_only": True, "split": args.split,
                  "mode": "tokens+regex" if model else "regex-only",
                  "input_path": "raw-diagnostic" if args.raw_control else "native-json-detect+anon",
                  "policy": "broad-diagnostic" if args.broad_policy_control else "agent-trace-v1",
                  "threshold": args.threshold if model else None, "recall_first": args.recall_first,
                  "model_hashes": model_hashes, "binary_sha256": binary_hash, "nym": version,
                  "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
                  "git": git_output("rev-parse", "HEAD"),
                  "git_dirty": bool(status) if status is not None else None,
                  "requested_cases": len(cases), "errors": errors, "gate_passed": passed,
                  "totals": totals(rows), "cases": rows}
        print(json.dumps(report, indent=2))
        return int(bool(errors) or (args.gate and not passed))
    except (OSError, ValueError, TypeError, subprocess.TimeoutExpired):
        print(json.dumps({"errors": 1, "error": "benchmark validation/execution failed"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
