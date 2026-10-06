#!/usr/bin/env python3
"""Local-only canary recall and stratified precision review (standard library).

prepare --input TRACE.jsonl --selectors SELECT.jsonl --plan PLAN.jsonl --output-dir DIR
  Select decoded string leaves explicitly: {line:0,path:["message","text"],family:"code"}.
  line is the zero-based JSONL line index; blank lines are rejected, not skipped.
  Never select structural JSON metadata. Arbitrary key names/list indices work.
  Plan rows: {line:0,path:[...],at:UTF8_BYTE_OFFSET,label:"EMAIL",
              prefix:"contact: ",value:"fake@example.invalid",suffix:"\\n"}.
  Every insertion uses ORIGINAL decoded-text UTF-8 positions, not serialized JSON
  offsets or value searches. Same-offset insertions retain plan order. DIR gets
  injected.jsonl and bundle.json; all raw artifacts are private, local files.

score --bundle DIR/bundle.json --results RESULTS.jsonl [--require-full-recall]
sample --bundle DIR/bundle.json --results RESULTS.jsonl --output-dir REVIEW_DIR
  RESULTS has EXACTLY one {id:int,findings:[{start,end,matched_text,category}],
  output:string} per bundle unit. Feed unit.text to your local detector/placeholder
  anonymizer, preserving bytes/newlines; offsets are UTF-8 bytes. No subprocess,
  network, model download or implicit global config is used by this tool.
  Only the independent fixed nym placeholders reconcile; fake replacement and
  pseudonymization intentionally require a different evaluator. Missing/unknown
  units, malformed spans, overlapping findings or arbitrary output edits fail
  closed. Repeated values count as separate occurrences at their actual offsets.
  sample excludes canary-overlapping findings, exports local context, and leaves
  decision:null for a human to mark sensitive/benign/uncertain. For uncontaminated
  real precision, prepare a second bundle with an EMPTY plan on original text;
  all flagged spans then enter the sample population.

summarize --sample REVIEW_DIR/sample.json --reviews REVIEWED.jsonl
  Each reviewed row needs sample_id and decision; all sampled rows must be
  reviewed. Report population-weighted precision bounds by fixed category,
  not an unweighted biased sample ratio. Uncertain decisions widen the bounds.

Only value/path/hash-free fixed-vocabulary aggregates go to stdout. Files are
mode 0600, new output directories 0700, and raw inputs/outputs in ANY git working
tree are rejected (even gitignored). Do not upload exports. This tool does not
read real sessions unless the user explicitly supplies them; tests are synthetic.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

# Read-only reuse of the existing independent byte and placeholder contracts.
from run_bench import byte_coverage, finding_ranges
from run_trace_bench import reconcile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datagen"))
from agent_trace_corpus import VALUES  # noqa: E402

VERSION = "local-trace-canary-v1"
FAMILIES = {"code", "log", "config", "diff", "shell", "tool", "prose", "text"}
DECISIONS = {"sensitive", "benign", "uncertain"}


def private_path(path: Path) -> Path:
    """Refuse git worktrees, including ignored directories and linked worktrees."""
    path = path.expanduser().resolve()
    if any((parent / ".git").exists() for parent in (path, *path.parents)):
        raise ValueError("raw artifacts must stay outside git working trees")
    return path


def read_jsonl(path: Path) -> list:
    with private_path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def read_json(path: Path):
    return json.loads(private_path(path).read_text(encoding="utf-8"))


def local_directory(path: Path) -> Path:
    path = private_path(path)
    path.mkdir(parents=True, mode=0o700, exist_ok=False)
    return path


def write_private(path: Path, value, jsonl: bool = False) -> None:
    path = private_path(path)
    # O_EXCL also prevents accidentally following a final-component symlink.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        if jsonl:
            for row in value:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")


def location(row: dict) -> tuple:
    if not isinstance(row, dict) or type(row.get("line")) is not int or row["line"] < 0:
        raise ValueError("invalid line selector")
    path = row.get("path")
    if not isinstance(path, list) or any(type(key) not in (str, int) or (type(key) is int and key < 0) for key in path):
        raise ValueError("invalid decoded-text path")
    return row["line"], tuple(path)


def get_leaf(records: list, loc: tuple) -> str:
    line, path = loc
    try:
        value = records[line]
        for key in path:
            if (isinstance(value, list) and type(key) is int) or (isinstance(value, dict) and type(key) is str):
                value = value[key]
            else:
                raise ValueError("selector type mismatch")
    except (IndexError, KeyError) as error:
        raise ValueError("selector is absent") from error
    if not isinstance(value, str):
        raise ValueError("selector must address a decoded string")
    return value


def set_leaf(records: list, loc: tuple, value: str) -> None:
    line, path = loc
    if not path:
        records[line] = value
        return
    parent = records[line]
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value


def boundary(raw: bytes, at: int) -> None:
    if type(at) is not int or not 0 <= at <= len(raw):
        raise ValueError("invalid UTF-8 position")
    try:
        raw[:at].decode("utf-8")
        raw[at:].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("position splits UTF-8") from error


def prepare(records: list, selectors: list, plan: list) -> tuple[list, dict]:
    if not records or not selectors:
        raise ValueError("records and explicit text selectors are required")
    selected, grouped = {}, defaultdict(list)
    for selector in selectors:
        loc = location(selector)
        if loc in selected or selector.get("family", "text") not in FAMILIES:
            raise ValueError("duplicate selector or unknown family")
        selected[loc] = selector.get("family", "text")
        get_leaf(records, loc)
    for injection in plan:
        loc = location(injection)
        if loc not in selected:
            raise ValueError("injection is not in selected text")
        value, label = injection.get("value"), injection.get("label")
        if (not isinstance(value, str) or not value.strip() or "<" in value or ">" in value
                or label not in VALUES or any(not isinstance(injection.get(k, ""), str) for k in ("prefix", "suffix"))):
            raise ValueError("invalid canary value/class/context")
        boundary(get_leaf(records, loc).encode("utf-8"), injection.get("at"))
        grouped[loc].append(injection)
    # Roundtrip copy avoids mutating caller input; selected decoded text only.
    result = json.loads(json.dumps(records, ensure_ascii=False))
    units = []
    for loc, family in selected.items():
        raw = get_leaf(records, loc).encode("utf-8")
        chunks, gold, cursor, length = [], [], 0, 0
        for injection in sorted(grouped[loc], key=lambda row: row["at"]):
            at = injection["at"]
            chunk = raw[cursor:at]
            prefix = injection.get("prefix", "").encode("utf-8")
            value = injection["value"].encode("utf-8")
            suffix = injection.get("suffix", "").encode("utf-8")
            chunks.extend((chunk, prefix, value, suffix))
            start = length + len(chunk) + len(prefix)
            gold.append({"start": start, "end": start + len(value), "label": injection["label"], "value": injection["value"]})
            length += len(chunk) + len(prefix) + len(value) + len(suffix)
            cursor = at
        chunks.append(raw[cursor:])
        text = b"".join(chunks).decode("utf-8")
        set_leaf(result, loc, text)
        units.append({"id": len(units), "line": loc[0], "path": list(loc[1]),
                      "family": family, "text": text, "canaries": gold})
    bundle = {"version": VERSION, "units": units}
    validate_bundle(bundle)
    return result, bundle


def validate_bundle(bundle: dict) -> None:
    if not isinstance(bundle, dict) or bundle.get("version") != VERSION or not isinstance(bundle.get("units"), list) or not bundle["units"]:
        raise ValueError("invalid bundle")
    seen_locations = set()
    for i, unit in enumerate(bundle["units"]):
        if (not isinstance(unit, dict) or type(unit.get("id")) is not int or unit["id"] != i
                or unit.get("family") not in FAMILIES or not isinstance(unit.get("text"), str)
                or not isinstance(unit.get("canaries"), list)):
            raise ValueError("invalid bundle unit")
        loc = location(unit)
        if loc in seen_locations:
            raise ValueError("duplicate bundle location")
        seen_locations.add(loc)
        raw, end = unit["text"].encode("utf-8"), 0
        for canary in unit["canaries"]:
            if not isinstance(canary, dict) or canary.get("label") not in VALUES:
                raise ValueError("invalid canary class")
            a, b = canary.get("start"), canary.get("end")
            boundary(raw, a)
            boundary(raw, b)
            if not a < b or a < end or raw[a:b].decode() != canary.get("value"):
                raise ValueError("canary offset/content disagreement or overlap")
            end = b


def validate_results(bundle: dict, results: list) -> dict:
    """Strict completeness and occurrence-safe output proof before any sampling."""
    validate_bundle(bundle)
    if not isinstance(results, list):
        raise ValueError("results must be a list")
    indexed = {}
    for result in results:
        if (not isinstance(result, dict) or type(result.get("id")) is not int
                or not 0 <= result["id"] < len(bundle["units"]) or result["id"] in indexed):
            raise ValueError("unknown or duplicate result unit")
        unit = bundle["units"][result["id"]]
        findings = result.get("findings")
        ranges = finding_ranges(unit["text"], findings)
        # Exact duplicates do not inflate recall/precision; conflicting duplicate
        # category attribution is ambiguous and cannot support a class report.
        unique = {}
        for finding, span in zip(findings, ranges):
            if span in unique and unique[span]["category"] != finding["category"]:
                raise ValueError("conflicting duplicate categories")
            unique[span] = finding
        if not reconcile(unit["text"], result.get("output"), unique):
            raise ValueError("output cannot be reconciled to exact placeholder replacements")
        indexed[result["id"]] = list(unique.values())
    if set(indexed) != {unit["id"] for unit in bundle["units"]}:
        raise ValueError("missing result units")
    return indexed


def score(bundle: dict, results: list) -> dict:
    indexed = validate_results(bundle, results)
    classes = defaultdict(lambda: {"occurrences": 0, "full_removed": 0, "partial": 0, "missed": 0})
    for unit in bundle["units"]:
        ranges = [(row["start"], row["end"]) for row in indexed[unit["id"]]]
        for canary in unit["canaries"]:
            span = canary["start"], canary["end"]
            covered = byte_coverage(span, ranges)
            counts = classes[canary["label"]]
            counts["occurrences"] += 1
            counts["full_removed"] += int(covered == span[1] - span[0])
            counts["partial"] += int(0 < covered < span[1] - span[0])
            counts["missed"] += int(covered == 0)
    for counts in classes.values():
        counts["recall"] = counts["full_removed"] / counts["occurrences"]
    return {"units": len(bundle["units"]), "reconciled": True, "canary_classes": dict(sorted(classes.items())),
            "real_precision_measured": False}


def length_bin(size: int) -> str:
    for limit, name in ((4, "1-4"), (16, "5-16"), (64, "17-64")):
        if size <= limit:
            return name
    return "65+"


def sample(bundle: dict, results: list, seed: int = 47, per_stratum: int = 8, context: int = 80) -> dict:
    if per_stratum < 1 or context < 0:
        raise ValueError("invalid sampling size/context")
    indexed = validate_results(bundle, results)
    strata, excluded = defaultdict(list), 0
    for unit in bundle["units"]:
        raw = unit["text"].encode("utf-8")
        for finding in indexed[unit["id"]]:
            a, b = finding["start"], finding["end"]
            if any(a < gold["end"] and gold["start"] < b for gold in unit["canaries"]):
                excluded += 1
                continue
            stratum = ":".join((finding["category"], unit["family"], length_bin(b - a)))
            # Decode context in characters to avoid broken byte boundaries.
            left, value, right = raw[:a].decode(), raw[a:b].decode(), raw[b:].decode()
            strata[stratum].append({"unit_id": unit["id"], "start": a, "end": b,
                                    "value": value, "before": left[-context:] if context else "",
                                    "after": right[:context], "category": finding["category"], "decision": None})
    rows, population = [], {}
    for stratum, candidates in sorted(strata.items()):
        population[stratum] = len(candidates)
        candidates.sort(key=lambda row: hashlib.sha256(f'{seed}:{stratum}:{row["unit_id"]}:{row["start"]}:{row["end"]}'.encode()).digest())
        for row in candidates[:per_stratum]:
            rows.append({**row, "stratum": stratum, "sample_id": len(rows)})
    return {"version": VERSION, "population": population, "excluded_canary_findings": excluded,
            "seed": seed, "per_stratum": per_stratum, "rows": rows}


def summarize(review_sample: dict, reviews: list) -> dict:
    if not isinstance(review_sample, dict) or review_sample.get("version") != VERSION:
        raise ValueError("invalid review sample")
    population, rows = review_sample.get("population"), review_sample.get("rows")
    if not isinstance(population, dict) or not isinstance(rows, list) or not isinstance(reviews, list):
        raise ValueError("invalid review structure")
    valid_strata = {f"{category}:{family}:{size}" for category in ("contact", "identity", "financial", "network", "authentication", "social", "other")
                    for family in FAMILIES for size in ("1-4", "5-16", "17-64", "65+")}
    if any(key not in valid_strata or type(count) is not int or count < 1 for key, count in population.items()):
        raise ValueError("invalid population strata")
    strata_by_id, sizes = {}, Counter()
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or type(row.get("sample_id")) is not int or row["sample_id"] != i or row.get("stratum") not in population:
            raise ValueError("invalid sample row")
        strata_by_id[i] = row["stratum"]
        sizes[row["stratum"]] += 1
    if set(sizes) != set(population) or any(count > population[key] for key, count in sizes.items()):
        raise ValueError("sample cannot represent population")
    votes, seen = defaultdict(Counter), set()
    for review in reviews:
        if (not isinstance(review, dict) or type(review.get("sample_id")) is not int
                or review["sample_id"] not in strata_by_id or review["sample_id"] in seen
                or review.get("decision") not in DECISIONS):
            raise ValueError("invalid or duplicate review")
        uid = review["sample_id"]
        seen.add(uid)
        votes[strata_by_id[uid]][review["decision"]] += 1
    if seen != set(strata_by_id):
        raise ValueError("all sampled rows must be reviewed")
    categories = defaultdict(lambda: {"flagged_population": 0, "reviewed": 0, "uncertain": 0,
                                      "estimated_sensitive_lower": 0.0, "estimated_sensitive_upper": 0.0})
    for stratum, count in population.items():
        stats = categories[stratum.split(":")[0]]
        vote, size = votes[stratum], sizes[stratum]
        stats["flagged_population"] += count
        stats["reviewed"] += size
        stats["uncertain"] += vote["uncertain"]
        stats["estimated_sensitive_lower"] += count * vote["sensitive"] / size
        stats["estimated_sensitive_upper"] += count * (vote["sensitive"] + vote["uncertain"]) / size
    for stats in categories.values():
        stats["precision_lower"] = stats.pop("estimated_sensitive_lower") / stats["flagged_population"]
        stats["precision_upper"] = stats.pop("estimated_sensitive_upper") / stats["flagged_population"]
    return {"categories": dict(sorted(categories.items())), "reviewed": len(rows),
            "strata": len(population), "bounds_include_uncertainty_not_sampling_error": True,
            "canaries_excluded": True, "is_population_census": all(sizes[key] == count for key, count in population.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    for name in ("input", "selectors", "plan", "output-dir"):
        prepare_parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("score", "sample"):
        command = commands.add_parser(name)
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--results", type=Path, required=True)
        if name == "score":
            command.add_argument("--require-full-recall", action="store_true")
        else:
            command.add_argument("--output-dir", type=Path, required=True)
            command.add_argument("--seed", type=int, default=47)
            command.add_argument("--per-stratum", type=int, default=8)
            command.add_argument("--context", type=int, default=80)
    summary_parser = commands.add_parser("summarize")
    summary_parser.add_argument("--sample", type=Path, required=True)
    summary_parser.add_argument("--reviews", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            records, bundle = prepare(read_jsonl(args.input), read_jsonl(args.selectors), read_jsonl(args.plan))
            directory = local_directory(args.output_dir)
            write_private(directory / "injected.jsonl", records, jsonl=True)
            write_private(directory / "bundle.json", bundle)
            report = {"units": len(bundle["units"]), "canary_occurrences": sum(len(unit["canaries"]) for unit in bundle["units"])}
        elif args.command == "score":
            report = score(read_json(args.bundle), read_jsonl(args.results))
            print(json.dumps(report, sort_keys=True))
            return int(args.require_full_recall and (not report["canary_classes"] or any(counts["recall"] != 1 for counts in report["canary_classes"].values())))
        elif args.command == "sample":
            review_sample = sample(read_json(args.bundle), read_jsonl(args.results), args.seed, args.per_stratum, args.context)
            directory = local_directory(args.output_dir)
            write_private(directory / "sample.json", review_sample)
            write_private(directory / "review.jsonl", review_sample["rows"], jsonl=True)
            report = {"sampled": len(review_sample["rows"]), "flagged_population": sum(review_sample["population"].values()),
                      "strata": len(review_sample["population"]), "excluded_canary_findings": review_sample["excluded_canary_findings"]}
        else:
            report = summarize(read_json(args.sample), read_jsonl(args.reviews))
        print(json.dumps(report, sort_keys=True))
        return 0
    except (ValueError, KeyError, TypeError, OSError, UnicodeError):
        # Parser errors may include raw data/paths. NEVER echo exception text.
        print("local evaluation failed: invalid input/output or unsafe local destination", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
