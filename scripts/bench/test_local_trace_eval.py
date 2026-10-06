"""Synthetic-only offline tests: no access to any actual session or model."""
import copy
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

import local_trace_eval as local


def transformed(text, findings):
    raw, chunks, cursor = text.encode(), [], 0
    for finding in sorted(findings, key=lambda row: row["start"]):
        a, b = finding["start"], finding["end"]
        chunks.extend((raw[cursor:a], b"<PII>"))
        cursor = b
    chunks.append(raw[cursor:])
    return b"".join(chunks).decode()


def finding(text, a, b, category="identity"):
    return {"start": a, "end": b, "matched_text": text.encode()[a:b].decode(), "category": category}


def results_for(bundle, include=None):
    results = []
    for unit in bundle["units"]:
        findings = [finding(unit["text"], gold["start"], gold["end"]) for i, gold in enumerate(unit["canaries"])
                    if include is None or i in include]
        results.append({"id": unit["id"], "findings": findings, "output": transformed(unit["text"], findings)})
    return results


class LocalTraceTests(unittest.TestCase):
    def fixture(self):
        records = [{"message": {"text": "λ☃ original\nPIN=7301 is a parser token", "role": "assistant"},
                    "odd.key": ["不変", "text"]}, {"log": "build"}]
        selectors = [{"line": 0, "path": ["message", "text"], "family": "code"},
                     {"line": 0, "path": ["odd.key", 0], "family": "tool"},
                     {"line": 1, "path": ["log"], "family": "log"}]
        plan = [{"line": 0, "path": ["message", "text"], "at": 5, "label": "PIN", "prefix": " bank PIN=", "value": "7301", "suffix": "; "},
                {"line": 0, "path": ["message", "text"], "at": 5, "label": "PIN", "prefix": " other PIN=", "value": "7301", "suffix": "; "},
                {"line": 0, "path": ["odd.key", 0], "at": 6, "label": "GIVEN_NAME", "prefix": " contact=", "value": "Zoë李", "suffix": " "}]
        return records, selectors, plan

    def test_injection_original_utf8_positions_preserves_structure(self):
        records, selectors, plan = self.fixture()
        original = copy.deepcopy(records)
        injected, bundle = local.prepare(records, selectors, plan)
        self.assertEqual(records, original)
        self.assertEqual(injected[0]["message"]["role"], "assistant")
        self.assertEqual(injected[0]["odd.key"][1], "text")
        self.assertEqual(bundle["units"][0]["text"], "λ☃ bank PIN=7301;  other PIN=7301;  original\nPIN=7301 is a parser token")
        self.assertEqual([gold["start"] for gold in bundle["units"][0]["canaries"]], [15, 32])
        self.assertEqual(bundle["units"][1]["text"], "不変 contact=Zoë李 ")
        for unit in bundle["units"]:
            for gold in unit["canaries"]:
                self.assertEqual(unit["text"].encode()[gold["start"]:gold["end"]].decode(), gold["value"])

    def test_duplicate_canaries_count_by_occurrence_not_value(self):
        _, bundle = local.prepare(*self.fixture())
        full = local.score(bundle, results_for(bundle))
        self.assertEqual(full["canary_classes"]["PIN"], {"occurrences": 2, "full_removed": 2, "partial": 0, "missed": 0, "recall": 1.0})
        partial = local.score(bundle, results_for(bundle, include={0}))
        self.assertEqual(partial["canary_classes"]["PIN"], {"occurrences": 2, "full_removed": 1, "partial": 0, "missed": 1, "recall": 0.5})
        # A third, unannotated occurrence of the same value remains deliberately.
        self.assertIn("PIN=7301 is a parser token", results_for(bundle)[0]["output"])

    def test_partial_unicode_coverage_not_full_recall(self):
        _, bundle = local.prepare(*self.fixture())
        results = results_for(bundle)
        gold = bundle["units"][1]["canaries"][0]
        f = finding(bundle["units"][1]["text"], gold["start"], gold["end"] - 3)
        results[1] = {"id": 1, "findings": [f], "output": transformed(bundle["units"][1]["text"], [f])}
        self.assertEqual(local.score(bundle, results)["canary_classes"]["GIVEN_NAME"]["partial"], 1)

    def test_bad_utf8_and_serialized_json_positions_rejected(self):
        for offset in (1, 3, 999, -1, True):
            records, selectors, plan = self.fixture()
            plan[0]["at"] = offset
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                local.prepare(records, selectors, plan)

    def test_path_type_mismatch_non_string_and_duplicates_fail(self):
        records, selectors, plan = self.fixture()
        for bad in ([{"line": 0, "path": ["odd.key", "0"]}], [{"line": 0, "path": ["message"]}], selectors + selectors[:1], [{"line": True, "path": []}]):
            with self.assertRaises(ValueError):
                local.prepare(records, bad, [])
        plan[0]["path"] = ["message", "role"]
        with self.assertRaises(ValueError):
            local.prepare(records, selectors, plan)

    def test_empty_plan_for_uncontaminated_precision(self):
        records, selectors, _ = self.fixture()
        injected, bundle = local.prepare(records, selectors, [])
        self.assertEqual(injected, records)
        self.assertTrue(all(unit["canaries"] == [] for unit in bundle["units"]))
        self.assertEqual(local.score(bundle, results_for(bundle))["canary_classes"], {})

    def test_blank_jsonl_line_rejected_not_silently_reindexed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.jsonl"
            path.write_text('{"text":"first"}\n\n{"text":"third"}\n')
            with self.assertRaises(ValueError):
                local.read_jsonl(path)

    def test_arbitrary_scalar_jsonl_root(self):
        records, bundle = local.prepare(["λ"], [{"line": 0, "path": []}],
                                        [{"line": 0, "path": [], "at": 2, "value": "Zoë", "label": "GIVEN_NAME"}])
        self.assertEqual(records, ["λZoë"])
        self.assertEqual(bundle["units"][0]["canaries"], [{"start": 2, "end": 6, "label": "GIVEN_NAME", "value": "Zoë"}])

    def test_malformed_findings_and_output_fail_closed(self):
        _, bundle = local.prepare(*self.fixture())
        for mutation in ("missing", "duplicate", "unknown", "offset", "matched", "category", "output", "placeholder", "overlap", "utf8", "null"):
            results = results_for(bundle)
            if mutation == "missing":
                results.pop()
            elif mutation == "duplicate":
                results.append(results[0])
            elif mutation == "unknown":
                results[0]["id"] = 99
            elif mutation == "output":
                results[0]["output"] += " changed"
            elif mutation == "placeholder":
                results[0]["output"] = results[0]["output"].replace("<PII>", "<7301>")
            elif mutation == "overlap":
                results[0]["findings"].append(finding(bundle["units"][0]["text"], 14, 18))
            elif mutation == "utf8":
                results[0]["findings"][0].update(start=1, end=2)
            elif mutation == "null":
                results[0]["findings"] = None
            else:
                key, value = {"offset": ("end", 999), "matched": ("matched_text", "wrong"), "category": ("category", "private-name")}[mutation]
                results[0]["findings"][0][key] = value
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                local.score(bundle, results)

    def test_output_cannot_move_redaction_to_duplicate_value(self):
        _, bundle = local.prepare(*self.fixture())
        results = results_for(bundle, include={0})
        # Both replace the same substring once, but at DIFFERENT occurrences.
        text = bundle["units"][0]["text"]
        gold = bundle["units"][0]["canaries"][1]
        results[0]["output"] = transformed(text, [finding(text, gold["start"], gold["end"])])
        with self.assertRaises(ValueError):
            local.score(bundle, results)

    def test_duplicate_findings_deduplicate_but_category_conflicts_fail(self):
        _, bundle = local.prepare(*self.fixture())
        results = results_for(bundle)
        results[0]["findings"].append(dict(results[0]["findings"][0]))
        self.assertEqual(local.score(bundle, results)["canary_classes"]["PIN"]["full_removed"], 2)
        results[0]["findings"][-1]["category"] = "contact"
        with self.assertRaises(ValueError):
            local.score(bundle, results)

    def test_tampered_bundle_fails(self):
        _, bundle = local.prepare(*self.fixture())
        bundle["units"][0]["canaries"][0]["value"] = "incorrect"
        with self.assertRaises(ValueError):
            local.score(bundle, results_for(bundle))

    def test_precision_sample_stratified_and_canary_excluded(self):
        _, bundle = local.prepare(*self.fixture())
        results = results_for(bundle)
        # Actual flagged non-canary occurrences, including a duplicate literal.
        text = bundle["units"][0]["text"]
        a = text.encode().rfind(b"7301")
        results[0]["findings"].append(finding(text, a, a + 4, "financial"))
        results[0]["output"] = transformed(text, results[0]["findings"])
        results[2] = {"id": 2, "findings": [finding("build", 0, 5, "other")], "output": "<PII>"}
        sampled = local.sample(bundle, results, per_stratum=1)
        self.assertEqual(sampled, local.sample(bundle, results, per_stratum=1))
        self.assertEqual(sampled["population"], {"financial:code:1-4": 1, "other:log:5-16": 1})
        self.assertEqual(sampled["excluded_canary_findings"], 3)
        self.assertEqual(len(sampled["rows"]), 2)
        self.assertTrue(all(row["decision"] is None for row in sampled["rows"]))
        reviews = [{"sample_id": row["sample_id"], "decision": "benign"} for row in sampled["rows"]]
        report = local.summarize(sampled, reviews)
        self.assertTrue(report["is_population_census"])
        self.assertEqual(report["categories"]["financial"]["precision_upper"], 0.0)
        for secret in ("7301", "build", "odd.key", "message"):
            self.assertNotIn(secret, json.dumps(report))

    def test_weighted_precision_not_naive_sample_ratio(self):
        sampled = {"version": local.VERSION, "population": {"identity:code:1-4": 90, "identity:log:5-16": 10},
                   "rows": [{"sample_id": 0, "stratum": "identity:code:1-4"}, {"sample_id": 1, "stratum": "identity:log:5-16"}]}
        reviews = [{"sample_id": 0, "decision": "sensitive"}, {"sample_id": 1, "decision": "uncertain"}]
        report = local.summarize(sampled, reviews)
        self.assertEqual(report["categories"]["identity"], {"flagged_population": 100, "reviewed": 2, "uncertain": 1, "precision_lower": 0.9, "precision_upper": 1.0})
        self.assertFalse(report["is_population_census"])
        for bad in (reviews[:1], reviews + reviews[:1], [{"sample_id": 0, "decision": "private-data"}, reviews[1]]):
            with self.assertRaises(ValueError):
                local.summarize(sampled, bad)

    def test_git_artifacts_refused_and_local_permissions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            git = root / "repo"
            git.mkdir()
            (git / ".git").write_text("gitdir: /elsewhere")
            with self.assertRaises(ValueError):
                local.write_private(git / "ignored" / "raw.json", {"private": "data"})
            out = local.local_directory(root / "private")
            local.write_private(out / "raw.json", {"synthetic": "data"})
            self.assertEqual(stat.S_IMODE(out.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE((out / "raw.json").stat().st_mode), 0o600)
            with self.assertRaises(FileExistsError):
                local.write_private(out / "raw.json", {})
            with self.assertRaises(FileExistsError):
                local.local_directory(out)

    def test_cli_end_to_end_and_value_free_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records, selectors, plan = self.fixture()
            for filename, data in (("input", records), ("selectors", selectors), ("plan", plan)):
                local.write_private(root / f"{filename}.jsonl", data, jsonl=True)
            args = [sys.executable, local.__file__]
            proc = subprocess.run(args + ["prepare", "--input", str(root / "input.jsonl"), "--selectors", str(root / "selectors.jsonl"), "--plan", str(root / "plan.jsonl"), "--output-dir", str(root / "prepared")], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            bundle = local.read_json(root / "prepared" / "bundle.json")
            local.write_private(root / "results.jsonl", results_for(bundle), jsonl=True)
            score_args = ["score", "--bundle", str(root / "prepared" / "bundle.json"), "--results", str(root / "results.jsonl"), "--require-full-recall"]
            proc = subprocess.run(args + score_args, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(proc.stdout)
            self.assertEqual(report["canary_classes"]["PIN"]["recall"], 1.0)
            # Make one natural flagged span available to the review CLI.
            results = results_for(bundle)
            results[2] = {"id": 2, "findings": [finding("build", 0, 5, "other")], "output": "<PII>"}
            (root / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in results))
            proc = subprocess.run(args + ["sample", "--bundle", str(root / "prepared" / "bundle.json"), "--results", str(root / "results.jsonl"), "--output-dir", str(root / "review")], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(json.loads(proc.stdout)["sampled"], 1)
            review_rows = local.read_jsonl(root / "review" / "review.jsonl")
            for row in review_rows:
                row["decision"] = "benign"
            local.write_private(root / "reviewed.jsonl", review_rows, jsonl=True)
            proc = subprocess.run(args + ["summarize", "--sample", str(root / "review" / "sample.json"), "--reviews", str(root / "reviewed.jsonl")], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(json.loads(proc.stdout)["categories"]["other"]["precision_upper"], 0.0)
            self.assertNotIn("build", proc.stdout)
            local.write_private(root / "missing.jsonl", results_for(bundle, include=set()), jsonl=True)
            missing_args = list(score_args)
            missing_args[4] = str(root / "missing.jsonl")
            proc = subprocess.run(args + missing_args, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1)
            (root / "results.jsonl").write_text('{"private-secret-value": malformed}')
            proc = subprocess.run(args + score_args, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertNotIn("private-secret-value", proc.stdout + proc.stderr)
            self.assertNotIn(str(root), proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
