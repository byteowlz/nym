"""Offline precision/utility accounting regressions; no model or private traces."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
import sys
import os
import subprocess
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
SPEC = importlib.util.spec_from_file_location("trace_bench", Path(__file__).with_name("run_trace_bench.py"))
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def case(text="keep secret secret"):
    return {"id": "unit", "split": "holdout", "trace": {"records": [
        {"id": "00000000-0000-4000-8000-000000000001", "parentId": None,
         "timestamp": "2026-01-02T03:04:05Z", "type": "message", "message": {
             "role": "assistant", "content": [{"type": "text", "text": text}],
             "usage": {"input": 4096, "cached": False, "cost": 0.25}}}]},
        "gold": [{"path": "records[0].message.content[0].text", "start": 5, "end": 11,
                  "value": "secret", "class": "api_key", "scope": "regex"},
                 {"path": "records[0].message.content[0].text", "start": 12, "end": 18,
                  "value": "secret", "class": "api_key", "scope": "regex"}],
        "benign": [{"path": "records[0].message.content[0].text", "start": 0, "end": 4, "value": "keep"}]}


def detected(start, end, value):
    return {"start": start, "end": end, "matched_text": value, "category": "authentication"}


def score(fx, findings, text):
    path = fx["gold"][0]["path"]
    transformed = copy.deepcopy(fx["trace"])
    transformed["records"][0]["message"]["content"][0]["text"] = text
    return bench.score_case(fx, {path: findings}, transformed)


class TraceAccountingTests(unittest.TestCase):
    def test_repeated_occurrences_scored_positionally(self):
        row = score(case(), [detected(5, 11, "secret")], "keep <API_KEY> secret")
        self.assertEqual(row["classes"]["api_key"], {"gold": 2, "full_bytes": 1,
                         "nonwhitespace": 1, "partial": 0, "transformed_full_bytes": 1,
                         "transformed_nonwhitespace": 1, "scope": "regex"})

    def test_spill_into_benign_is_fp_even_if_finding_overlaps_gold(self):
        row = score(case(), [detected(0, 11, "keep secret")], "<PII> secret")
        self.assertEqual(row["detection"], {"findings": 1, "false_positive_findings": 0,
                                          "detected_bytes": 11, "gold_bytes": 6, "false_positive_bytes": 5})
        self.assertEqual(row["benign"], {"occurrences": 1, "flagged": 1, "preserved": 0})

    def test_whitespace_only_gap_qualified_but_leaked_letter_not_hidden(self):
        fx = case("keep Jörg Qwertz")
        fx["gold"] = [{"path": fx["gold"][0]["path"], "start": 5, "end": 17,
                       "value": "Jörg Qwertz", "class": "person", "scope": "ner"}]
        row = score(fx, [detected(5, 10, "Jörg"), detected(11, 17, "Qwertz")], "keep <PII> <PII>")
        self.assertEqual(row["classes"]["person"], {"gold": 1, "full_bytes": 0,
                         "nonwhitespace": 1, "partial": 1, "transformed_full_bytes": 0,
                         "transformed_nonwhitespace": 1, "scope": "ner"})
        leaked = score(fx, [detected(5, 10, "Jörg"), detected(11, 16, "Qwert")], "keep <PII> <PII>z")
        self.assertEqual(leaked["classes"]["person"]["transformed_nonwhitespace"], 0)

    def test_detection_is_not_proof_of_actual_transformation(self):
        row = score(case(), [detected(5, 11, "secret"), detected(12, 18, "secret")], "keep <PII> secr")
        self.assertEqual(row["output_mismatches"], 1)
        self.assertEqual(row["classes"]["api_key"]["transformed_full_bytes"], 0)

    def test_duplicate_findings_do_not_inflate_coverage(self):
        row = score(case(), [detected(5, 11, "secret")] * 2, "keep <PII> secret")
        self.assertEqual(row["detection"]["detected_bytes"], 6)
        self.assertEqual(row["classes"]["api_key"]["full_bytes"], 1)

    def test_annotations_validate_bytes_overlap_duplicates_and_structural_scope(self):
        for mutate in (
            lambda fx: fx["gold"][0].update(start=True),
            lambda fx: fx["gold"][0].update(value="missing"),
            lambda fx: fx["gold"][0].update(start=4, end=5, value=" "),
            lambda fx: fx["gold"].append(dict(fx["gold"][0])),
            lambda fx: fx["benign"].append({k: v for k, v in fx["gold"][0].items() if k in ("path", "start", "end", "value")}),
            lambda fx: fx["gold"][0].update(path="records[0].id", start=0, end=36, value=fx["trace"]["records"][0]["id"]),
        ):
            fx = case()
            mutate(fx)
            with self.subTest(fx=fx), self.assertRaises(ValueError):
                bench.validate_case(fx)

    def test_scalar_types_ids_and_references_checked_on_actual_output(self):
        fx = case()
        for field, value in (("id", "changed"), ("parentId", "changed"), ("timestamp", "changed")):
            output = copy.deepcopy(fx["trace"])
            output["records"][0][field] = value
            self.assertGreater(bench.score_case(fx, {}, output)["structure_errors"], 0)
        output = copy.deepcopy(fx["trace"])
        output["records"][0]["message"]["usage"]["input"] = "4096"
        self.assertGreater(bench.score_case(fx, {}, output)["structure_errors"], 0)
        output = copy.deepcopy(fx["trace"])
        output["records"][0]["message"]["usage"]["cached"] = 0
        self.assertGreater(bench.score_case(fx, {}, output)["structure_errors"], 0)

    def test_no_blanket_metadata_whitelist(self):
        fx = case()
        fx["trace"]["metadata"] = {"id": "identity", "email": "fiction@example.invalid"}
        scanned = bench.scan_paths(fx["trace"])
        self.assertIn("metadata.id", scanned)
        self.assertIn("metadata.email", scanned)
        self.assertNotIn("records[0].id", scanned)
        self.assertIn("records[0].type", scanned)
        fx["trace"]["records"][0]["message"]["content"][0]["id"] = "sensitive-text-metadata"
        fx["trace"]["records"][0]["message"]["toolCallId"] = "sensitive-user-metadata"
        scanned = bench.scan_paths(fx["trace"])
        self.assertIn("records[0].message.content[0].id", scanned)
        self.assertIn("records[0].message.toolCallId", scanned)

    def test_invalid_source_id_and_tool_relationships_rejected(self):
        from build_trace_fixture import document
        fx = document()["cases"][0]
        fx["trace"]["records"][1]["parentId"] = "dangling"
        with self.assertRaises(ValueError):
            bench.validate_case(fx)
        fx = document()["cases"][0]
        fx["trace"]["records"][2]["message"]["toolCallId"] = "dangling"
        with self.assertRaises(ValueError):
            bench.validate_case(fx)
        fx = document()["cases"][0]
        fx["trace"]["records"][1]["id"] = fx["trace"]["records"][0]["id"]
        with self.assertRaises(ValueError):
            bench.validate_case(fx)

    def test_benign_same_literal_other_field_cannot_mask_damage(self):
        fx = case()
        fx["trace"]["metadata"] = {"note": "keep"}
        row = score(fx, [detected(0, 11, "keep secret")], "<PII> secret")
        self.assertEqual(row["benign"]["preserved"], 0)

    def test_zero_findings_does_not_pass_recall_or_identity_transform(self):
        row = score(case(), [], "keep secret secret")
        self.assertFalse(bench.gate([row], ner=False))

    def test_scope_ner_and_policy_misses_reported_not_silently_gated_as_regex(self):
        fx = case()
        fx["gold"][1].update(scope="policy", **{"class": "codename"})
        row = score(fx, [detected(5, 11, "secret")], "keep <PII> secret")
        self.assertTrue(bench.gate([row], ner=False))
        fx["gold"][1].update(scope="ner", **{"class": "person"})
        row = score(fx, [detected(5, 11, "secret")], "keep <PII> secret")
        self.assertTrue(bench.gate([row], ner=False))
        self.assertFalse(bench.gate([row], ner=True))

    def test_stock_corpus_is_valid_and_has_disjoint_holdout_names(self):
        doc = json.loads(Path(bench.FIXTURE).read_text())
        bench.validate_document(doc)
        self.assertEqual({fx["split"] for fx in doc["cases"]}, {"selection", "holdout"})
        names = {split: {g["value"] for fx in doc["cases"] if fx["split"] == split
                         for g in fx["gold"] if g["class"] == "person"} for split in ("selection", "holdout")}
        self.assertTrue(all(names.values()))
        self.assertFalse(names["selection"] & names["holdout"])

    def test_optional_ner_rejects_repo_ids_and_missing_files_before_execution(self):
        with self.assertRaises(ValueError):
            bench.cached_model("org/repository")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                bench.cached_model(directory)
            for file in ("config.json", "tokenizer.json", "model_int8.onnx"):
                (Path(directory) / file).write_bytes(b"synthetic checkpoint stub")
            model, hashes = bench.cached_model(directory)
            self.assertEqual(model, Path(directory).resolve())
            self.assertEqual(set(hashes), {"weights", "config", "tokenizer"})
            self.assertEqual(len(set(hashes.values())), 1)

    def test_fixture_generator_is_deterministic_not_model_labeled(self):
        from build_trace_fixture import document
        self.assertEqual(json.loads(Path(bench.FIXTURE).read_text()), document())

    def test_raw_control_maps_escaped_unicode_and_repeated_occurrences(self):
        doc = json.loads(Path(bench.FIXTURE).read_text())
        for fx in doc["cases"]:
            raw = bench.raw_case(fx)
            bench.validate_case(raw)
            self.assertEqual(json.loads(raw["trace"]["serialized"]), fx["trace"])
            self.assertEqual(len(raw["gold"]), len(fx["gold"]))
            self.assertEqual(len(raw["benign"]), len(fx["benign"]))
            self.assertEqual(len({(row["start"], row["end"]) for row in raw["gold"]}), len(fx["gold"]))

    def test_all_occurrence_offsets_reported_without_values(self):
        row = score(case(), [detected(5, 11, "secret")], "keep <PII> secret")
        self.assertEqual([(gold["start"], gold["end"], gold["covered_bytes"])
                          for gold in row["gold_occurrences"]], [(5, 11, 6), (12, 18, 0)])
        self.assertEqual(row["finding_offsets"], [{"field": 3, "start": 5, "end": 11,
                                                 "category": "authentication"}])
        self.assertEqual(len(row["benign_occurrences"]), 1)
        self.assertNotIn("secret", json.dumps(row))
        self.assertNotIn("keep", json.dumps(row))

    def test_unannotated_safe_false_positive_is_not_ignored(self):
        row = score(case(), [detected(0, 4, "keep")], "<PII> secret secret")
        self.assertEqual(row["detection"]["false_positive_findings"], 1)
        self.assertEqual(row["detection"]["false_positive_bytes"], 4)
        self.assertFalse(bench.gate([row]))

    def test_overlap_benign_literal_counts_and_fp_cannot_mask_loss(self):
        fx = case("aaaa secret secret")
        fx["benign"] = [{"path": fx["gold"][0]["path"], "start": 0, "end": 3, "value": "aaa"}]
        row = score(fx, [], "aaa secret secret")
        self.assertEqual(row["benign"]["preserved"], 0)
        self.assertEqual(row["output_mismatches"], 1)

    def test_positive_same_literal_in_same_field_does_not_penalize_safe_occurrence(self):
        fx = case()
        fx["gold"] = fx["gold"][1:]
        fx["benign"] = [{"path": fx["gold"][0]["path"], "start": 5, "end": 11, "value": "secret"}]
        row = score(fx, [detected(12, 18, "secret")], "keep secret <PII>")
        self.assertEqual(row["benign"], {"occurrences": 1, "flagged": 0, "preserved": 1})

    def test_reconciliation_rejects_secret_embedded_in_fake_placeholder(self):
        self.assertFalse(bench.reconcile("PIN 4096", "PIN <PIN4096>", [(4, 8)]))
        self.assertFalse(bench.reconcile("ABC123", "<ABC123>", [(0, 6)]))

    def test_reconciliation_rejects_fuzzy_alignments_or_overlapping_findings(self):
        self.assertFalse(bench.reconcile("abcdef", "<PII>", [(0, 4), (2, 6)]))
        self.assertFalse(bench.reconcile("keep secret", "keep <PII>et", [(5, 11)]))
        self.assertTrue(bench.reconcile("keep secret\r\n", "keep <PII>\r\n", [(5, 11)]))


class TraceInvocationTests(unittest.TestCase):
    def test_cli_gate_exit_status_and_invalid_annotations_fail_before_binary(self):
        from build_trace_fixture import document
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_text(json.dumps(document()))
            argv = ["trace_bench", "--binary", sys.executable, "--fixtures", str(path), "--gate"]
            missed = score(case(), [], "keep secret secret")
            with patch("sys.argv", argv), patch.object(bench, "run_version", return_value=("nym 0.3.0", None)), \
                    patch.object(bench.Runner, "evaluate", return_value=missed), patch("builtins.print") as printed:
                self.assertEqual(bench.main(), 1)
                self.assertFalse(json.loads(printed.call_args.args[0])["gate_passed"])
            invalid = document()
            invalid["cases"][0]["gold"][0]["value"] = "absent"
            path.write_text(json.dumps(invalid))
            with patch("sys.argv", argv), patch.object(bench, "run_version") as version, patch("builtins.print"):
                self.assertEqual(bench.main(), 1)
                version.assert_not_called()

    def test_parsed_runner_uses_native_json_detection_offsets_and_actual_anon(self):
        fx = case()
        path = fx["gold"][0]["path"]
        findings = [{**detected(5, 11, "secret"), "path": path},
                    {**detected(12, 18, "secret"), "path": path}]
        output = copy.deepcopy(fx["trace"])
        output["records"][0]["message"]["content"][0]["text"] = "keep <API_KEY> <API_KEY>"
        with tempfile.TemporaryDirectory() as directory:
            runner = bench.Runner("/synthetic/nym", directory)
            def invoke(command, text, fmt="text", exclusions=()):
                self.assertEqual(fmt, "json")
                self.assertEqual(json.loads(text), fx["trace"])
                self.assertIn("records[0].id", exclusions)
                return json.dumps(findings if command == "detect" else output)
            with patch.object(runner, "invoke", side_effect=invoke) as child:
                result = runner.evaluate(fx)
                self.assertTrue(bench.gate([result]))
                self.assertEqual(child.call_count, 2)
                self.assertEqual(result["classes"]["api_key"]["transformed_full_bytes"], 2)

    def test_runner_hermetic_config_and_exact_options(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"NYM_NER_MODEL": "unapproved/repository"}):
            runner = bench.Runner("/synthetic/nym", directory)
            response = subprocess.CompletedProcess([], 0, stdout=b"{}\n", stderr=b"")
            with patch.object(bench.subprocess, "run", return_value=response) as run:
                self.assertEqual(runner.invoke("anon", "Jörg\r\n", "json", ["session.id"]), "{}\n")
                args = run.call_args.args[0]
                self.assertIn("json", args)
                self.assertIn("--no-ner", args)
                self.assertIn(bench.EXCLUDED, args)
                self.assertIn("session.id", args)
                self.assertEqual(run.call_args.kwargs["input"], "Jörg\r\n".encode())
                env = run.call_args.kwargs["env"]
                self.assertNotIn("NYM_NER_MODEL", env)
                self.assertEqual(env["HF_HUB_OFFLINE"], "1")
                self.assertEqual(env["XDG_CONFIG_HOME"], directory)
                self.assertEqual(run.call_args.kwargs["cwd"], Path(directory))

    def test_no_child_error_values_surfaced(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = bench.Runner("/synthetic/nym", directory)
            response = subprocess.CompletedProcess([], 1, stdout=b"secret", stderr=b"/private/secret")
            with patch.object(bench.subprocess, "run", return_value=response), self.assertRaisesRegex(ValueError, "benchmark child failed"):
                runner.invoke("detect", "secret")


class ArchivedMeasurementTests(unittest.TestCase):
    def test_corrected_regex_measurement_is_green_without_relabeling_gold(self):
        directory = Path(__file__).with_name("reports")
        corrected = json.loads((directory / "agent_trace_regex_corrected.json").read_text())
        historical = json.loads((directory / "agent_trace_measurements.json").read_text())["measurements"][0]
        self.assertEqual(corrected["schema"], "agent-trace-corrected-regex-v1")
        report = corrected["measurement"]
        self.assertEqual(report["fixture_sha256"], historical["fixture_sha256"])
        self.assertNotEqual(report["binary_sha256"], historical["binary_sha256"])
        self.assertEqual(report["mode"], "regex-only")
        self.assertEqual(report["split"], "all")
        self.assertEqual(report["errors"], 0)
        self.assertTrue(report["gate_passed"])
        self.assertTrue(bench.gate(report["cases"]))
        self.assertEqual(report["totals"]["detection"]["false_positive_bytes"], 0)
        self.assertEqual(report["totals"]["byte_precision"], 1.0)
        self.assertEqual(report["totals"]["benign"], {"occurrences": 104, "flagged": 0, "preserved": 104})
        self.assertEqual(sum(row["transformed_full_bytes"] for row in report["totals"]["classes"].values()
                             if row["scope"] == "regex"), 32)

    def test_archived_matrix_offsets_and_fp_bytes_independently_recounted(self):
        fixture_bytes = Path(bench.FIXTURE).read_bytes()
        fixtures = {fx["id"]: fx for fx in json.loads(fixture_bytes)["cases"]}
        archived = json.loads(Path(__file__).with_name("reports").joinpath("agent_trace_measurements.json").read_text())
        current = json.loads(Path(__file__).with_name("reports").joinpath("agent_trace_measurements_corrected.json").read_text())
        baseline = json.loads(Path(__file__).with_name("reports").joinpath("agent_trace_regex_corrected.json").read_text())["measurement"]
        cohorts = [archived["measurements"], current["measurements"] + [baseline]]
        for cohort in cohorts:
            self.assertEqual(len({report["binary_sha256"] for report in cohort}), 1)
        reports = [report for cohort in cohorts for report in cohort]
        self.assertEqual(len({report["binary_sha256"] for report in reports}), 2)
        for report in reports:
            self.assertEqual(report["fixture_sha256"], hashlib.sha256(fixture_bytes).hexdigest())
            self.assertEqual(report["errors"], 0)
            self.assertEqual(report["totals"], bench.totals(report["cases"]))
            for result in report["cases"]:
                fx = fixtures[result["case"]]
                if report["input_path"] == "raw-diagnostic":
                    fx = bench.raw_case(fx)
                fields = list(bench.scan_paths(fx["trace"]).items())
                seen_gold, seen_detected, findings = set(), set(), []
                for offset in result["finding_offsets"]:
                    text = fields[offset["field"]][1].encode()
                    self.assertTrue(0 <= offset["start"] < offset["end"] <= len(text))
                    text[:offset["start"]].decode()
                    text[offset["start"]:offset["end"]].decode()
                    points = {(offset["field"], byte) for byte in range(offset["start"], offset["end"])}
                    findings.append(points)
                    seen_detected.update(points)
                expected_gold = []
                for field, (path, _text) in enumerate(fields):
                    for gold in fx["gold"]:
                        if gold["path"] == path:
                            expected_gold.append((field, gold["start"], gold["end"], gold["class"]))
                            seen_gold.update((field, byte) for byte in range(gold["start"], gold["end"]))
                self.assertEqual([(g["field"], g["start"], g["end"], g["class"])
                                  for g in result["gold_occurrences"]], expected_gold)
                self.assertEqual(result["detection"], {
                    "findings": len(findings),
                    "false_positive_findings": sum(not (points & seen_gold) for points in findings),
                    "detected_bytes": len(seen_detected), "gold_bytes": len(seen_detected & seen_gold),
                    "false_positive_bytes": len(seen_detected - seen_gold)})
                for gold in result["gold_occurrences"]:
                    points = {(gold["field"], byte) for byte in range(gold["start"], gold["end"])}
                    self.assertEqual(gold["covered_bytes"], len(points & seen_detected))
                    self.assertEqual(gold["full_bytes"], points <= seen_detected)
                self.assertNotIn("matched_text", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
