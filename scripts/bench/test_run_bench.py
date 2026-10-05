"""Dedicated synthetic benchmark regressions. Run with uv run --no-project -m unittest discover -s scripts/bench."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("run_bench", Path(__file__).with_name("run_bench.py"))
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def fixture(text="keep secret", spans=None, benign=None):
    return {"text": text, "spans": spans if spans is not None else [{"value": "secret", "class": "api_key"}],
            "benign": benign if benign is not None else ["keep"]}


def finding(text, value, category="authentication", occurrence=0):
    raw = text.encode("utf-8")
    value_bytes = value.encode("utf-8")
    start = -1
    for _ in range(occurrence + 1):
        start = raw.index(value_bytes, start + 1)
    return {"start": start, "end": start + len(value_bytes), "matched_text": value, "category": category}


def invoke_main(doc, args=()):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fixtures.json"
        path.write_text(json.dumps(doc))
        with patch("sys.argv", ["run_bench", "--fixtures", str(path), "--json", *args]), patch("builtins.print") as output:
            code = bench.main()
        summary = json.loads(output.call_args_list[0].args[0])
        rendered = " ".join(str(call.args) for call in output.call_args_list)
        return code, summary, rendered


class ScoringTests(unittest.TestCase):
    def test_absent_and_empty_annotations_fail(self):
        for key, annotation in (("spans", {"value": "", "class": "email"}),
                                ("spans", {"value": "absent", "class": "email"}),
                                ("benign", ""), ("benign", "absent")):
            with self.subTest(key=key, annotation=annotation), self.assertRaises(ValueError):
                bench.validate_fixture(fixture(**{key: [annotation]}))

    def test_benign_counted_once_despite_overlapping_findings_and_duplicates(self):
        fx = fixture(benign=["keep", "keep"])
        score = bench.score_fixture(fx, [finding(fx["text"], "keep"), finding(fx["text"], "kee")], "gone secret")
        self.assertEqual(score["benign"], {"total": 1, "flagged": 1, "preserved": 0})
        self.assertEqual(sum(row["fp"] for row in score["classes"].values()), 2)

    def test_multilingual_byte_offsets_and_actual_preservation(self):
        fx = fixture("電話: Jürgen secret 保持", benign=["電話", "保持"])
        score = bench.score_fixture(fx, [finding(fx["text"], "secret")], "電話: Jürgen <KEY> 保持")
        self.assertEqual(score["classes"]["api_key"], {"gold": 1, "full": 1, "partial": 0, "fp": 0})
        self.assertEqual(score["benign"], {"total": 2, "flagged": 0, "preserved": 2})
        bad = finding(fx["text"], "secret")
        bad["start"] = fx["text"].index("secret")
        with self.assertRaises(ValueError):
            bench.score_fixture(fx, [bad], fx["text"])

    def test_full_vs_partial_union_and_duplicate_findings(self):
        fx = fixture()
        partial = bench.score_fixture(fx, [finding(fx["text"], "secr")], "keep <KEY>et")
        self.assertEqual(partial["classes"]["api_key"], {"gold": 1, "full": 0, "partial": 1, "fp": 0})
        findings = [finding(fx["text"], value) for value in ("secr", "secr", "ret")]
        full = bench.score_fixture(fx, findings, "keep <KEY>")
        self.assertEqual(full["classes"]["api_key"], {"gold": 1, "full": 1, "partial": 0, "fp": 0})

    def test_same_literal_elsewhere_does_not_give_recall(self):
        fx = fixture("secret secret", benign=[])
        score = bench.score_fixture(fx, [finding(fx["text"], "secret")], "<KEY> secret")
        self.assertEqual(score["classes"]["api_key"], {"gold": 2, "full": 1, "partial": 0, "fp": 0})

    def test_actual_anon_not_detection_controls_preservation(self):
        fx = fixture("keep keep secret")
        score = bench.score_fixture(fx, [], "keep <KEY>")
        self.assertEqual(score["benign"], {"total": 1, "flagged": 0, "preserved": 0})
        score = bench.score_fixture(fx, [finding(fx["text"], "keep")], fx["text"])
        self.assertEqual(score["benign"], {"total": 1, "flagged": 1, "preserved": 1})

    def test_overlapping_literal_occurrences_must_all_survive(self):
        fx = fixture("aaaa secret", benign=["aaa"])
        score = bench.score_fixture(fx, [], "aaa <KEY>")
        self.assertEqual(score["benign"], {"total": 1, "flagged": 0, "preserved": 0})

    def test_invalid_offsets_fail(self):
        fx = fixture("電話 secret", benign=[])
        for start, end in ((-1, 2), (1, 3), (0, 100), (0, 0), (True, 2)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                bench.score_fixture(fx, [{"start": start, "end": end, "matched_text": "電話", "category": "social"}], fx["text"])

    def test_stock_fixtures_valid(self):
        doc = json.loads(Path(bench.FIXTURES_DEFAULT).read_text())
        for fx in doc["fixtures"]:
            bench.validate_fixture(fx)


class RunnerTests(unittest.TestCase):
    def run_main(self, doc, findings=None, anonymized="keep <KEY>", args=()):
        with patch.object(bench, "run_detect", return_value=(findings or [], None)), \
             patch.object(bench, "run_anon", return_value=(anonymized, None)), \
             patch.object(bench, "run_version", return_value=("nym 0.3.0", None)), \
             patch.object(bench, "provenance", return_value={"binary_sha256": "0" * 64}):
            return invoke_main(doc, args)[:2]

    def test_partial_does_not_pass_gate(self):
        fx = fixture()
        code, summary = self.run_main({"version": "1", "fixtures": [fx]}, [finding(fx["text"], "secr")], args=("--fail-on-recall", "0.9"))
        self.assertEqual(code, 1)
        self.assertEqual(summary["per_class"][0]["partial_recall"], 1.0)
        self.assertEqual(summary["per_class"][0]["recall"], 0.0)

    def test_invalid_annotations_fail_before_invoking_binary(self):
        with patch.object(bench, "run_detect") as detect, patch.object(bench, "run_version") as version:
            code, summary, _ = invoke_main({"version": "1", "fixtures": [fixture(benign=["absent"])]})
            self.assertEqual(code, 1)
            self.assertEqual(summary["errors"], 1)
            detect.assert_not_called()
            version.assert_not_called()

    def test_mixed_scope_does_not_hide_in_scope_miss(self):
        out = fixture()
        out["out_of_regex_scope"] = ["api_key"]
        code, summary = self.run_main({"version": "1", "fixtures": [fixture(), out]}, args=("--fail-on-recall", "0.9"))
        self.assertEqual(code, 1)
        self.assertEqual(summary["per_class"][0]["in_scope_gold"], 1)

    def test_benign_gate_checks_actual_changes_too(self):
        code, summary = self.run_main({"version": "1", "fixtures": [fixture()]}, anonymized="lost <KEY>", args=("--fail-on-benign",))
        self.assertEqual(code, 1)
        self.assertEqual(summary["benign_preservation"], 0.0)
        self.assertEqual(summary["benign_literals_flagged"], 0)

    def test_reports_do_not_leak_child_errors(self):
        with patch.object(bench, "run_detect", return_value=(None, "sensitive /private/path secret")), \
             patch.object(bench, "run_anon", return_value=(None, "sensitive mapping")):
            code, summary, rendered = invoke_main({"version": "1", "fixtures": [fixture()]})
            self.assertEqual(code, 1)
            self.assertGreater(summary["errors"], 0)
            self.assertNotIn("sensitive", rendered)
            self.assertNotIn("/private", rendered)

    def test_fresh_provenance_in_consecutive_checkouts(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "fake-nym"
            binary.write_text("synthetic executable")
            fixture_path = root / "fixture.json"
            fixture_path.write_text('{"version":"1","fixtures":[]}')
            revisions = []
            try:
                for index in range(2):
                    repo = root / str(index)
                    repo.mkdir()
                    subprocess.run(["git", "init", "-q", str(repo)], check=True)
                    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-q", "--allow-empty", "-m", str(index)], check=True)
                    expected = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
                    os.chdir(repo)
                    result = bench.provenance(fixture_path, {"fixtures": []}, binary)
                    revisions.append(result["git"])
                    self.assertEqual(result["git"], expected)
                    self.assertEqual(result["fixture_sha256"], hashlib.sha256(fixture_path.read_bytes()).hexdigest())
                    self.assertEqual(result["binary_sha256"], hashlib.sha256(binary.read_bytes()).hexdigest())
                    self.assertEqual(result["fixture_hashes"], [])
                    self.assertFalse(result["git_dirty"])
                    (repo / "dirty.txt").write_text("synthetic")
                    self.assertTrue(bench.provenance(fixture_path, {"fixtures": []}, binary)["git_dirty"])
                self.assertNotEqual(*revisions)
                fixture_path.write_text('{"version":"2","fixtures":[]}')
                changed = bench.provenance(fixture_path, {"fixtures": []}, binary)
                self.assertNotEqual(changed["fixture_sha256"], result["fixture_sha256"])
                snapshot = b'{"version":"1","fixtures":[]}'
                frozen = bench.provenance(fixture_path, {"fixtures": []}, binary, snapshot)
                self.assertEqual(frozen["fixture_sha256"], hashlib.sha256(snapshot).hexdigest())
            finally:
                os.chdir(original)


class InvocationTests(unittest.TestCase):
    def test_exact_utf8_stdin_and_matching_detection_anon_options(self):
        text = "電話 keep secret\r\n"
        response = subprocess.CompletedProcess([], 0, stdout=b"keep\r\n", stderr=b"")
        with patch.object(bench.subprocess, "run", return_value=response) as run:
            self.assertEqual(bench.run_anon(text, False, None), ("keep\r\n", None))
            self.assertEqual(run.call_args.kwargs["input"], text.encode("utf-8"))
            argv = run.call_args.args[0]
            for flag in ("--no-ner", "--ruleset", "--min-confidence", "--strategy", "--format"):
                self.assertIn(flag, argv)

    def test_bad_json_and_os_errors_are_sanitized(self):
        with patch.object(bench, "run_nym", return_value=("secret /private/path", None)):
            self.assertEqual(bench.run_detect("text", False, None), (None, "invalid detection JSON"))
        with patch.object(bench.subprocess, "run", side_effect=OSError("secret /private/path")):
            self.assertEqual(bench.run_anon("text", False, None), (None, "nym invocation failed"))

    def test_actual_multilingual_anonymization(self):
        if not Path(bench.BIN).is_file():
            self.skipTest("build isolated benchmark binary with just bench first")
        text = "電話 keep user123 team@example.org\nline1 api_path GRAFANA_API_KEY"
        fx = fixture(text, spans=[{"value": "user123", "class": "username"},
                                  {"value": "team@example.org", "class": "email"}],
                     benign=["電話", "keep", "line1", "api_path", "GRAFANA_API_KEY"])
        findings, error = bench.run_detect(text, False, None)
        self.assertIsNone(error)
        anonymized, error = bench.run_anon(text, False, None)
        self.assertIsNone(error)
        score = bench.score_fixture(fx, findings, anonymized)
        self.assertEqual(score["benign"], {"total": 5, "flagged": 0, "preserved": 5})
        self.assertEqual(score["classes"], {
            "username": {"gold": 1, "full": 1, "partial": 0, "fp": 0},
            "email": {"gold": 1, "full": 1, "partial": 0, "fp": 0}})
        for span in fx["spans"]:
            self.assertNotIn(span["value"], anonymized)


class EndToEndProvenanceTests(unittest.TestCase):
    def test_consecutive_runner_processes_capture_checkout_and_fixture_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "fake-nym"
            fx = fixture()
            detection = json.dumps([finding(fx["text"], "secret")])
            binary.write_text("#!/usr/bin/env bash\ncase \"$1\" in\n"
                              "--version) printf 'nym 0.3.0\\n';;\n"
                              f"detect) printf '%s\\n' '{detection}';;\n"
                              "anon) printf 'keep <KEY>\\n';;\nesac\n")
            binary.chmod(0o700)
            path = root / "fixture.json"
            doc = {"version": "1", "fixtures": [fx]}
            path.write_text(json.dumps(doc))
            reports = []
            for index in range(2):
                repo = root / str(index)
                repo.mkdir()
                subprocess.run(["git", "init", "-q", str(repo)], check=True)
                subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-q", "--allow-empty", "-m", str(index)], check=True)
                process = subprocess.run(["uv", "run", "--no-project", str(Path(bench.__file__).resolve()),
                                          "--fixtures", str(path), "--json", "--fail-on-recall", "1", "--fail-on-benign"],
                                         cwd=repo, env={**os.environ, "NYM_BIN": str(binary)},
                                         capture_output=True, text=True, timeout=30)
                self.assertEqual(process.returncode, 0, process.stderr)
                report = json.loads(process.stdout)
                expected = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
                self.assertEqual(report["git"], expected)
                self.assertEqual(report["fixture_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                expected_hash = hashlib.sha256(json.dumps(fx, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
                self.assertEqual(report["fixture_hashes"], [expected_hash])
                self.assertEqual(report["binary_sha256"], hashlib.sha256(binary.read_bytes()).hexdigest())
                self.assertNotIn(str(root), process.stdout)
                self.assertNotIn("matched_text", process.stdout)
                self.assertNotIn("secret", process.stdout)
                reports.append(report)
            self.assertNotEqual(reports[0]["git"], reports[1]["git"])


if __name__ == "__main__":
    unittest.main()
