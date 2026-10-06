#!/usr/bin/env python3
"""Protocol safety tests, not model quality measurements."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from system1_bench import (disk_inventory, fixtures, local_url, metrics, parse_response,
                          query, request_body, summarize, unique_object)


class TestSystem1(unittest.TestCase):
    def test_synthetic_offsets_and_contrast_pairs(self):
        cases = fixtures()
        self.assertEqual(len(cases), 20)
        self.assertEqual(sum(c["sensitive"] for c in cases), 10)
        for case in cases:
            self.assertEqual(case["text"].encode()[case["start"]:case["end"]].decode(), case["span"])
            body = request_body(case, "systemone", "synthetic-model")
            self.assertEqual(set(body["questions"]), {"cand_0"})
            self.assertNotIn("sensitive", body["state"]["untrusted_candidates"][0])
            self.assertNotIn("hard", body["state"]["untrusted_candidates"][0])
        same = [c for c in cases if c["span"] == "4096"]
        self.assertEqual({c["sensitive"] for c in same}, {True, False})

    def test_locality_is_not_a_catalog_tag(self):
        for url in ["http://example.com:1234", "http://localhost:1234", "http://127.0.0.1.evil:1234",
                    "http://127.0.0.1:1234/?token=x", "http://127.0.0.1@evil:1234", "file:///tmp/model"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                local_url(url)
        self.assertEqual(local_url("http://[::1]:1234/"), "http://[::1]:1234")

    def test_native_choice_validation(self):
        answer = {"type": "choice", "choice": "keep", "confidence": 0.5,
                  "probabilities": {"redact": 0.1, "keep": 0.8, "flag": 0.1}}
        self.assertEqual(parse_response({"answers": {"cand_0": answer}}, "systemone"),
                         {"verdict": "keep", "confidence": 0.5, "probabilities": answer["probabilities"]})
        for malformed in [{}, {"other": answer}, {"cand_0": answer, "other": answer}]:
            with self.assertRaises(ValueError):
                parse_response({"answers": malformed}, "systemone")
        for probs in [{"keep": 1}, {"redact": 0.2, "keep": 0.2, "flag": 0.2},
                      {"redact": float("nan"), "keep": 1, "flag": 0}]:
            with self.assertRaises(ValueError):
                parse_response({"answers": {"cand_0": {**answer, "probabilities": probs}}}, "systemone")

    def test_duplicate_keys_and_invalid_chat_class_rejected(self):
        with self.assertRaises(ValueError):
            json.loads('{"cand_0":1,"cand_0":2}', object_pairs_hook=unique_object)
        data = {"choices": [{"message": {"content": '{"index":0,"verdict":"keep","class":9,"confidence":1}'}}]}
        with self.assertRaises(ValueError):
            parse_response(data, "chat")

    def test_invalid_reply_is_not_model_recall(self):
        cases = fixtures()
        rows = [{"id": c["id"], "repeat": 0, "verdict": "flag", "confidence": None,
                 "probabilities": None, "error": "JSONDecodeError", "latency_ms": 1} for c in cases]
        summary = summarize(cases, rows)
        self.assertIsNone(summary["raw_model_valid_only"])
        self.assertEqual(summary["valid_first_pass"], 0)
        self.assertEqual(summary["invalid_or_transport"], 20)
        self.assertEqual(summary["fail_closed_policy"]["unsafe_keeps"], 0)
        self.assertEqual(summary["fail_closed_policy"]["benign_utility"], 0)

    def test_known_control_and_hard_policy(self):
        cases = fixtures()
        # Deliberately hostile stub recommendation: keep every secret.
        rows = [{"id": c["id"], "repeat": 0, "verdict": "keep", "confidence": 1,
                 "probabilities": {"redact": 0, "keep": 1, "flag": 0}, "error": None,
                 "latency_ms": 1} for c in cases]
        summary = summarize(cases, rows)
        self.assertEqual(summary["raw_model_valid_only"]["unsafe_keeps"], 10)
        self.assertEqual(summary["fail_closed_policy"]["unsafe_keeps"], 0)
        self.assertTrue(all(not r["deployment_authorized"] for r in summary["exploratory_selective_abstention"]))
        self.assertEqual(metrics(cases, ["redact"] * 20)["benign_utility"], 0)

    def test_timeout_missing_and_redirect_never_benign(self):
        for status, body in [(200, b'{}'), (503, b'{}'), (302, b'')]:
            class Handler(BaseHTTPRequestHandler):
                def do_POST(self):
                    self.rfile.read(int(self.headers["Content-Length"]))
                    self.send_response(status)
                    self.send_header("Location", "http://example.com/private")
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, *args):
                    return

            server = HTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.handle_request)
            thread.start()
            try:
                result = query(f"http://127.0.0.1:{server.server_port}/v1/systemone", {}, "systemone", 0.1)
                self.assertEqual(result["verdict"], "flag")
                self.assertIsNotNone(result["error"])
                self.assertNotIn("example.com", json.dumps(result))
            finally:
                thread.join(timeout=2)
                server.server_close()
        result = query("http://127.0.0.1:1/v1/systemone", {}, "systemone", 0.1)
        self.assertEqual(result["verdict"], "flag")
        self.assertIsNotNone(result["error"])

    def test_disk_inventory_deduplicates_symlinks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "weights").write_bytes(b"abcd")
            (root / "alias").symlink_to(root / "weights")
            self.assertEqual(disk_inventory([root]), {"deduplicated_file_bytes": 4, "file_count": 1})


if __name__ == "__main__":
    unittest.main()
