#!/usr/bin/env python3
"""Synthetic-only Pi launcher tests: no real session reads or browser launches."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("review-pi-terms.sh")
FAKE = r'''#!/usr/bin/env bash
set -eu
if [[ "$*" == *"--help"* ]]; then printf '%s\n' '--recursive'; exit 0; fi
printf '%s\0' "$0" "$@" >> "$FAKE_LOG"
printf '\n' >> "$FAKE_LOG"
if [[ "$*" == *" discover "* && "${FAKE_FAIL:-0}" != 0 ]]; then exit "$FAKE_FAIL"; fi
output=''
while (($#)); do if [[ "$1" == '--output' ]]; then output=$2; break; fi; shift; done
[[ -n "$output" ]]
printf '%s\n' '{"synthetic":true}' > "$output"
'''


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sessions = self.root / "sessions with spaces"
        (self.sessions / "nested").mkdir(parents=True)
        (self.sessions / "nested" / "fake.jsonl").write_text('{"text":"Synthetic ExampleCase"}\n')
        self.binary = self.root / "fake nym"
        self.binary.write_text(FAKE);self.binary.chmod(0o700)
        self.log = self.root / "argv.log"
        self.env = {**os.environ, "HOME": str(self.root), "XDG_STATE_HOME": str(self.root / "state"),
                    "XDG_DATA_HOME": str(self.root / "data"), "FAKE_LOG": str(self.log)}
        self.env.pop("NYM_BINARY", None)

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, *args, binary=True, env=None, shell="bash"):
        defaults = ["--sessions", str(self.sessions), "--no-open"]
        if binary: defaults += ["--binary", str(self.binary)]
        return subprocess.run([shell, str(SCRIPT), *defaults, *args], capture_output=True,
                              env=env or self.env, timeout=15)

    def test_private_artifacts_arguments_and_unchanged_inputs(self):
        run = self.root / "private output"
        original = (self.sessions / "nested/fake.jsonl").read_bytes()
        result = self.invoke("--output-dir", str(run), "--limit", "30", "--max-distinct", "1000",
                             "--phrase-words", "2", "--exclude-file", str(self.sessions / "held-out.jsonl"))
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(sorted(p.name for p in run.iterdir()), ["config.toml", "discovery.json", "review.html"])
        self.assertEqual((self.sessions / "nested/fake.jsonl").read_bytes(), original)
        if os.name == "posix":
            self.assertEqual(run.stat().st_mode & 0o777, 0o700)
            self.assertTrue(all(p.stat().st_mode & 0o777 == 0o600 for p in run.iterdir()))
        argv = self.log.read_bytes().split(b'\0')
        for value in [str(self.sessions), "--recursive", "jsonl,ndjson", "**.text", "**.thinking",
                      "--limit", "30", "--max-distinct", "1000", "--exclude-file"]:
            self.assertIn(value.encode(), argv)
        self.assertFalse((self.root / ".config/nym").exists())

    def test_existing_output_is_never_overwritten(self):
        run = self.root / "existing";run.mkdir();(run / "keep").write_text("preserve")
        result = self.invoke("--output-dir", str(run))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(sorted(p.name for p in run.iterdir()), ["keep"])
        self.assertEqual((run / "keep").read_text(), "preserve")

    def test_discovery_failure_preserves_status_and_does_not_start_review(self):
        run = self.root / "failed"
        result = self.invoke("--output-dir", str(run), env={**self.env, "FAKE_FAIL": "2"})
        self.assertEqual(result.returncode, 2)
        self.assertEqual(sorted(p.name for p in run.iterdir()), ["config.toml"])
        self.assertNotIn(b'review\0', self.log.read_bytes())

    def test_default_output_is_durable_and_works_inside_or_outside_mux(self):
        for mux in ({}, {"TMUX": "synthetic", "HERDR_TAB_ID": "synthetic"}):
            result = self.invoke(env={**self.env, **mux})
            self.assertEqual(result.returncode, 0, result.stderr.decode())
        runs = list((self.root / "state/nym/pi-terms").iterdir())
        self.assertEqual(len(runs), 2)
        self.assertTrue(all((p / "review.html").is_file() for p in runs))

    def test_latest_compatible_local_build_is_selected_without_installation(self):
        builds = self.root / "data/nym/builds"
        for name in ("old", "new"):
            folder = builds / name;folder.mkdir(parents=True)
            candidate = folder / "nym";candidate.write_text(FAKE);candidate.chmod(0o700)
            os.utime(candidate, (1 if name == "old" else 2,) * 2)
        result = self.invoke(binary=False)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(self.log.read_bytes().split(b'\0')[0], str(builds / "new/nym").encode())
        self.log.unlink()
        # Make newest incompatible: resolver must retain a compatible older build.
        (builds / "new/nym").write_text("#!/usr/bin/env bash\nprintf 'old help'\n");(builds / "new/nym").chmod(0o700)
        result = self.invoke(binary=False)
        self.assertEqual(result.returncode, 0, result.stderr.decode())

        self.assertEqual(self.log.read_bytes().split(b'\0')[0], str(builds / "old/nym").encode())

    def test_explicit_or_environment_incompatible_binary_refuses_fallback(self):
        bad = self.root / "old nym";bad.write_text("#!/usr/bin/env bash\nprintf 'old help'\n");bad.chmod(0o700)
        result = self.invoke(binary=False, env={**self.env, "NYM_BINARY": str(bad)})
        self.assertEqual(result.returncode, 1)
        self.assertFalse((self.root / "state").exists())

    def test_macos_system_bash_handles_empty_optional_array(self):
        if not Path("/bin/bash").exists(): self.skipTest("no system Bash")
        result = self.invoke(shell="/bin/bash")
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_help_and_invalid_arguments_do_not_create_artifacts(self):
        result = subprocess.run(["bash", str(SCRIPT), "--help"], capture_output=True, env=self.env)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(self.invoke("--unknown").returncode, 1)
        self.assertEqual(self.invoke("--limit").returncode, 1)
        self.assertFalse((self.root / "state").exists())


if __name__ == "__main__":
    unittest.main()
