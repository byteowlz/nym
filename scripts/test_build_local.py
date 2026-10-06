#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Offline provenance/activation tests: Git fixtures and fake Cargo, no builds."""

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import build_local as builder


FEATURES = {
    "default": ["streaming", "ner"],
    "streaming": ["tokio"],
    "tokio": ["dep:tokio"],
    "ner": ["tokenizers", "dep:ort"],
    "tokenizers": ["dep:tokenizers"],
    "decision": ["ureq"],
    "ureq": ["dep:ureq"],
    "gpu": ["ner", "tokenizers/fast", "ureq?/tls"],
}
TARGET = "fixture-unknown-linux"


class BuildFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="nym-offline-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.repo = self.root / "source"
        self.repo.mkdir()
        self.store = self.root / "builds"
        self.destination = self.root / "bin" / "nym"
        self.real_run = builder.run
        env = builder.clean_environment()
        env["HOME"] = str(self.root)
        env["XDG_CONFIG_HOME"] = str(self.root)
        env["GIT_CONFIG_NOSYSTEM"] = "1"
        env["GIT_CONFIG_GLOBAL"] = os.devnull
        self.real_run(["git", "init", "-q"], cwd=self.repo, env=env)
        (self.repo / "Cargo.toml").write_text("fixture manifest\n")
        (self.repo / "source.rs").write_text("fixture source\n")
        (self.repo / ".gitignore").write_text("ignored\n")
        self.real_run(["git", "add", "."], cwd=self.repo, env=env)
        self.real_run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                       "-c", "commit.gpgsign=false", "-c", f"core.hooksPath={self.root / 'no-hooks'}",
                       "commit", "-qm", "fixture"], cwd=self.repo, env=env)
        self.commands = []
        self.target_dirs = []
        self.invalid_version = False
        self.version_transform = lambda version: version
        self.change_source = False
        self.fail_cargo = False
        self.run_patch = mock.patch.object(builder, "run", side_effect=self.fake_run)
        self.run_patch.start()
        self.addCleanup(self.run_patch.stop)
        self.version_patch = mock.patch.object(builder, "read_version", side_effect=self.fixture_version)
        self.version_patch.start()
        self.addCleanup(self.version_patch.stop)

    @staticmethod
    def fixture_version(binary):
        return binary.read_text().strip()

    def fake_run(self, command, *, cwd, env, capture=True):
        if command[0] == "git":
            return self.real_run(command, cwd=cwd, env=env, capture=capture)
        self.commands.append((command, dict(env)))
        if command[:2] == ["cargo", "metadata"]:
            return json.dumps({"packages": [{"name": "nym", "version": "0.3.0",
                              "manifest_path": str(self.repo / "Cargo.toml"),
                              "features": FEATURES}]}).encode()
        if command == ["rustc", "-vV"]:
            return f"rustc 1.90.0\nhost: {TARGET}\n".encode()
        if command == ["rustc", "--version"]:
            return b"rustc 1.90.0 (fixture 2025-01-01)"
        if command == ["cargo", "--version"]:
            return b"cargo 1.90.0 (fixture 2025-01-01)"
        if command[:2] == ["cargo", "build"]:
            self.target_dirs.append(Path(env["CARGO_TARGET_DIR"]))
            if self.fail_cargo:
                raise builder.BuildError("fixture cargo failure")
            profile = next(profile for profile, flags in builder.PROFILES.items()
                           if command[command.index("--target") + 2:] == flags)
            target = command[command.index("--target") + 1]
            output = Path(env["CARGO_TARGET_DIR"]) / target / "release" / (
                "nym.exe" if "windows" in target else "nym")
            output.parent.mkdir(parents=True)
            version = builder.expected_version("0.3.0", builder.source_state(self.repo),
                                               builder.resolved_features(FEATURES, profile), target)
            output.write_text("nym invalid" if self.invalid_version else self.version_transform(version) + "\n")
            if self.change_source:
                (self.repo / "source.rs").write_text("changed during compilation\n")
            return b""
        raise AssertionError(f"unexpected command: {command}")

    def build(self, profile="minimal", activate=False, **kwargs):
        return builder.build(self.repo, self.store, profile,
                             activation=self.destination if activate else None, **kwargs)

    def test_verified_manifest_and_no_default_activation_or_private_paths(self):
        directory = self.build("default")
        manifest = builder.verify_build(directory)
        source = builder.source_state(self.repo)
        self.assertEqual(manifest, {
            "schema_version": 1, "executable": "nym",
            "sha256": builder.sha256_file(directory / "nym"),
            "source_fingerprint": {"algorithm": "sha256-git-files-v1", "sha256": source.sha256,
                                   "file_count": source.file_count},
            "git_revision": source.revision, "dirty": False,
            "package_version": "0.3.0", "rustc_version": "rustc 1.90.0 (fixture 2025-01-01)",
            "cargo_version": "cargo 1.90.0 (fixture 2025-01-01)", "target": TARGET,
            "build_profile": "release", "feature_profile": "default", "feature_flags": [],
            "features": ["default", "ner", "streaming", "tokenizers", "tokio"],
            "compiled_version": builder.expected_version("0.3.0", source,
                ["default", "ner", "streaming", "tokenizers", "tokio"], TARGET),
            "built_at_utc": manifest["built_at_utc"],
        })
        encoded = (directory / "build.json").read_text()
        self.assertNotIn(str(self.root), encoded)
        self.assertNotIn("source.rs", encoded)
        self.assertFalse(self.destination.exists())
        self.assertTrue(manifest["built_at_utc"].endswith("Z"))
        self.assertFalse((directory / "nym").stat().st_mode & 0o222)

    def test_all_profiles_locked_isolated_and_environment_scrubbed(self):
        with mock.patch.dict(os.environ, {"NYM_SECRET": "private", "NYM_BUILD_VERSION": "forged",
                                          "CARGO_FEATURE_FORGED": "1", "CARGO_TARGET_DIR": "shared"}):
            for profile, flags in builder.PROFILES.items():
                with self.subTest(profile=profile):
                    manifest = builder.verify_build(self.build(profile))
                    self.assertEqual(manifest["features"], builder.resolved_features(FEATURES, profile))
                    command, env = [entry for entry in self.commands if entry[0][1] == "build"][-1]
                    self.assertEqual(command, ["cargo", "build", "--locked", "--release", "--bin", "nym",
                                               "--target", TARGET, *flags])
                    self.assertNotEqual(env["CARGO_TARGET_DIR"], "shared")
                    self.assertFalse(any(key.startswith(("NYM_", "CARGO_FEATURE_")) for key in env))
        self.assertEqual(len(set(self.target_dirs)), 5)
        self.assertTrue(all(not path.exists() for path in self.target_dirs))

    def test_sorted_source_snapshot_tracks_content_paths_additions_deletions_not_ignored(self):
        original = builder.source_state(self.repo)
        (self.repo / "ignored").write_text("private ignored input")
        self.assertEqual(builder.source_state(self.repo), original)
        (self.repo / "untracked").write_text("new content")
        added = builder.source_state(self.repo)
        self.assertNotEqual(original.sha256, added.sha256)
        self.assertTrue(added.dirty)
        (self.repo / "untracked").rename(self.repo / "renamed")
        renamed = builder.source_state(self.repo)
        self.assertNotEqual(added.sha256, renamed.sha256)
        (self.repo / "renamed").unlink()
        self.assertEqual(builder.source_state(self.repo), original)
        (self.repo / "source.rs").write_text("edited\n")
        changed = builder.source_state(self.repo)
        self.assertNotEqual(changed.sha256, original.sha256)
        (self.repo / "source.rs").unlink()
        self.assertNotEqual(builder.source_state(self.repo).sha256, changed.sha256)

    def test_source_changes_rejected_preserves_existing_activation(self):
        self.destination.parent.mkdir()
        self.destination.write_bytes(b"old binary")
        self.change_source = True
        with self.assertRaisesRegex(builder.BuildError, "source changed"):
            self.build(activate=True)
        self.assertEqual(self.destination.read_bytes(), b"old binary")
        self.assertFalse(self.store.exists())

    def test_stale_plain_version_rejected_and_preserves_destination(self):
        self.destination.parent.mkdir()
        self.destination.write_bytes(b"old binary")
        self.invalid_version = True
        with self.assertRaisesRegex(builder.BuildError, "compiled version"):
            self.build(activate=True)
        self.assertEqual(self.destination.read_bytes(), b"old binary")
        self.assertFalse(self.store.exists())

    def test_each_compiled_provenance_dimension_must_match(self):
        source = builder.source_state(self.repo)
        replacements = [(source.revision, "0" * 40), (".clean", ".dirty"),
                        ("features=minimal", "features=ner"), (TARGET, "different-target")]
        for old, new in replacements:
            with self.subTest(dimension=old):
                self.version_transform = lambda version: version.replace(old, new)
                with self.assertRaisesRegex(builder.BuildError, "compiled version"):
                    self.build()
                self.assertFalse(self.store.exists())

    def test_cargo_failure_preserves_destination(self):
        self.destination.parent.mkdir()
        self.destination.write_bytes(b"old binary")
        self.fail_cargo = True
        with self.assertRaisesRegex(builder.BuildError, "cargo failure"):
            self.build(activate=True)
        self.assertEqual(self.destination.read_bytes(), b"old binary")

    def test_identical_repeat_reuses_immutable_build(self):
        first = self.build()
        manifest_before = (first / "build.json").read_bytes()
        second = self.build()
        self.assertEqual(second, first)
        self.assertEqual((second / "build.json").read_bytes(), manifest_before)
        self.assertEqual(list(self.store.iterdir()), [first])

    def test_dirty_snapshot_embedded_and_fingerprinted(self):
        (self.repo / "source.rs").write_text("dirty fixture\n")
        directory = self.build()
        manifest = builder.verify_build(directory)
        self.assertTrue(manifest["dirty"])
        self.assertIn(".dirty (features=minimal;target=", manifest["compiled_version"])
        self.assertEqual(manifest["source_fingerprint"]["sha256"], builder.source_state(self.repo).sha256)

    def test_corrupt_binary_rejected_before_activation_and_repeat(self):
        directory = self.build()
        executable = directory / "nym"
        executable.chmod(0o755)
        executable.write_bytes(b"corrupted")
        self.destination.parent.mkdir()
        self.destination.write_bytes(b"old binary")
        with self.assertRaisesRegex(builder.BuildError, "sha256 mismatch"):
            builder.activate(directory, self.destination)
        self.assertEqual(self.destination.read_bytes(), b"old binary")
        with self.assertRaisesRegex(builder.BuildError, "sha256 mismatch"):
            self.build()
        self.assertEqual(executable.read_bytes(), b"corrupted")

    def test_invalid_manifest_filename_rejected_without_execution(self):
        directory = self.build()
        path = directory / "build.json"
        manifest = json.loads(path.read_text())
        path.chmod(0o644)
        for filename in ("../nym", "/nym", "sub/nym", "C:\\private\\nym.exe", "nym.bat"):
            with self.subTest(filename=filename):
                manifest["executable"] = filename
                path.write_text(json.dumps(manifest))
                with mock.patch.object(builder, "read_version") as version:
                    with self.assertRaisesRegex(builder.BuildError, "filename"):
                        builder.verify_build(directory)
                    version.assert_not_called()

    def test_manifest_version_and_identity_mismatch_rejected(self):
        directory = self.build()
        path = directory / "build.json"
        original = json.loads(path.read_text())
        path.chmod(0o644)
        modified = copy.deepcopy(original)
        modified["compiled_version"] = "nym 0.3.0"
        path.write_text(json.dumps(modified))
        with self.assertRaisesRegex(builder.BuildError, "compiled version"):
            builder.verify_build(directory)
        modified = copy.deepcopy(original)
        modified["cargo_version"] = "cargo 1.91.0 (different toolchain)"
        path.write_text(json.dumps(modified))
        with self.assertRaisesRegex(builder.BuildError, "identity mismatch"):
            builder.verify_build(directory)

    def test_atomic_activation_checks_candidate_then_installed_hash_version(self):
        directory = self.build()
        self.destination.parent.mkdir()
        self.destination.write_bytes(b"old binary")
        observed = []

        def version(path):
            observed.append(path)
            if path.parent.name.startswith(".nym-activate-"):
                self.assertEqual(self.destination.read_bytes(), b"old binary")
            return self.fixture_version(path)

        with mock.patch.object(builder, "read_version", side_effect=version), \
                mock.patch.object(builder.os, "replace", wraps=os.replace) as replace:
            builder.activate(directory, self.destination)
        self.assertEqual(replace.call_count, 1)
        self.assertEqual(observed[-1], self.destination)
        self.assertEqual(builder.sha256_file(self.destination), builder.verify_build(directory)["sha256"])
        self.assertEqual(list(self.destination.parent.iterdir()), [self.destination])

    def test_candidate_failure_preserves_destination(self):
        directory = self.build()
        self.destination.parent.mkdir()
        self.destination.write_bytes(b"old binary")

        def invalid_candidate(path):
            return "nym invalid" if path.parent.name.startswith(".nym-activate-") else self.fixture_version(path)

        with mock.patch.object(builder, "read_version", side_effect=invalid_candidate):
            with self.assertRaisesRegex(builder.BuildError, "compiled version"):
                builder.activate(directory, self.destination)
        self.assertEqual(self.destination.read_bytes(), b"old binary")
        self.assertEqual(list(self.destination.parent.iterdir()), [self.destination])

    def test_post_activation_failure_rolls_back_existing_or_removes_new_file(self):
        directory = self.build()
        self.destination.parent.mkdir()
        for existing in (True, False):
            with self.subTest(existing=existing):
                if existing:
                    self.destination.write_bytes(b"old binary")
                else:
                    self.destination.unlink(missing_ok=True)

                def invalid_installed(path):
                    return "nym invalid" if path == self.destination else self.fixture_version(path)

                with mock.patch.object(builder, "read_version", side_effect=invalid_installed):
                    with self.assertRaisesRegex(builder.BuildError, "compiled version"):
                        builder.activate(directory, self.destination)
                self.assertEqual(self.destination.exists(), existing)
                if existing:
                    self.assertEqual(self.destination.read_bytes(), b"old binary")
                self.assertFalse(any(path.name.startswith(".nym-") for path in self.destination.parent.iterdir()))

    def test_failed_atomic_replace_preserves_existing(self):
        directory = self.build()
        self.destination.parent.mkdir()
        self.destination.write_bytes(b"old binary")
        with mock.patch.object(builder.os, "replace", side_effect=OSError("fixture failure")):
            with self.assertRaises(OSError):
                builder.activate(directory, self.destination)
        self.assertEqual(self.destination.read_bytes(), b"old binary")
        self.assertEqual(list(self.destination.parent.iterdir()), [self.destination])

    def test_source_change_during_publication_refuses_activation(self):
        self.destination.parent.mkdir()
        self.destination.write_bytes(b"old binary")
        real_publish = builder.publish

        def changing_publish(*args):
            directory = real_publish(*args)
            (self.repo / "source.rs").write_text("changed while publishing\n")
            return directory

        with mock.patch.object(builder, "publish", side_effect=changing_publish):
            with self.assertRaisesRegex(builder.BuildError, "source changed"):
                self.build(activate=True)
        self.assertEqual(self.destination.read_bytes(), b"old binary")

    def test_binary_change_during_version_probe_rejected(self):
        directory = self.build()
        manifest = builder.verify_build(directory)
        binary = self.root / "mutable-fixture"
        binary.write_bytes((directory / "nym").read_bytes())

        def changing_version(path):
            version = self.fixture_version(path)
            path.write_bytes(b"changed during version probe")
            return version

        with mock.patch.object(builder, "read_version", side_effect=changing_version):
            with self.assertRaisesRegex(builder.BuildError, "changed during verification"):
                builder.verify_binary(binary, manifest)

    def test_concurrent_identical_publication_reuses_winner(self):
        def race(stage, destination):
            shutil.copytree(stage, destination)
            raise FileExistsError("fixture competing publisher")

        with mock.patch.object(Path, "rename", autospec=True, side_effect=race):
            directory = self.build()
        self.assertEqual(directory.name, builder.build_identifier(builder.verify_build(directory)))
        self.assertEqual(list(self.store.iterdir()), [directory])

    def test_windows_executable_name_and_explicit_activation(self):
        directory = self.build(target="fixture-pc-windows-msvc", activate=True)
        manifest = builder.verify_build(directory)
        self.assertEqual(manifest["executable"], "nym.exe")
        self.assertEqual(builder.sha256_file(self.destination), manifest["sha256"])

    def test_activation_lock_and_immutable_destination_refused(self):
        directory = self.build()
        with self.assertRaisesRegex(builder.BuildError, "immutable"):
            builder.activate(directory, directory / "nym")
        self.destination.parent.mkdir()
        lock = self.destination.parent / ".nym.nym-activation.lock"
        lock.write_bytes(b"")
        with self.assertRaisesRegex(builder.BuildError, "activation already"):
            builder.activate(directory, self.destination)
        self.assertTrue(lock.exists())
        self.assertFalse(self.destination.exists())

    def test_store_inside_source_rejected_without_build(self):
        with self.assertRaisesRegex(builder.BuildError, "outside"):
            builder.build(self.repo, self.repo / "builds", "minimal")
        self.assertEqual(self.commands, [])


class Helpers(unittest.TestCase):
    def test_dependency_feature_resolution_and_normalization(self):
        self.assertEqual(builder.resolved_features(FEATURES, "minimal"), [])
        self.assertEqual(builder.resolved_features(FEATURES, "decision"), ["decision", "ureq"])
        self.assertEqual(builder.resolved_features(FEATURES, "ner"), ["ner", "tokenizers"])
        self.assertEqual(builder.resolved_features(FEATURES, "full"), sorted(FEATURES))
        self.assertEqual(builder.resolved_features({"default": ["with_under"], "with_under": []}, "default"),
                         ["default", "with-under"])

    def test_xdg_and_windows_store_fallbacks(self):
        with mock.patch.dict(os.environ, {"XDG_DATA_HOME": "/fixture/data"}, clear=True):
            self.assertEqual(builder.default_store(), Path("/fixture/data/nym/builds"))
        native_path = type(Path("/"))
        with mock.patch.dict(os.environ, {"LOCALAPPDATA": "/fixture/local"}, clear=True), \
                mock.patch.object(builder.os, "name", "nt"), \
                mock.patch.object(builder, "Path", native_path):
            self.assertEqual(builder.default_store(), native_path("/fixture/local/nym/builds"))

    def test_version_subprocess_has_only_version_argument_and_safe_config_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "nym"
            binary.write_bytes(b"fixture")
            with mock.patch.dict(os.environ, {"NYM_CONFIG": "private", "NYM_SECRET": "private"}), \
                    mock.patch.object(builder.subprocess, "run", return_value=subprocess.CompletedProcess(
                        [], 0, b"nym fixture\n", b"")) as process:
                self.assertEqual(builder.read_version(binary), "nym fixture")
            args, kwargs = process.call_args
            self.assertEqual(args[0], [str(binary.resolve()), "--version"])
            self.assertFalse(any(key.startswith("NYM_") for key in kwargs["env"]))
            self.assertEqual(kwargs["env"]["HOME"], kwargs["env"]["XDG_CONFIG_HOME"])
            self.assertNotEqual(kwargs["cwd"], binary.parent)
            self.assertEqual(kwargs["timeout"], 30)

    @unittest.skipIf(os.name == "nt", "POSIX executable fixture; Windows behavior tested with fake binary reader")
    def test_real_fixture_executable_version(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "nym"
            binary.write_text(f"#!{sys.executable}\nimport sys\n"
                              "assert sys.argv[1:] == ['--version']\nprint('nym fixture')\n")
            binary.chmod(0o755)
            self.assertEqual(builder.read_version(binary), "nym fixture")


if __name__ == "__main__":
    unittest.main()
