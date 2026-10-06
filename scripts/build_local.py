#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Build and optionally activate a verified, immutable local nym executable."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile


SCHEMA_VERSION = 1
PROFILES = {
    "default": [],
    "minimal": ["--no-default-features"],
    "ner": ["--no-default-features", "--features", "ner"],
    "full": ["--all-features"],
    "decision": ["--no-default-features", "--features", "decision"],
}
SAFE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+_-]*\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class BuildError(Exception):
    """A fail-closed build, provenance or activation error."""


def clean_environment() -> dict[str, str]:
    """Do not let nym configuration or forged feature/version values leak in."""
    return {
        key: value for key, value in os.environ.items()
        if not key.upper().startswith(("NYM_", "CARGO_FEATURE_"))
    }


def run(command: list[str], *, cwd: Path, env: dict[str, str], capture: bool = True) -> bytes:
    result = subprocess.run(command, cwd=cwd, env=env, check=False,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.PIPE if capture else None)
    if result.returncode:
        raise BuildError(f"{Path(command[0]).name} command failed; no artifact activated")
    return result.stdout or b""


def git(repo: Path, *args: str) -> bytes:
    return run(["git", *args], cwd=repo, env=clean_environment())


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class SourceState:
    sha256: str
    file_count: int
    revision: str
    dirty: bool


def source_state(repo: Path) -> SourceState:
    """Hash sorted git-tracked plus unignored paths/content, never list values."""
    paths = sorted(set(git(repo, "ls-files", "-z", "--cached", "--others",
                           "--exclude-standard").split(b"\0")) - {b""})
    digest = hashlib.sha256(b"nym-source-v1\0")
    for raw in paths:
        path = repo / os.fsdecode(raw)
        if path.is_symlink():
            # Cover both the link spelling and the file Cargo would read.
            content = b"link\0" + os.fsencode(os.readlink(path))
            if path.is_file():
                content += b"\0" + bytes.fromhex(sha256_file(path))
            elif path.exists():
                raise BuildError("directory symlinks/submodules are unsupported source inputs")
        elif path.is_file():
            executable = bool(path.stat().st_mode & stat.S_IXUSR)
            content = b"file\0" + bytes([executable]) + bytes.fromhex(sha256_file(path))
        elif not path.exists():
            content = b"missing\0"  # Tracked deletions are part of a dirty snapshot.
        else:
            raise BuildError("non-file/submodule source inputs are unsupported")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    revision = git(repo, "rev-parse", "HEAD").decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        raise BuildError("source must have a full Git revision")
    dirty = bool(git(repo, "status", "--porcelain", "--untracked-files=normal"))
    return SourceState(digest.hexdigest(), len(paths), revision, dirty)


def assert_unchanged(repo: Path, before: SourceState) -> None:
    if source_state(repo) != before:
        raise BuildError("source changed during build; retry when all writers are quiet")


def resolved_features(feature_map: dict[str, list[str]], profile: str) -> list[str]:
    """Resolve package features exactly as exposed through CARGO_FEATURE_*.

    Metadata contains implicit optional dependency features. dep: references do
    not enable a feature name, while non-weak dependency/feature references do.
    """
    roots = {"default": ["default"], "minimal": [], "ner": ["ner"],
             "decision": ["decision"], "full": list(feature_map)}[profile]
    active: set[str] = set()
    pending = list(roots)
    while pending:
        feature = pending.pop()
        if feature in active:
            continue
        if feature not in feature_map:
            raise BuildError("requested feature is absent from Cargo metadata")
        active.add(feature)
        for dependency in feature_map[feature]:
            if dependency.startswith("dep:"):
                continue
            name, slash, _ = dependency.partition("/")
            if slash and (name.endswith("?") or name not in feature_map):
                continue
            pending.append(name)
    # This is the normalization used by build.rs for Cargo's env variable names.
    return sorted({name.lower().replace("_", "-") for name in active})


def expected_version(package_version: str, source: SourceState,
                     features: list[str], target: str) -> str:
    state = "dirty" if source.dirty else "clean"
    return (f"nym {package_version}+g{source.revision}.{state} "
            f"(features={','.join(features) or 'minimal'};target={target})")


def read_version(binary: Path) -> str:
    """Run only --version, without real user configuration or nym env values."""
    with tempfile.TemporaryDirectory(prefix="nym-version-") as temporary:
        sandbox = Path(temporary)
        env = clean_environment()
        for name in ("HOME", "USERPROFILE", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
                     "XDG_STATE_HOME", "APPDATA", "LOCALAPPDATA"):
            env[name] = str(sandbox)
        result = subprocess.run([str(binary.resolve()), "--version"], cwd=sandbox,
                                env=env, capture_output=True, timeout=30, check=False)
        if result.returncode or result.stderr:
            raise BuildError("executable --version failed")
        try:
            return result.stdout.decode("utf-8").strip()
        except UnicodeDecodeError as error:
            raise BuildError("executable --version is not UTF-8") from error


def verify_binary(binary: Path, manifest: dict) -> None:
    if binary.is_symlink() or not binary.is_file():
        raise BuildError("artifact executable must be a regular file")
    if sha256_file(binary) != manifest["sha256"]:
        raise BuildError("executable sha256 mismatch")
    if read_version(binary) != manifest["compiled_version"]:
        raise BuildError("compiled version/provenance mismatch")
    # Also detect mutation while the process was executing.
    if sha256_file(binary) != manifest["sha256"]:
        raise BuildError("executable changed during verification")


def _manifest_source(manifest: dict) -> SourceState:
    fingerprint = manifest.get("source_fingerprint", {})
    if (not isinstance(fingerprint, dict)
            or fingerprint.get("algorithm") != "sha256-git-files-v1"
            or not SHA256.fullmatch(str(fingerprint.get("sha256", "")))
            or type(fingerprint.get("file_count")) is not int
            or fingerprint["file_count"] < 0):
        raise BuildError("invalid source fingerprint in manifest")
    revision = manifest.get("git_revision", "")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        raise BuildError("invalid Git revision in manifest")
    if type(manifest.get("dirty")) is not bool:
        raise BuildError("invalid Git state in manifest")
    return SourceState(fingerprint["sha256"], fingerprint["file_count"],
                       revision, manifest["dirty"])


def _manifest_features(manifest: dict) -> list[str]:
    profile = manifest.get("feature_profile")
    if (not isinstance(profile, str) or profile not in PROFILES
            or manifest.get("feature_flags") != PROFILES[profile]
            or manifest.get("build_profile") != "release"):
        raise BuildError("invalid build profile in manifest")
    features = manifest.get("features")
    if (not isinstance(features, list)
            or any(not isinstance(item, str) or not SAFE_TOKEN.fullmatch(item) for item in features)
            or features != sorted(set(features))):
        raise BuildError("invalid feature provenance in manifest")
    return features


def _manifest_toolchain(manifest: dict) -> None:
    for key in ("rustc_version", "cargo_version"):
        value = manifest.get(key)
        if not isinstance(value, str) or not value.startswith(key.removesuffix("_version") + " ") or "\n" in value:
            raise BuildError("invalid toolchain version in manifest")
    try:
        timestamp = datetime.fromisoformat(manifest["built_at_utc"])
        if timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
            raise ValueError("not UTC")
    except (KeyError, TypeError, ValueError) as error:
        raise BuildError("invalid UTC timestamp in manifest") from error


def validate_manifest(manifest: dict) -> None:
    """Validate provenance before constructing any filename or running a binary."""
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise BuildError("unsupported build manifest schema")
    if manifest.get("executable") not in ("nym", "nym.exe"):
        raise BuildError("invalid executable filename in manifest")
    if not SHA256.fullmatch(str(manifest.get("sha256", ""))):
        raise BuildError("invalid executable hash in manifest")
    for key in ("package_version", "target"):
        if not isinstance(manifest.get(key), str) or not SAFE_TOKEN.fullmatch(manifest[key]):
            raise BuildError(f"invalid {key} in manifest")
    source = _manifest_source(manifest)
    features = _manifest_features(manifest)
    if manifest.get("compiled_version") != expected_version(
            manifest["package_version"], source, features, manifest["target"]):
        raise BuildError("manifest compiled version/provenance mismatch")
    _manifest_toolchain(manifest)


def identity(manifest: dict) -> dict:
    return {key: value for key, value in manifest.items() if key != "built_at_utc"}


def build_identifier(manifest: dict) -> str:
    validate_manifest(manifest)
    encoded = json.dumps(identity(manifest), sort_keys=True, separators=(",", ":")).encode()
    return f"nym-{manifest['package_version']}-{hashlib.sha256(encoded).hexdigest()}"


def verify_build(directory: Path) -> dict:
    if directory.is_symlink() or (directory / "build.json").is_symlink():
        raise BuildError("build directory/manifest must not be a symlink")
    try:
        manifest = json.loads((directory / "build.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BuildError("cannot read build manifest") from error
    if not isinstance(manifest, dict):
        raise BuildError("invalid build manifest")
    validate_manifest(manifest)
    if directory.name != build_identifier(manifest):
        raise BuildError("build directory identity mismatch")
    verify_binary(directory / manifest["executable"], manifest)
    return manifest


def seal(path: Path, executable: bool = False) -> None:
    path.chmod(0o555 if executable else 0o444)


def publish(binary: Path, manifest: dict, store: Path) -> Path:
    """Never overwrite a versioned build; identical repeats reuse verified data."""
    destination = store / build_identifier(manifest)
    store.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        existing = verify_build(destination)
        if identity(existing) != identity(manifest):
            raise BuildError("existing build provenance differs")
        return destination
    with tempfile.TemporaryDirectory(prefix=".nym-stage-", dir=store) as temporary:
        stage = Path(temporary) / destination.name
        stage.mkdir()
        executable = stage / manifest["executable"]
        shutil.copyfile(binary, executable)
        executable.chmod(0o755)
        (stage / "build.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8")
        verify_build(stage)
        seal(executable, executable=True)
        seal(stage / "build.json")
        try:
            stage.rename(destination)
        except OSError:
            # A competing publisher may have installed the same immutable ID.
            if not destination.exists() or identity(verify_build(destination)) != identity(manifest):
                raise BuildError("cannot publish immutable build") from None
    verify_build(destination)
    return destination


@contextmanager
def _activation_lock(destination: Path):
    lock = destination.parent / f".{destination.name}.nym-activation.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise BuildError("activation already in progress (or a stale activation lock exists)") from error
    os.close(descriptor)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _atomic_copy(binary: Path, destination: Path, manifest: dict) -> None:
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise BuildError("activation destination must be a regular file")
    with tempfile.TemporaryDirectory(prefix=".nym-activate-", dir=destination.parent) as temporary:
        candidate = Path(temporary) / binary.name
        backup = Path(temporary) / "previous"
        shutil.copyfile(binary, candidate)
        candidate.chmod(0o755)
        verify_binary(candidate, manifest)
        if destination.exists():
            shutil.copy2(destination, backup)
        # Replacement is on the destination filesystem, not a cross-device move.
        os.replace(candidate, destination)
        try:
            verify_binary(destination, manifest)
        except (BuildError, OSError, subprocess.SubprocessError):
            if backup.exists():
                os.replace(backup, destination)
            else:
                destination.unlink(missing_ok=True)
            raise


def activate(directory: Path, destination: Path) -> None:
    """Verify before atomic replacement. A failed check preserves the old file."""
    manifest = verify_build(directory)
    binary = directory / manifest["executable"]
    destination = destination.expanduser().absolute()
    if destination.is_symlink() or destination.is_dir():
        raise BuildError("activation destination must not be a symlink/directory")
    if destination == binary.absolute() or directory.resolve() in destination.resolve().parents:
        raise BuildError("activation cannot modify an immutable build")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _activation_lock(destination):
        _atomic_copy(binary, destination, manifest)


def default_store() -> Path:
    if os.environ.get("XDG_DATA_HOME"):
        return Path(os.path.expandvars(os.environ["XDG_DATA_HOME"])).expanduser() / "nym" / "builds"
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "nym" / "builds"
    return Path.home() / ".local" / "share" / "nym" / "builds"


def outside_source(path: Path, repo: Path) -> None:
    if path.resolve().is_relative_to(repo.resolve()):
        raise BuildError("build storage and temporary targets must be outside the source repository")


def _cargo_provenance(repo: Path, profile: str, target: str | None, env: dict[str, str]) -> dict:
    metadata = json.loads(run(["cargo", "metadata", "--locked", "--no-deps",
                               "--format-version", "1", *PROFILES[profile]], cwd=repo, env=env))
    packages = [package for package in metadata["packages"]
                if package["name"] == "nym"
                and Path(package["manifest_path"]).resolve() == repo / "Cargo.toml"]
    if len(packages) != 1:
        raise BuildError("expected the root nym binary package")
    package = packages[0]
    verbose_rustc = run(["rustc", "-vV"], cwd=repo, env=env).decode()
    host = next((line.removeprefix("host: ") for line in verbose_rustc.splitlines()
                 if line.startswith("host: ")), None)
    target = target or host
    if not target or not SAFE_TOKEN.fullmatch(target):
        raise BuildError("invalid or missing Rust target triple")
    if not SAFE_TOKEN.fullmatch(package["version"]):
        raise BuildError("invalid package version")
    return {
        "package_version": package["version"],
        "features": resolved_features(package["features"], profile),
        "target": target,
        "rustc_version": run(["rustc", "--version"], cwd=repo, env=env).decode().strip(),
        "cargo_version": run(["cargo", "--version"], cwd=repo, env=env).decode().strip(),
        "build_profile": "release",
        "feature_profile": profile,
        "feature_flags": PROFILES[profile],
    }


def build(repo: Path, store: Path, profile: str, target: str | None = None,
          activation: Path | None = None) -> Path:
    repo = repo.resolve()
    store = store.expanduser().resolve()
    outside_source(store, repo)
    if activation is not None:
        outside_source(activation.expanduser(), repo)
    with tempfile.TemporaryDirectory(prefix="nym-cargo-target-") as temporary:
        target_dir = Path(temporary)
        outside_source(target_dir, repo)
        env = clean_environment()
        env["CARGO_TARGET_DIR"] = str(target_dir)
        # Snapshot before even metadata can run (locked forbids lockfile writes).
        before = source_state(repo)
        provenance = _cargo_provenance(repo, profile, target, env)
        target = provenance["target"]
        assert_unchanged(repo, before)
        run(["cargo", "build", "--locked", "--release", "--bin", "nym",
             "--target", target, *PROFILES[profile]], cwd=repo, env=env, capture=False)
        assert_unchanged(repo, before)
        filename = "nym.exe" if "windows" in target else "nym"
        binary = target_dir / target / "release" / filename
        if binary.is_symlink() or not binary.is_file():
            raise BuildError("Cargo did not produce the expected regular executable")
        manifest = {
            **provenance,
            "schema_version": SCHEMA_VERSION,
            "executable": filename,
            "sha256": sha256_file(binary),
            "source_fingerprint": {"algorithm": "sha256-git-files-v1", "sha256": before.sha256,
                                   "file_count": before.file_count},
            "git_revision": before.revision,
            "dirty": before.dirty,
            "compiled_version": expected_version(provenance["package_version"], before,
                                                 provenance["features"], target),
            "built_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        validate_manifest(manifest)
        verify_binary(binary, manifest)
        assert_unchanged(repo, before)
        directory = publish(binary, manifest, store)
        # Refuse activation if a writer changed source during publication, too.
        assert_unchanged(repo, before)
        if activation is not None:
            activate(directory, activation)
        return directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="default")
    parser.add_argument("--target", help="native Rust target triple; executable must run for verification")
    parser.add_argument("--builds-dir", type=Path, default=default_store())
    parser.add_argument("--activate", type=Path, help="explicit executable destination; no activation by default")
    parser.add_argument("--verify", type=Path, metavar="BUILD_DIR", help="verify an existing build without compiling")
    args = parser.parse_args()
    try:
        if args.verify:
            manifest = verify_build(args.verify)
            if args.activate:
                activate(args.verify, args.activate)
            print(json.dumps({"verified": args.verify.name, "sha256": manifest["sha256"],
                              "compiled_version": manifest["compiled_version"],
                              "activated": bool(args.activate)}, sort_keys=True))
        else:
            repo = Path(__file__).resolve().parents[1]
            directory = build(repo, args.builds_dir, args.profile, args.target, args.activate)
            print(json.dumps({"build_id": directory.name, "activated": bool(args.activate)}, sort_keys=True))
        return 0
    except (BuildError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        # Raw OS/subprocess errors can include private absolute paths or output.
        message = str(error) if isinstance(error, BuildError) else type(error).__name__
        parser.exit(1, f"local build failed: {message}\n")


if __name__ == "__main__":
    raise SystemExit(main())
