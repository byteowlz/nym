# Versioned local builds

Build from a quiet Git checkout using Rust/Cargo, Git, and `uv` (Python 3.11+;
stdlib only). Nothing is installed, published, or configured by default:

```sh
just build-local minimal
just test-build-local
```

Profiles are explicit:

| Profile | Cargo feature selection |
| --- | --- |
| `default` | Package defaults |
| `minimal` | `--no-default-features` |
| `ner` | `--no-default-features --features ner` (CPU NER) |
| `decision` | `--no-default-features --features decision` |
| `full` | `--all-features` |

`full` includes all platform/hardware backends, so it may not compile on a given
machine. The builder does not silently downgrade features. Verification requires
the executable to run locally; cross-compilation alone is not sufficient.

Each release uses `cargo build --locked --release --bin nym` and a unique temporary
`CARGO_TARGET_DIR` outside the checkout. It never uses a shared target directory;
the isolated target is removed afterward. Cargo may fetch dependencies/runtime
artifacts; only the offline tests are guaranteed network-free.

## Provenance and mismatch checks

Artifacts live under `$XDG_DATA_HOME/nym/builds`, falling back to
`~/.local/share/nym/builds` (Windows: `%LOCALAPPDATA%/nym/builds`). Override storage
with `--builds-dir`; storage inside the source repository is rejected.

Each immutable `nym-<package-version>-<identity-sha256>` directory contains the
executable and `build.json` (schema version 1). The identity hashes all manifest
fields except creation time, including the executable SHA-256, source fingerprint,
full Git revision/dirty state, toolchain versions, target, release/profile flags,
resolved/sorted package features, and compiled version. Identical repeats verify
and reuse the existing directory; they never overwrite it. Files are marked
read-only, but this is integrity checking, not protection against the file owner.

The source fingerprint hashes sorted Git-tracked plus unignored untracked paths,
file contents, executable bits, symlink spelling/file contents, and tracked
deletions. The manifest records only the aggregate hash and file count, never
source values, filename lists, environment values, or absolute paths. Directory
symlinks and submodules are rejected rather than incompletely fingerprinted.
Ignored files and external toolchain/dependency inputs are not source-fingerprinted;
this is provenance, not a claim of reproducible or hermetic builds.

Fingerprint, revision, and dirty state are checked before/after compilation and
before activation. Editing while building causes failure: retry after all agents
and other writers are quiet. Before/after snapshots cannot detect an edit reverted
between checks, so a quiet checkout is still required.

The executable's `--version` must exactly match the expected package version,
full Git revision, clean/dirty state, resolved features and target embedded by
`build.rs`. A stale plain package version fails. Every artifact and installed copy
is checked for both SHA-256 and compiled version. Version probes remove `NYM_*`
environment variables and use temporary home/config/data/state directories.

## Explicit activation

```sh
just install-versioned minimal
# Or choose the destination explicitly:
uv run --script scripts/build_local.py --profile decision --activate ~/.cargo/bin/nym
```

The recipe explicitly activates `~/.cargo/bin/nym` (`nym.exe` on Windows). Existing
interactive/legacy install recipes are unchanged and do not provide this provenance.

Activation verifies the artifact and a sibling temporary copy before an atomic
same-filesystem replacement, then verifies the installed hash/version. Failures
before replacement leave the old executable untouched; failed installed verification
rolls back to a saved copy (or removes a newly installed file). No symlink is required;
symlink/directory destinations are rejected. An exclusive sibling activation lock
prevents this tool's concurrent activations to the same destination. A process crash
may leave a lock/temporary file requiring manual inspection; this is not a
power-loss/crash-recovery protocol. Windows may refuse replacing an executable
currently in use; the old installation remains in place.

Verify or reactivate an existing build without compilation:

```sh
uv run --script scripts/build_local.py --verify "$BUILD_DIR"
uv run --script scripts/build_local.py --verify "$BUILD_DIR" --activate ~/.cargo/bin/nym
```

Verification checks the recorded build, not whether today's source matches it.
Compare a new build's source fingerprint/identity when establishing source-to-installed
parity. An installed version string alone cannot distinguish dirty source snapshots;
use the manifest fingerprint and executable hash together.
