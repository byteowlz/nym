#!/usr/bin/env bash
set -euo pipefail
# Wrapper for the offline synthetic benchmark. Builds the binary if needed and
# runs the Python runner with regex-only detection (no model download).
cd "$(git rev-parse --show-toplevel)"

BIN="${NYM_BIN:-target/release/nym}"
if [ ! -x "$BIN" ]; then
    echo "building release binary for the benchmark..."
    cargo build --release --features ner >/dev/null 2>&1 || cargo build --release >/dev/null 2>&1
fi

export NYM_BIN="$BIN"
# Record pinned versions so the run is reproducible.
"$NYM_BIN" --version > /tmp/nym_bench_version.txt 2>&1 || true
git rev-parse --short HEAD > /tmp/nym_bench_git.txt 2>&1 || true

python3 scripts/bench/run_bench.py "$@"