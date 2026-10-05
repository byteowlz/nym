#!/usr/bin/env bash
set -euo pipefail
# Offline regex-only benchmark. Rebuild through Cargo's freshness check instead
# of silently reusing a binary that predates the source being measured.
cd "$(git rev-parse --show-toplevel)"

if [ -z "${NYM_BIN:-}" ]; then
    cargo build --release --no-default-features --target-dir target/bench
    export NYM_BIN="$PWD/target/bench/release/nym"
fi

uv run --no-project scripts/bench/run_bench.py "$@"
