# Benchmark-specific recipes; do not overwrite another agent's feature binary.
set working-directory := "../.."
set positional-arguments

target := env_var_or_default("NYM_BENCH_TARGET_DIR", "/tmp/nym-benchmark-target")

# Handwritten corpus regeneration, no detector or network.
fixtures:
    uv run --no-project scripts/bench/build_trace_fixture.py

# Mandatory offline accounting and native JSON utility tests.
test:
    uv run --no-project -m unittest discover -s scripts/bench -v
    cargo test --offline --no-default-features --target-dir '{{target}}' --test trace_benchmark

check:
    cargo check --offline --no-default-features --target-dir '{{target}}'

# Only the owned Rust test; do not format sibling agents' in-flight source.
fmt:
    rustfmt --edition 2024 tests/trace_benchmark.rs

# Read-only measurement; NYM_BIN avoids rebuilding an explicitly chosen binary.
bench *args:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "${NYM_BIN:-}" ]; then
        cargo build --offline --no-default-features --target-dir '{{target}}'
        export NYM_BIN='{{target}}/debug/nym'
    fi
    uv run --no-project scripts/bench/run_trace_bench.py "$@"

# Strict in-scope recall, precision, transformed utility and structure gate.
gate:
    just --justfile scripts/bench/trace.just bench --gate
