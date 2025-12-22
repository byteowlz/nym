set positional-arguments

# Default recipe: show help
default:
    @just --list

# Display help
help:
    @just --list

# =============================================================================
# Development
# =============================================================================

# Format code
fmt:
    cargo fmt

# Run clippy lints
clippy:
    cargo clippy --all-features --tests

# Fix clippy warnings
fix *args:
    cargo clippy --fix --all-features --tests --allow-dirty "$@"

# Check code compiles (fast)
check:
    cargo check

# Check with NER feature
check-ner:
    cargo check --features ner

# =============================================================================
# Testing
# =============================================================================

# Run all tests
test:
    cargo test

# Run tests with NER feature
test-ner:
    cargo test --features ner

# Run tests with all features (NER + defaults)
test-all:
    cargo test --all-features

# Run tests with NER and bench features
test-ner-bench:
    cargo test --features "ner,bench"

# Run tests with nextest (faster, install with: cargo install cargo-nextest)
test-fast:
    cargo nextest run --no-fail-fast

# =============================================================================
# Building
# =============================================================================

# Build debug binary
build:
    cargo build

# Build release binary (CPU only)
build-release:
    cargo build --release

# Build release with NER support (CPU)
build-ner:
    cargo build --release --features ner

# Build release with NER + bench support
build-ner-bench:
    cargo build --release --features "ner,bench"

# Build release with NER + CoreML (macOS Apple Silicon)
build-ner-coreml:
    cargo build --release --features ner-coreml

# Build release with NER + CUDA (NVIDIA GPU)
build-ner-cuda:
    cargo build --release --features ner-cuda

# Build release with NER + TensorRT (NVIDIA optimized)
build-ner-tensorrt:
    cargo build --release --features ner-tensorrt

# Build release with NER + ROCm (AMD GPU)
build-ner-rocm:
    cargo build --release --features ner-rocm

# Build release with NER + DirectML (Windows GPU)
build-ner-directml:
    cargo build --release --features ner-directml

# Build release with NER + OpenVINO (Intel)
build-ner-openvino:
    cargo build --release --features ner-openvino

# Build release with all features (NER + defaults)
build-full:
    cargo build --release --all-features

# Build minimal (no default features)
build-minimal:
    cargo build --release --no-default-features

# =============================================================================
# Installation
# =============================================================================

# Interactive install with hardware selection
install:
    ./scripts/install-nym.sh

# Install nym with default features (streaming, progress)
install-default:
    cargo install --path .

# Install nym with NER support (CPU)
install-ner:
    cargo install --path . --features ner

# Install nym with NER + bench support
install-ner-bench:
    cargo install --path . --features "ner,bench"

# Install nym with all features
install-full:
    cargo install --path . --all-features

# =============================================================================
# Running
# =============================================================================

# Run nym with arguments
run *args:
    cargo run -- "$@"

# Run nym with NER feature
run-ner *args:
    cargo run --features ner -- "$@"

# Detect PII in a file
detect file:
    cargo run --release -- detect "{{file}}"

# Detect PII with NER
detect-ner file:
    cargo run --release --features ner -- detect --ner "{{file}}"

# Anonymize a file
anon file:
    cargo run --release -- anon "{{file}}"

# Anonymize with NER
anon-ner file:
    cargo run --release --features ner -- anon --ner "{{file}}"

# =============================================================================
# Examples
# =============================================================================

# Test with example person.md
example-detect:
    cat examples/person.md | cargo run --release -- detect

# Test NER with example person.md
example-detect-ner:
    cat examples/person.md | cargo run --release --features ner -- detect --ner -v

# Anonymize example person.md
example-anon:
    cat examples/person.md | cargo run --release -- anon

# Anonymize example with NER
example-anon-ner:
    cat examples/person.md | cargo run --release --features ner -- anon --ner -v

# Stream anonymize example (streaming is default)
example-stream:
    cat examples/person.md | cargo run --release -- anon --stream

# Stream detect example (streaming is default)
example-stream-detect:
    cat examples/person.md | cargo run --release -- detect --stream

# =============================================================================
# Benchmarking
# =============================================================================

# Run benchmark on HuggingFace dataset (100 examples)
bench-hf:
    cargo run --release --features "ner,bench" -- bench ai4privacy/pii-masking-300k --split train -n 100 --ner

# Run benchmark on HuggingFace dataset (10% sample)
bench-hf-sample dataset="ai4privacy/pii-masking-300k" pct="10":
    cargo run --release --features "ner,bench" -- bench {{dataset}} --split train -p {{pct}} --ner

# Run benchmark on local JSONL file
bench-local file:
    cargo run --release --features "ner,bench" -- bench "{{file}}" --ner

# =============================================================================
# Documentation
# =============================================================================

# Generate and open documentation
docs:
    cargo doc --open --no-deps

# Generate documentation with all features
docs-all:
    cargo doc --open --no-deps --all-features

# =============================================================================
# Maintenance
# =============================================================================

# Update dependencies
update:
    cargo update

# Clean build artifacts
clean:
    cargo clean

# Show active toolchain
toolchain:
    rustup show active-toolchain

# Fetch dependencies
fetch:
    cargo fetch

# =============================================================================
# json schema for config
# =============================================================================
update-schema:
    @cd $(git rev-parse --show-toplevel) && ./scripts/copy_config_schema.sh
