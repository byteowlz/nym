#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Detect platform
OS="$(uname -s)"
ARCH="$(uname -m)"

echo "=========================================="
echo "  nym - PII Anonymization CLI Installer"
echo "=========================================="
echo ""
echo "Detected platform: ${OS} (${ARCH})"
echo ""
echo "Default features included:"
echo "  - Streaming mode (pipe-friendly stdin/stdout processing)"
echo "  - Progress bars"
echo ""

# Check if NER is wanted
echo "Do you want NER support for detecting names and addresses?"
echo "NER uses machine learning and requires downloading a ~50MB model."
echo ""
echo "  1) No  - Regex-only detection (fast, no model download)"
echo "  2) Yes - Include NER support (more accurate for names/addresses)"
echo ""
read -p "Enter choice [1]: " NER_CHOICE
NER_CHOICE="${NER_CHOICE:-1}"

# Start with default features
FEATURES=""
ENABLE_NER=false
ENABLE_BENCH=false

if [[ "${NER_CHOICE}" == "2" ]]; then
    ENABLE_NER=true
    echo ""
    echo "NER enabled. Select hardware acceleration:"
    echo ""
    echo "  1) CPU only - works everywhere"
    echo "  2) CoreML (Apple Silicon) - macOS only, uses Neural Engine"
    echo "  3) CUDA (NVIDIA GPU) - requires CUDA toolkit"
    echo "  4) TensorRT (NVIDIA optimized) - requires TensorRT"
    echo "  5) ROCm (AMD GPU) - requires ROCm"
    echo "  6) DirectML (Windows GPU) - Windows only"
    echo "  7) OpenVINO (Intel) - Intel CPU/GPU optimization"
    echo ""

    # Default based on platform
    if [[ "${OS}" == "Darwin" && "${ARCH}" == "arm64" ]]; then
        DEFAULT_HW="2"
        echo "Recommended for Apple Silicon: CoreML (2)"
    elif [[ "${OS}" == "Linux" ]] && command -v nvidia-smi &> /dev/null; then
        DEFAULT_HW="3"
        echo "NVIDIA GPU detected, recommended: CUDA (3)"
    else
        DEFAULT_HW="1"
        echo "Recommended: CPU only (1)"
    fi

    echo ""
    read -p "Enter choice [${DEFAULT_HW}]: " HW_CHOICE
    HW_CHOICE="${HW_CHOICE:-$DEFAULT_HW}"

    case "${HW_CHOICE}" in
        1) FEATURES="ner" ;;
        2)
            if [[ "${OS}" != "Darwin" ]]; then
                echo "Warning: CoreML is only available on macOS. Using CPU."
                FEATURES="ner"
            else
                FEATURES="ner-coreml"
            fi
            ;;
        3) FEATURES="ner-cuda" ;;
        4) FEATURES="ner-tensorrt" ;;
        5) FEATURES="ner-rocm" ;;
        6) FEATURES="ner-directml" ;;
        7) FEATURES="ner-openvino" ;;
        *)
            echo "Invalid choice. Using CPU-only NER."
            FEATURES="ner"
            ;;
    esac
fi

# Ask about bench feature
echo ""
echo "Do you want benchmarking support for accuracy testing?"
echo "Bench allows testing detection accuracy against HuggingFace datasets."
echo ""
echo "  1) No  - Skip benchmarking"
echo "  2) Yes - Include bench command (downloads datasets from HuggingFace)"
echo ""
read -p "Enter choice [1]: " BENCH_CHOICE
BENCH_CHOICE="${BENCH_CHOICE:-1}"

if [[ "${BENCH_CHOICE}" == "2" ]]; then
    ENABLE_BENCH=true
    if [[ -n "${FEATURES}" ]]; then
        FEATURES="${FEATURES},bench"
    else
        FEATURES="bench"
    fi
fi

echo ""
echo "=========================================="
echo "  Building nym..."
echo "=========================================="
echo ""

# Default features are always included (streaming, progress)
if [[ -n "${FEATURES}" ]]; then
    echo "Features: default + ${FEATURES}"
    cargo build --release --features "${FEATURES}" --manifest-path "${ROOT_DIR}/Cargo.toml"
else
    echo "Features: default (streaming, progress)"
    cargo build --release --manifest-path "${ROOT_DIR}/Cargo.toml"
fi

echo ""
echo "=========================================="
echo "  Installing nym..."
echo "=========================================="
echo ""

if [[ -n "${FEATURES}" ]]; then
    cargo install --path "${ROOT_DIR}" --features "${FEATURES}" --force
else
    cargo install --path "${ROOT_DIR}" --force
fi

echo ""
echo "=========================================="
echo "  Installation complete!"
echo "=========================================="
echo ""
echo "Binary installed to: \$HOME/.cargo/bin/nym"
echo ""
echo "Quick start:"
echo "  nym detect file.txt          # Detect PII in a file"
echo "  nym anon file.txt            # Anonymize PII in a file"
echo "  nym anon -k keys.jsonl file  # Anonymize with reversible key file"
echo "  nym deanon -k keys.jsonl     # Restore original PII"
echo ""
echo "Streaming mode (included by default):"
echo "  cat file.txt | nym anon --stream         # Stream anonymize"
echo "  cat file.txt | nym detect --stream       # Stream detect"
echo "  some_cmd | nym anon --stream | next_cmd  # Pipeline processing"
echo ""

if [[ "${ENABLE_NER}" == "true" ]]; then
    echo "NER support enabled. Use --ner flag to detect names/addresses:"
    echo "  nym detect --ner file.txt"
    echo "  nym anon --ner file.txt"
    echo ""
    echo "By default nym runs BOTH NER backends and merges results:"
    echo "  gliner  - zero-shot GLiNER span model"
    echo "  openmed - OpenMed clinical/HIPAA PII model (DeBERTa token classifier)"
    echo "Both auto-download from the Hub on first use - no setup needed."
    echo ""
    echo "Pick a single backend with [ner] backend = \"gliner\" | \"openmed\" | \"both\"."
    echo "Choose the OpenMed size via openmed_model = \"Wismut/openmed-onnx/{small,base,large}\"."
    echo "See docs/openmed-ner.md for details."
    echo ""
fi

if [[ "${ENABLE_BENCH}" == "true" ]]; then
    echo "Benchmarking support enabled. Test accuracy with:"
    echo "  nym bench ai4privacy/pii-masking-300k --ner -n 100"
    echo "  nym bench /path/to/dataset.jsonl"
    echo ""
fi
