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

# NER (name/address detection) is always included. Only the ONNX Runtime
# execution provider (hardware acceleration) is selectable. CPU is the base
# `ner` feature, already in nym's default feature set.
FEATURES=""
ENABLE_NER=true
ENABLE_BENCH=false

echo "NER is built in. Select hardware acceleration for it:"
echo ""
echo "  1) CPU only - works everywhere (default)"
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
elif [[ "${OS}" == "Linux" ]] && nvidia-smi -L &> /dev/null; then
    # `nvidia-smi -L` succeeds only when the driver is loaded AND a GPU is
    # present -- unlike `command -v nvidia-smi`, which just finds the binary
    # (installed but non-functional on machines with no working NVIDIA GPU).
    DEFAULT_HW="3"
    echo "NVIDIA GPU detected, recommended: CUDA (3)"
else
    DEFAULT_HW="1"
    echo "Recommended: CPU only (1)"
fi

echo ""
read -p "Enter choice [${DEFAULT_HW}]: " HW_CHOICE
HW_CHOICE="${HW_CHOICE:-$DEFAULT_HW}"

# CPU uses the default features (ner is already default) -> empty FEATURES.
case "${HW_CHOICE}" in
    1) FEATURES="" ;;
    2)
        if [[ "${OS}" != "Darwin" ]]; then
            echo "Warning: CoreML is only available on macOS. Using CPU."
            FEATURES=""
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
        FEATURES=""
        ;;
esac

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

# NER links the ONNX Runtime shared library dynamically. `cargo install` copies
# only the binary to ~/.cargo/bin, leaving libonnxruntime.so behind -> the loader
# can't find it at runtime. Fix: build with an rpath of $ORIGIN so the binary
# searches its own directory, then copy the .so next to it after install.
if [[ "${ENABLE_NER}" == "true" ]]; then
    export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-Wl,-rpath,\$ORIGIN"
fi

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

# Place libonnxruntime.so (and any selected execution-provider libs) next to the
# installed binary so the $ORIGIN rpath resolves them. Symlinks are dereferenced
# with `cp -L`. Only the libs the selected feature actually loads are copied.
if [[ "${ENABLE_NER}" == "true" ]]; then
    BIN_DIR="${CARGO_INSTALL_ROOT:-${CARGO_HOME:-$HOME/.cargo}}/bin"
    REL_DIR="${ROOT_DIR}/target/release"
    LIBS=("libonnxruntime.so")
    case "${FEATURES}" in
        *cuda*)     LIBS+=("libonnxruntime_providers_shared.so" "libonnxruntime_providers_cuda.so") ;;
        *tensorrt*) LIBS+=("libonnxruntime_providers_shared.so" "libonnxruntime_providers_cuda.so" "libonnxruntime_providers_tensorrt.so") ;;
        *rocm*)     LIBS+=("libonnxruntime_providers_shared.so" "libonnxruntime_providers_rocm.so") ;;
    esac
    echo ""
    echo "Bundling ONNX Runtime libraries into ${BIN_DIR}:"
    for lib in "${LIBS[@]}"; do
        if [[ -e "${REL_DIR}/${lib}" ]]; then
            cp -Lf "${REL_DIR}/${lib}" "${BIN_DIR}/${lib}"
            echo "  ${lib}"
        fi
    done
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
    echo "NER is built in. Use --ner to detect names/addresses:"
    echo "  nym detect --ner file.txt"
    echo "  nym anon --ner file.txt"
    echo ""
    echo "By default nym runs BOTH NER backends and merges results:"
    echo "  gliner  - zero-shot GLiNER span model"
    echo "  tokens  - token-classification PII model (nym, OpenMed, Rampart, any HF model)"
    echo "Both auto-download from the Hub on first use - no setup needed."
    echo ""
    echo "Browse and switch models:"
    echo "  nym models list                # catalog: * default, ✓ downloaded"
    echo "  nym models pull [query]        # fuzzy-pick and download"
    echo "  nym models use [query]         # set the default model"
    echo "See docs/ner-backends.md for details."
    echo ""
fi

if [[ "${ENABLE_BENCH}" == "true" ]]; then
    echo "Benchmarking support enabled. Test accuracy with:"
    echo "  nym bench ai4privacy/pii-masking-300k --ner -n 100"
    echo "  nym bench /path/to/dataset.jsonl"
    echo ""
fi
