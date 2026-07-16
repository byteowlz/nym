#!/usr/bin/env bash
# Convert an OpenMed PII DeBERTa-v2 token-classification model to ONNX, and
# (by default) produce an int8-quantized variant for smaller/faster CPU loads.
#
# Produces in the output dir:
#   model.onnx        - fp32 export
#   model_int8.onnx   - dynamic int8 quantization (preferred by nym if present)
#   tokenizer.json, config.json, ...
#
# nym's OpenMed backend loads model_int8.onnx when present, else model.onnx.
#
# WARNING: int8 quantization is OFF by default. Dynamic int8 quantization is
# quantization-sensitive for these DeBERTa-v2 models: it works well for the
# small 44M model (566 MB -> 172 MB, no measurable accuracy loss) but COLLAPSES
# the base/large models (they predict one label for every token). If you enable
# it (NYM_QUANTIZE=1), always validate the result with a quick `nym detect` run.
#
# Requires `uv`. All Python deps run in an ephemeral uv environment so neither
# this repo nor the openmed checkout is modified.
#
# DO NOT use this for models we trained ourselves: optimum-onnx pins
# transformers<5, so the ephemeral env can resolve a DIFFERENT transformers
# than the training env and faithfully export the wrong forward (this cost the
# v2 student -3.5 ai4 F1, undetected by optimum's validation). For our own
# checkpoints use scripts/export_onnx.py, run by the training venv's python --
# it verifies the export against torch at multiple lengths and padded batches.
#
# Usage:
#   scripts/convert_openmed_onnx.sh [MODEL_ID] [OUTPUT_DIR]
#   NYM_QUANTIZE=1 scripts/convert_openmed_onnx.sh   # also emit int8 (small only)
#
# Defaults to the small 44M PII model.
set -euo pipefail

MODEL_ID="${1:-OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1}"
OUT_DIR="${2:-models/$(basename "$MODEL_ID")-onnx}"
QUANTIZE="${NYM_QUANTIZE:-0}"

echo ">> Exporting $MODEL_ID -> $OUT_DIR"
mkdir -p "$OUT_DIR"

uv run --no-project \
  --with "optimum-onnx" \
  --with "onnx" \
  --with "onnxruntime" \
  --with "torch" \
  optimum-cli export onnx \
    --model "$MODEL_ID" \
    --task token-classification \
    --opset 18 \
    "$OUT_DIR"

if [[ "$QUANTIZE" == "1" ]]; then
  echo ">> Quantizing (dynamic int8) -> $OUT_DIR/model_int8.onnx"
  uv run --no-project \
    --with "onnx" \
    --with "onnxruntime" \
    --with "sympy" \
    python - "$OUT_DIR" <<'PY'
import sys
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

out = Path(sys.argv[1])
src = out / "model.onnx"
dst = out / "model_int8.onnx"
quantize_dynamic(
    model_input=str(src),
    model_output=str(dst),
    weight_type=QuantType.QInt8,
)
print(f"   fp32: {src.stat().st_size / 1e6:.1f} MB")
print(f"   int8: {dst.stat().st_size / 1e6:.1f} MB")
PY
fi

echo ">> Done. Contents:"
ls -lh "$OUT_DIR"
