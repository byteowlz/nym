#!/usr/bin/env python3
"""Generate fixed-shape ONNX variants of a token-classification model for
CoreML static-shape buckets.

The Apple GPU path (trx-zp2t) needs ONNX files whose input/output dimensions
are concrete: the CoreML execution provider only compiles an MLProgram when no
dimension is unbounded, and one specialization per shape bucket is what makes
it fast. This script rewrites the symbolic dimensions (`batch_size`,
`sequence_length`) of an exported fp32 `model.onnx` into concrete numbers, one
file per bucket, without touching the graph itself.

Usage:
    uv run --no-project --with onnx python scripts/make_static_buckets.py \
        <model_dir> [--buckets 64x64,16x128,4x512]

Writes `<model_dir>/static/model_static_<B>x<S>.onnx` for every bucket.
The runtime picks these up when `[ner] provider = "auto"` on a CoreML build
(see docs/ner-backends.md); without them it stays on the CPU path.
"""

import argparse
import pathlib
import sys

import onnx

DEFAULT_BUCKETS = "64x64,16x128,4x512"


def pin_dimensions(value_type, batch: int, seq: int) -> bool:
    """Replace symbolic dims with (batch, seq). Returns True when it changed."""
    dims = value_type.tensor_type.shape.dim
    changed = False
    for dim in dims:
        if dim.HasField("dim_value"):
            continue
        symbolic = dim.dim_param or ""
        if symbolic == "sequence_length":
            value = seq
        elif symbolic in ("batch_size", "batch", ""):
            value = batch
        else:
            sys.exit(f"unexpected symbolic dimension {symbolic!r}; buckets need batch/seq axes only")
        dim.ClearField("dim_param")
        dim.dim_value = value
        changed = True
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_dir", type=pathlib.Path, help="dir containing fp32 model.onnx")
    parser.add_argument("--buckets", default=DEFAULT_BUCKETS, help="comma-separated BxS shapes")
    args = parser.parse_args()

    source = args.model_dir / "model.onnx"
    if not source.is_file():
        sys.exit(f"no fp32 model at {source}; point --model-dir at the fp32 export")
    out_dir = args.model_dir / "static"
    out_dir.mkdir(parents=True, exist_ok=True)

    for bucket in args.buckets.split(","):
        batch_s, seq_s = bucket.lower().split("x")
        batch, seq = int(batch_s), int(seq_s)
        if min(batch, seq) < 1:
            sys.exit(f"invalid bucket {bucket}")

        model = onnx.load(str(source))
        touched = []
        for value in (*model.graph.input, *model.graph.output):
            if pin_dimensions(value.type, batch, seq):
                touched.append(value.name)
        target = out_dir / f"model_static_{batch}x{seq}.onnx"
        onnx.save(model, str(target))
        print(f"{target} (fixed {', '.join(touched)})")


if __name__ == "__main__":
    main()
