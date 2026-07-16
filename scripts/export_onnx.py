#!/usr/bin/env python3
"""Export a token-classification checkpoint to ONNX and PROVE it matches torch.

Run this with the SAME interpreter that trained the model:

    .venv/bin/python scripts/export_onnx.py models/<ckpt> models/<ckpt>-onnx

Why not optimum-cli in an ephemeral env: `optimum-onnx` pins transformers<5,
so `uv run --with optimum-onnx` resolves a different transformers than the
training venv. ModernBERT's forward changed between 4.x and 5.x, and a v2
artifact exported that way faithfully reproduced the WRONG implementation --
-3.5 ai4 F1 / -4 name recall vs the trained weights, while still passing
optimum's single-short-input validation at 2e-5. The graph must be traced by
the implementation the weights were trained under; exporting in the training
interpreter guarantees that.

The verify gate that would have caught it (and now gates every export):
  - torch-vs-ONNX logits compared at lengths 8..320, not one dummy length
    (the skew was visible at EVERY length, but only checked at one);
  - a padded + batched case, because tracing loves to constant-fold the
    attention mask and every all-ones-mask test would still pass;
  - ONNX runs in a SUBPROCESS that never imports torch: torch and onnxruntime
    fight over OpenMP in one process and whichever initialises first silently
    changes the other's numerics.

Also strips the truncation baked into tokenizer.json by training (max_length
256): anything loading the published artifact with default settings would
silently cap inference at 256 tokens.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

VERIFY_LENS = [8, 16, 32, 48, 64, 96, 128, 192, 256, 320]
TOL = 1e-3


def verify_stage(npz_path, onnx_path):
    """Subprocess entry: onnxruntime only -- torch must never be imported here."""
    import numpy as np
    import onnxruntime as ort

    d = np.load(npz_path)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    failed = False
    for L in VERIFY_LENS:
        ids = d[f"ids_{L}"][None, :].astype(np.int64)
        got = sess.run(None, {"input_ids": ids,
                              "attention_mask": np.ones_like(ids)})[0][0]
        diff = float(np.abs(got - d[f"logits_{L}"]).max())
        status = "ok" if diff < TOL else "FAIL"
        print(f"  len {L:4d}: max |onnx-torch| = {diff:.2e}  {status}")
        failed |= diff >= TOL

    # padded + batched: real tokens must give the same logits as unpadded
    ids = d["ids_192"].astype(np.int64)
    ref = sess.run(None, {"input_ids": ids[None, :],
                          "attention_mask": np.ones((1, 192), np.int64)})[0][0]
    padded = np.concatenate([ids, np.zeros(64, np.int64)])
    batch = np.stack([padded, padded])
    mask = np.stack([np.concatenate([np.ones(192, np.int64), np.zeros(64, np.int64)]),
                     np.concatenate([np.ones(128, np.int64), np.zeros(128, np.int64)])])
    got = sess.run(None, {"input_ids": batch, "attention_mask": mask})[0][0][:192]
    diff = float(np.abs(got - ref).max())
    status = "ok" if diff < TOL else "FAIL"
    print(f"  padded batch : max diff vs unpadded = {diff:.2e}  {status}")
    failed |= diff >= TOL
    sys.exit(1 if failed else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForTokenClassification

    print(f"exporting {args.ckpt} with transformers {transformers.__version__}, "
          f"torch {torch.__version__}")
    args.out.mkdir(parents=True, exist_ok=True)
    model = AutoModelForTokenClassification.from_pretrained(args.ckpt).eval()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    # trace long enough that windowed attention (mmBERT: +/-64) actually binds;
    # a short dummy would trace the local mask as a no-op
    L = 256
    ex = (torch.randint(5, 1000, (1, L), dtype=torch.long),
          torch.ones((1, L), dtype=torch.long))
    torch.onnx.export(
        Wrap(model), ex, str(args.out / "model.onnx"),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"},
                      "attention_mask": {0: "batch_size", 1: "sequence_length"},
                      "logits": {0: "batch_size", 1: "sequence_length"}},
        opset_version=args.opset, do_constant_folding=True, dynamo=False,
    )

    for f in ("config.json", "tokenizer_config.json"):
        if (args.ckpt / f).exists():
            shutil.copy(args.ckpt / f, args.out / f)
    tok = json.load(open(args.ckpt / "tokenizer.json"))
    if tok.get("truncation"):
        print(f"stripping baked truncation from tokenizer.json: {tok['truncation']}")
        tok["truncation"] = None
    json.dump(tok, open(args.out / "tokenizer.json", "w"), ensure_ascii=False)

    # torch reference logits, then hand off to a torch-free subprocess
    rng = np.random.default_rng(0)
    vocab = model.config.vocab_size
    dump = {}
    with torch.no_grad():
        for L in VERIFY_LENS:
            ids = torch.tensor(rng.integers(5, vocab, (1, L)), dtype=torch.long)
            dump[f"ids_{L}"] = ids[0].numpy()
            dump[f"logits_{L}"] = model(
                input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0].numpy()
    with tempfile.NamedTemporaryFile(suffix=".npz") as tf:
        np.savez(tf.name, **dump)
        print("verifying export in a torch-free subprocess:")
        r = subprocess.run([sys.executable, __file__, "--verify-stage",
                            tf.name, str(args.out / "model.onnx")])
    if r.returncode != 0:
        raise SystemExit("EXPORT VERIFICATION FAILED -- do not ship this artifact")
    print(f"export verified: {args.out}/model.onnx matches torch at "
          f"lengths {VERIFY_LENS[0]}..{VERIFY_LENS[-1]} incl. padded batch")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--verify-stage":
        verify_stage(sys.argv[2], sys.argv[3])
    main()
