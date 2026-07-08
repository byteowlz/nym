#!/usr/bin/env python3
"""Initialize a depth-pruned student from the trained mmBERT-small PII model.

Keeps a subset of transformer layers (default: every other -> 22 becomes 11) and
renumbers them; embedding + classifier head carry over unchanged. The result is
NOT usable as-is — it must be re-finetuned (it recovers quickly because every
kept layer already knows the task).

Usage (on the training box):
  .venv/bin/python scripts/layer_drop.py \
      --src models/nym-pii-mmsmall --out models/nym-pii-mmsmall-11L --keep-every 2
"""
import argparse
import json
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-every", type=int, default=2,
                    help="keep layers 0, k, 2k, ... (2 -> half depth)")
    ap.add_argument("--keep-layers", default=None,
                    help="explicit comma list of layer indices to keep (overrides --keep-every)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    model = AutoModelForTokenClassification.from_pretrained(args.src)
    cfg = model.config
    n = cfg.num_hidden_layers
    if args.keep_layers:
        keep = sorted(int(x) for x in args.keep_layers.split(","))
    else:
        keep = list(range(0, n, args.keep_every))
        if keep[-1] != n - 1:
            keep[-1] = n - 1  # always keep the last layer (feeds the head)
    print(f"layers: {n} -> {len(keep)} (keeping {keep})")

    sd = model.state_dict()
    layer_re = re.compile(r"^(.*\blayers\.)(\d+)(\..*)$")
    new_sd = {}
    remap = {old: new for new, old in enumerate(keep)}
    for k, v in sd.items():
        m = layer_re.match(k)
        if not m:
            new_sd[k] = v
            continue
        idx = int(m.group(2))
        if idx in remap:
            new_sd[f"{m.group(1)}{remap[idx]}{m.group(3)}"] = v
        # dropped layers: skip

    cfg.num_hidden_layers = len(keep)
    # ModernBERT: per-layer attention kinds (global vs local sliding-window) must
    # track the kept layers so each layer keeps the attention type it was trained with.
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = [cfg.layer_types[i] for i in keep]
    small = AutoModelForTokenClassification.from_config(cfg)
    missing, unexpected = small.load_state_dict(new_sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    if missing:
        print(f"note: {len(missing)} missing keys (fresh init): {missing[:4]}")

    out = Path(args.out)
    small.save_pretrained(out)
    try:
        AutoTokenizer.from_pretrained(args.src).save_pretrained(out)
    except Exception as exc:  # tokenizer optional; final run can pass it explicitly
        print(f"tokenizer not copied: {exc}")
    total = sum(p.numel() for p in small.parameters())
    print(f"saved {out}: {total/1e6:.0f}M params "
          f"(body ~{(total - cfg.vocab_size*cfg.hidden_size)/1e6:.0f}M + emb)")


if __name__ == "__main__":
    main()
