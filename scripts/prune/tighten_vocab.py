#!/usr/bin/env python3
"""Coverage-targeted vocab tightening for a token-classification ONNX model.

Unlike build_pruned_tokenizer.py (keeps EVERY corpus-used token, ~lossless),
this keeps only the tokens covering --coverage of occurrences PER CORPUS
(union across corpora, so rare-script tokens survive if any corpus needs them),
plus specials, byte tokens, and the BPE merge-closure. Dropped tokens fall back
to byte tokens (byte_fallback=True), costing a few extra fragments — accuracy
must be re-verified on the benchmark gauntlet before accepting.

Usage:
  python tighten_vocab.py --src-tokenizer models/nym-pii-onnx/tokenizer.json \
      --src-onnx models/nym-pii-pre16mk-onnx/model.onnx \
      --out models/nym-pii-small \
      --corpus data/pii-bank.train.jsonl --corpus data/real-filtered.jsonl \
      --coverage 0.995
"""
import argparse
import collections
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer


def count_corpus(tk, path, cap=None):
    counter = collections.Counter()
    buf, n = [], 0
    def flush():
        for enc in tk.encode_batch(buf):
            counter.update(enc.ids)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("__done__"):
            continue
        buf.append(obj["text"])
        n += 1
        if cap and n >= cap:
            break
        if len(buf) >= 2000:
            flush(); buf = []
    flush()
    sys.stderr.write(f"  {path}: {n:,} texts, {len(counter):,} distinct tokens\n")
    return counter


def coverage_keep(counter, coverage):
    total = sum(counter.values())
    keep, cum = set(), 0
    for tid, c in counter.most_common():
        keep.add(tid)
        cum += c
        if cum / total >= coverage:
            break
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-tokenizer", required=True)
    ap.add_argument("--src-onnx", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--corpus", action="append", required=True)
    ap.add_argument("--coverage", type=float, default=0.995)
    args = ap.parse_args()

    tk = Tokenizer.from_file(args.src_tokenizer)
    raw = json.load(open(args.src_tokenizer))
    model = raw["model"]
    vocab = model["vocab"]
    id2tok = {i: t for t, i in vocab.items()}
    merges = model["merges"]
    V = len(vocab)

    # union of per-corpus coverage keeps
    keep = set()
    for c in args.corpus:
        cnt = count_corpus(tk, c)
        k = coverage_keep(cnt, args.coverage)
        sys.stderr.write(f"  -> {len(k):,} tokens for {args.coverage:.1%} coverage\n")
        keep |= k
    sys.stderr.write(f"union: {len(keep):,}\n")

    # specials + byte tokens
    added = {a["id"] for a in raw.get("added_tokens", [])}
    byte_toks = {vocab[t] for t in vocab
                 if len(t) == 6 and t.startswith("<0x") and t.endswith(">")}
    keep |= added | byte_toks

    # merge-closure
    tok2parts = {}
    for m in merges:
        a, b = (m if isinstance(m, list) else m.split(" "))
        ab = a + b
        if ab in vocab:
            tok2parts[vocab[ab]] = (vocab.get(a), vocab.get(b))
    changed = True
    while changed:
        changed = False
        for tid in list(keep):
            pr = tok2parts.get(tid)
            if pr:
                for p in pr:
                    if p is not None and p not in keep:
                        keep.add(p)
                        changed = True
    sys.stderr.write(f"after closure+specials+bytes: {len(keep):,} of {V:,}\n")

    kept_sorted = sorted(keep)
    old2new = {o: n for n, o in enumerate(kept_sorted)}
    args.out.mkdir(parents=True, exist_ok=True)
    json.dump({"old2new": {str(k): v for k, v in old2new.items()},
               "kept_sorted": kept_sorted}, open(args.out / "idmap.json", "w"))

    new_vocab = {id2tok[o]: n for o, n in old2new.items()}
    new_merges = []
    for m in merges:
        a, b = (m if isinstance(m, list) else m.split(" "))
        if a in new_vocab and b in new_vocab and (a + b) in new_vocab:
            new_merges.append(m)
    raw["model"]["vocab"] = new_vocab
    raw["model"]["merges"] = new_merges
    for a in raw.get("added_tokens", []):
        a["id"] = old2new[a["id"]]

    def remap_pp(pp):
        if not isinstance(pp, dict):
            return
        for k, v in list(pp.items()):
            if k == "ids" and isinstance(v, list):
                pp[k] = [old2new.get(x, x) for x in v]
            elif isinstance(v, dict):
                remap_pp(v)
            elif isinstance(v, list):
                for it in v:
                    remap_pp(it)
    remap_pp(raw.get("post_processor") or {})
    json.dump(raw, open(args.out / "tokenizer.json", "w"), ensure_ascii=False)
    sys.stderr.write(f"tokenizer: {V:,} -> {len(new_vocab):,} "
                     f"(merges {len(merges):,} -> {len(new_merges):,})\n")

    # slice the ONNX embedding
    import numpy as np
    import onnx
    from onnx import numpy_helper
    kept_arr = np.array(kept_sorted, dtype=np.int64)
    m = onnx.load(args.src_onnx)
    for i, init in enumerate(m.graph.initializer):
        if len(init.dims) == 2 and init.dims[0] == V:
            arr = numpy_helper.to_array(init)
            m.graph.initializer[i].CopyFrom(
                numpy_helper.from_array(arr[kept_arr], name=init.name))
            sys.stderr.write(f"embedding: {arr.shape} -> ({len(kept_arr)}, {arr.shape[1]})\n")
            break
    else:
        sys.exit("embedding initializer not found")
    onnx.save(m, args.out / "model.onnx", save_as_external_data=False)
    sys.stderr.write(f"saved {args.out}/model.onnx\n")


if __name__ == "__main__":
    main()
