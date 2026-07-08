#!/usr/bin/env python3
"""Build a vocab-pruned tokenizer for the fine-tuned mmBERT PII model.

byte_fallback=True + token-classification head => pruning only touches the input
embedding. We keep every token actually used by the corpus, plus specials and the
256 byte tokens, plus the *merge-closure* (any sub-token needed to still build a
kept token via BPE merges) so nothing fragments differently than before.
"""
import json, collections, sys
from pathlib import Path
from tokenizers import Tokenizer

SRC = Path("models/nym-pii-onnx/tokenizer.json")
OUT = Path("models/nym-pii-pruned/tokenizer.json")
FILES = ["data/pii-bank.train.jsonl","data/pii-bank.val.jsonl","data/pii-bank.test.jsonl"]

tk = Tokenizer.from_file(str(SRC))
raw = json.load(open(SRC))
model = raw["model"]
vocab = model["vocab"]                 # token_str -> id
id2tok = {i:t for t,i in vocab.items()}
merges = model["merges"]               # list of "a b" or ["a","b"]
V = len(vocab)

# 1) tokens actually used by the corpus
counter = collections.Counter()
buf=[]
def flush():
    for enc in tk.encode_batch(buf): counter.update(enc.ids)
for f in FILES:
    for line in open(f):
        buf.append(json.loads(line)["text"])
        if len(buf)>=2000: flush(); buf.clear()
flush()
used = set(counter)                    # ids
print(f"used ids: {len(used):,} / {V:,}")

# 2) always-keep: specials/added + all single-byte tokens (byte_fallback safety net)
added = {a["id"] for a in raw.get("added_tokens",[])}
byte_toks = {vocab[t] for t in vocab if len(t)==6 and t.startswith("<0x") and t.endswith(">")}
print(f"added/special: {len(added)} | byte tokens: {len(byte_toks)}")
keep = set(used) | added | byte_toks

# 3) merge-closure: if we keep a token produced by merge 'a b'->'ab', keep a and b too.
tok2parts = {}
for m in merges:
    a,b = (m if isinstance(m,list) else m.split(" "))
    ab = a+b
    if ab in vocab: tok2parts[vocab[ab]] = (vocab.get(a), vocab.get(b))
changed=True; passes=0
while changed:
    changed=False; passes+=1
    for tid in list(keep):
        pr = tok2parts.get(tid)
        if pr:
            for p in pr:
                if p is not None and p not in keep:
                    keep.add(p); changed=True
print(f"after merge-closure ({passes} passes): keep {len(keep):,} ids")

# 4) contiguous remap, preserving original id order (specials stay at low ids)
kept_sorted = sorted(keep)
old2new = {old:new for new,old in enumerate(kept_sorted)}
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
json.dump({"old2new":{str(k):v for k,v in old2new.items()}, "kept_sorted":kept_sorted},
          open("models/nym-pii-pruned/idmap.json","w"))

# 5) new vocab + filtered merges (keep merge only if a,b,ab all kept)
new_vocab = {id2tok[o]: n for o,n in old2new.items()}
new_merges = []
for m in merges:
    a,b = (m if isinstance(m,list) else m.split(" "))
    ab=a+b
    if a in new_vocab and b in new_vocab and ab in new_vocab:
        new_merges.append(m)
print(f"merges: {len(merges):,} -> {len(new_merges):,}")

raw["model"]["vocab"] = new_vocab
raw["model"]["merges"] = new_merges
for a in raw.get("added_tokens",[]):
    a["id"] = old2new[a["id"]]
# post_processor / template ids that reference specials: remap if present
def remap_pp(pp):
    if not isinstance(pp,dict): return
    for k,v in list(pp.items()):
        if k=="ids" and isinstance(v,list): pp[k]=[old2new.get(x,x) for x in v]
        elif isinstance(v,dict): remap_pp(v)
        elif isinstance(v,list):
            for it in v: remap_pp(it)
remap_pp(raw.get("post_processor") or {})
json.dump(raw, open(OUT,"w"), ensure_ascii=False)
print(f"wrote {OUT}  (vocab {V:,} -> {len(new_vocab):,})")
