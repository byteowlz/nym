#!/usr/bin/env python3
"""Where do the missed PII spans live, and how many are recoverable at decode time?

Two questions, one pass over the OOD sets:

1. DECODE SWEEP -- nym's token backend argmaxes first and thresholds second
   (ner_token.rs), so a token whose entity mass is spread across classes
   (GIVEN_NAME .30 + SURNAME .25 vs O .45) is dropped even at threshold 0.
   Compare against flagging on total entity mass (1 - P(O) > t) across t.
   Char-level P/R -- for redaction, covered characters are what matter.

2. MISS AUTOPSY -- every gold span the argmax decode missed (<50% chars
   covered), bucketed by its max entity mass: 'blind' (<0.1, the model has no
   idea -- only data/labels can fix it), 'low' (0.1-0.3, loss-tilt territory),
   'split' (>=0.3, decode-recoverable); crossed with coarse type and script.

Usage:
  .venv/bin/python scripts/recall_autopsy.py --models small=models/nym-pii-final2 \
      teacher=models/nym-pii-arch-mmbase --device cuda:0
"""
import argparse
import collections
import json
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

THRESHOLDS = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]


def coarse(label):
    l = label.lower()
    if (any(k in l for k in ("firstname", "lastname", "surname", "givenname", "name", "fullname"))
            and "user" not in l and "company" not in l and "domain" not in l):
        return "name"
    if any(k in l for k in ("company", "organization", "org")):
        return "org"
    if any(k in l for k in ("street", "address", "city", "state", "country", "secaddress", "building")):
        return "address"
    return "other"


def script_of(s):
    return "nonlatin" if any(ord(c) > 0x24F for c in s) else "latin"


class ProbModel:
    def __init__(self, path, device, max_len=512):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.model_max_length = max_len
        self.model = AutoModelForTokenClassification.from_pretrained(path).to(device).eval()
        self.o_id = self.model.config.label2id["O"]
        self.device = device
        self.max_len = max_len

    @torch.no_grad()
    def probs(self, text):
        """-> [(char_a, char_b, p_entity, argmax_is_entity), ...] per token"""
        enc = self.tok(text, truncation=True, max_length=self.max_len,
                       return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        p = self.model(**enc).logits[0].float().softmax(-1)
        am = p.argmax(-1).tolist()
        po = p[:, self.o_id].tolist()
        out = []
        for (a, b), o, m in zip(offs, po, am):
            if a == b:
                continue
            out.append((a, b, 1.0 - o, m != self.o_id))
        return out


def char_prf(pred_chars, gold_chars):
    tp = len(pred_chars & gold_chars)
    fp = len(pred_chars - gold_chars)
    fn = len(gold_chars - pred_chars)
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return 100 * p, 100 * r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    rows = []
    for f in sorted(Path("data/testsets").glob("*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    rows += [json.loads(l) for l in open("data/wikiann_ood.jsonl")]
    print(f"{len(rows)} passages, {sum(len(r['entities']) for r in rows)} gold spans\n")

    for spec in args.models:
        name, path = spec.split("=", 1)
        m = ProbModel(path, args.device, args.max_len)

        # one inference pass, all analyses reuse it
        cache = [(r, m.probs(r["text"])) for r in rows]

        print(f"===== {name} =====")
        print(f"{'decode':>16s} {'charP':>6s} {'charR':>6s}")
        base_pred = []
        for r, toks in cache:
            base_pred.append({c for a, b, pe, ise in toks if ise for c in range(a, b)})
        gold_sets = [{c for e in r["entities"] for c in range(e["start"], e["end"])}
                     for r, _ in cache]
        allp = set()
        # aggregate over passages by summed tp/fp/fn
        def agg(preds):
            tp = fp = fn = 0
            for pc, gc in zip(preds, gold_sets):
                tp += len(pc & gc); fp += len(pc - gc); fn += len(gc - pc)
            p = tp / (tp + fp) if tp + fp else 0
            r_ = tp / (tp + fn) if tp + fn else 0
            return 100 * p, 100 * r_
        p, r_ = agg(base_pred)
        print(f"{'argmax (nym)':>16s} {p:6.1f} {r_:6.1f}")
        for t in THRESHOLDS:
            preds = [{c for a, b, pe, _ in toks if pe > t for c in range(a, b)}
                     for r, toks in cache]
            p, r_ = agg(preds)
            print(f"{'1-P(O)>' + f'{t:.2f}':>16s} {p:6.1f} {r_:6.1f}")

        # autopsy at argmax decode
        cat = collections.Counter()
        bytype = collections.defaultdict(collections.Counter)
        total = collections.Counter()
        for (r, toks), pc in zip(cache, base_pred):
            for e in r["entities"]:
                span = set(range(e["start"], e["end"]))
                t_ = coarse(e["label"])
                s_ = script_of(r["text"][e["start"]:e["end"]])
                total[(t_, s_)] += 1
                if len(span & pc) / max(len(span), 1) >= 0.5:
                    continue  # hit
                pe_max = max((pe for a, b, pe, _ in toks
                              if a < e["end"] and b > e["start"]), default=0.0)
                c = "split" if pe_max >= 0.3 else ("low" if pe_max >= 0.1 else "blind")
                cat[c] += 1
                bytype[(t_, s_)][c] += 1
        n_missed = sum(cat.values())
        n_gold = sum(total.values())
        print(f"\nmisses: {n_missed}/{n_gold} gold spans "
              f"({100*n_missed/max(n_gold,1):.1f}%)  "
              f"blind={cat['blind']} low={cat['low']} split={cat['split']}")
        print(f"{'type/script':>20s} {'gold':>6s} {'miss%':>6s} {'blind':>6s} {'low':>5s} {'split':>6s}")
        for k in sorted(total, key=lambda k: -sum(bytype[k].values())):
            b = bytype[k]
            nm = sum(b.values())
            print(f"{k[0]+'/'+k[1]:>20s} {total[k]:6d} {100*nm/max(total[k],1):6.1f} "
                  f"{b['blind']:6d} {b['low']:5d} {b['split']:6d}")
        print()
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
