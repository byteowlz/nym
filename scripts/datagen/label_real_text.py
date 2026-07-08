#!/usr/bin/env python3
"""Teacher-label real multilingual text (Wikipedia) for distillation.

License hygiene: source is Wikipedia ONLY (CC-BY-SA 4.0). Every record carries
source+license metadata. This corpus is TRAIN-ONLY — it must not be merged into
the published MIT synthetic dataset (which claims "no real personal information").

Streams wikimedia/wikipedia per language, chunks articles into short passages,
runs the fine-tuned mmBERT teacher on GPU, and writes char-offset JSONL in the
same shape as the synthetic data: {"text", "entities":[{start,end,label,conf}]}.

Keeps every passage with >=1 detected entity; keeps a fraction of all-O passages
(real negatives reduce false positives). Resumable: skips languages whose output
file already ends with the DONE marker.

Usage (on the training box):
  .venv/bin/python label_real_text.py --model models/nym-pii-mmbert \
      --out data/real --per-lang 25000
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch

# wikipedia code -> our language name (matches rubric languages)
LANGS = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "pl": "Polish",
    "sv": "Swedish", "cs": "Czech", "ro": "Romanian", "tr": "Turkish",
    "fi": "Finnish", "da": "Danish", "el": "Greek", "ru": "Russian",
    "uk": "Ukrainian", "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
    "ar": "Arabic", "hi": "Hindi",
}

SENT_SPLIT = re.compile(r"(?<=[.!?。！？؟])\s+|(?<=[。！？])")
DONE = json.dumps({"__done__": True})


def chunks_from_article(text, max_chars=350, min_chars=40):
    """Greedy sentence packing into short passages."""
    # first paragraph blocks, then sentences
    for para in text.split("\n"):
        para = para.strip()
        if len(para) < min_chars:
            continue
        sents = [s.strip() for s in SENT_SPLIT.split(para) if s and s.strip()]
        cur = ""
        for s in sents:
            if cur and len(cur) + len(s) + 1 > max_chars:
                if len(cur) >= min_chars:
                    yield cur
                cur = s
            else:
                cur = f"{cur} {s}".strip() if cur else s
        if len(cur) >= min_chars:
            yield cur[:max_chars * 2]


@torch.inference_mode()
def label_batch(model, tok, id2label, texts, device, max_length=256):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=max_length, return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc).logits
    probs = torch.softmax(logits.float(), dim=-1)
    conf, pred = probs.max(-1)
    pred = pred.cpu().numpy()
    conf = conf.cpu().numpy()
    mask = enc["attention_mask"].cpu().numpy()
    out = []
    for bi, text in enumerate(texts):
        ents = []
        cur = None  # [type, start, end, [confs]]

        def push():
            nonlocal cur
            if cur:
                t, s, e, cs = cur
                while s < e and text[s].isspace():
                    s += 1
                while e > s and text[e - 1].isspace():
                    e -= 1
                if e > s:
                    ents.append({"start": int(s), "end": int(e), "label": t,
                                 "conf": round(float(sum(cs) / len(cs)), 4)})
                cur = None

        for ti in range(len(pred[bi])):
            if not mask[bi][ti]:
                break
            st, en = int(offsets[bi][ti][0]), int(offsets[bi][ti][1])
            lab = id2label[str(int(pred[bi][ti]))]
            if st == en:
                continue  # specials
            t = lab[2:] if lab[:2] in ("B-", "I-") else (None if lab == "O" else lab)
            if t is None:
                push()
                continue
            if cur and cur[0] == t:
                cur[2] = en
                cur[3].append(conf[bi][ti])
            else:
                push()
                cur = [t, st, en, [conf[bi][ti]]]
        push()
        out.append(ents)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, default=Path("data/real"))
    ap.add_argument("--per-lang", type=int, default=25000, help="kept passages per language")
    ap.add_argument("--max-process", type=int, default=120000, help="processed passages cap per language")
    ap.add_argument("--neg-keep", type=float, default=0.3, help="keep prob for all-O passages")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--min-conf", type=float, default=0.0, help="drop entities below this conf (filter later instead)")
    ap.add_argument("--langs", default=",".join(LANGS), help="comma list of wiki codes")
    ap.add_argument("--snapshot", default="20231101")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    model.to(device).eval()
    id2label = {str(k): v for k, v in model.config.id2label.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    for code in args.langs.split(","):
        code = code.strip()
        out_path = args.out / f"real.{code}.jsonl"
        if out_path.exists():
            try:
                last = out_path.read_text().rstrip("\n").rsplit("\n", 1)[-1]
                if last == DONE:
                    sys.stderr.write(f"[{code}] already done, skipping\n")
                    continue
            except Exception:
                pass
        sys.stderr.write(f"[{code}] streaming wikipedia {args.snapshot}.{code}\n")
        ds = load_dataset("wikimedia/wikipedia", f"{args.snapshot}.{code}",
                          streaming=True, split="train")
        kept = pos = processed = 0
        buf = []
        with out_path.open("w") as f:
            def flush(buf):
                nonlocal kept, pos, processed
                if not buf:
                    return
                ents_list = label_batch(model, tok, id2label, buf, device)
                for text, ents in zip(buf, ents_list):
                    processed += 1
                    if args.min_conf:
                        ents = [e for e in ents if e["conf"] >= args.min_conf]
                    if not ents and rng.random() > args.neg_keep:
                        continue
                    rec = {"text": text, "entities": ents, "lang": code,
                           "source": "wikipedia", "license": "CC-BY-SA-4.0"}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    kept += 1
                    if ents:
                        pos += 1

            for art in ds:
                for ch in chunks_from_article(art.get("text") or ""):
                    buf.append(ch)
                    if len(buf) >= args.batch_size:
                        flush(buf)
                        buf = []
                if kept >= args.per_lang or processed >= args.max_process:
                    break
            if kept < args.per_lang and processed < args.max_process:
                flush(buf)
            f.write(DONE + "\n")
        sys.stderr.write(f"[{code}] kept {kept} ({pos} with PII) of {processed} processed\n")

    sys.stderr.write("all languages done\n")


if __name__ == "__main__":
    main()
