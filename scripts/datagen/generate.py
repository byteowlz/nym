#!/usr/bin/env python3
"""Synthesize PII token-classification training data with exact labels.

Pipeline:
  1. Get diverse [LABEL]-placeholder templates — from a local LLM (OpenAI-compatible
     endpoint: Ollama / LM Studio / llama.cpp) and/or the built-in seed bank.
  2. Fill placeholders with Faker, recording each value's exact character span.
  3. Mix in PII-free negatives, shuffle, split, and write JSONL:
       {"text": "...", "entities": [{"start": int, "end": int, "label": "..."}]}
  4. Optionally emit HF BIO token/tag pairs (--to-bio <tokenizer>).

Offsets are CHARACTER indices into `text` (the convention HF fast-tokenizer
offset_mapping uses for BERT/DeBERTa), so the BIO conversion is exact including
non-ASCII. The filler builds the string incrementally, so offsets are correct by
construction — no fragile search-and-replace.

Run (no LLM, seeds only — fully offline):
  uv run --with faker scripts/datagen/generate.py --no-llm -n 200 -o data/pii.jsonl

Run with a local model (e.g. LM Studio on :1234, Ollama on :11434):
  uv run --with faker --with openai scripts/datagen/generate.py \
      -n 5000 -o data/pii.jsonl \
      --base-url http://localhost:1234/v1 --model your-local-model

Convert to BIO for training:
  uv run --with faker --with transformers scripts/datagen/generate.py \
      -n 5000 -o data/pii.jsonl --to-bio bert-base-cased
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from faker import Faker

sys.path.insert(0, str(Path(__file__).parent))
from labels import ALLOWED_LABELS, generate_value  # noqa: E402
from seeds import NEGATIVE_TEMPLATES, SEED_TEMPLATES  # noqa: E402

PLACEHOLDER_RE = re.compile(r"\[([A-Z_]+)\]")


# --------------------------------------------------------------------------- fill
def fill_template(template: str, faker: Faker):
    """Substitute [LABEL] placeholders, returning (text, entities) with exact
    character offsets. Unknown-label placeholders are skipped (left as literals
    would poison training, so we drop the example instead — see generate())."""
    out = []
    entities = []
    cursor = 0
    pos = 0
    for m in PLACEHOLDER_RE.finditer(template):
        label = m.group(1)
        if label not in ALLOWED_LABELS:
            return None, None  # unknown placeholder -> reject template
        literal = template[pos:m.start()]
        out.append(literal)
        cursor += len(literal)
        value = generate_value(label, faker)
        out.append(value)
        entities.append({"start": cursor, "end": cursor + len(value), "label": label})
        cursor += len(value)
        pos = m.end()
    out.append(template[pos:])
    return "".join(out), entities


def validate(text: str, entities: list) -> bool:
    """Every entity span must slice back to a non-empty substring."""
    for e in entities:
        if not (0 <= e["start"] < e["end"] <= len(text)):
            return False
        if not text[e["start"]:e["end"]].strip():
            return False
    return True


# ---------------------------------------------------------------------------- llm
def llm_templates(base_url: str, model: str, api_key: str, n_batches: int, per_batch: int):
    """Ask a local OpenAI-compatible model for [LABEL]-placeholder templates."""
    try:
        from openai import OpenAI
    except ImportError:
        sys.stderr.write("openai not installed; add --with openai (or use --no-llm)\n")
        return []
    client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
    sys_prompt = (
        "You generate synthetic training templates for a PII detector. Output a "
        "JSON array of strings. Each string is a realistic sentence or short "
        "passage that uses PLACEHOLDERS in square brackets for personal data, "
        "e.g. 'Patient [GIVEN_NAME] [SURNAME] was seen on [DATE].' "
        "Use ONLY these placeholder labels: " + ", ".join(ALLOWED_LABELS) + ". "
        "Vary domain (clinical, finance, chat, email, forms, logs, legal), "
        "sentence length, and which labels appear. Put NO real data in the text — "
        "only placeholders and surrounding natural language. Return ONLY the JSON array."
    )
    templates = []
    for i in range(n_batches):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Generate {per_batch} diverse templates. Batch {i + 1}."},
                ],
                temperature=1.0,
            )
            content = resp.choices[0].message.content or ""
            templates.extend(_parse_json_array(content))
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"LLM batch {i + 1} failed: {exc}\n")
    return templates


def _parse_json_array(content: str):
    m = re.search(r"\[.*\]", content, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        return [t for t in arr if isinstance(t, str) and "[" in t]
    except json.JSONDecodeError:
        return []


# --------------------------------------------------------------------------- bio
def to_bio(records, tokenizer_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    out = []
    for rec in records:
        enc = tok(rec["text"], return_offsets_mapping=True, truncation=True, max_length=512)
        tags = ["O"] * len(enc["input_ids"])
        for ent in rec["entities"]:
            started = False
            for idx, (a, b) in enumerate(enc["offset_mapping"]):
                if a == b:  # special token
                    continue
                if a >= ent["end"] or b <= ent["start"]:
                    continue
                tags[idx] = ("B-" if not started else "I-") + ent["label"]
                started = True
        toks = tok.convert_ids_to_tokens(enc["input_ids"])
        out.append({"tokens": toks, "ner_tags": tags})
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--num", type=int, default=200, help="target number of examples")
    ap.add_argument("-o", "--out", type=Path, default=Path("pii_data.jsonl"))
    ap.add_argument("--no-llm", action="store_true", help="use only the seed templates")
    ap.add_argument("--base-url", default="http://localhost:1234/v1", help="OpenAI-compatible endpoint")
    ap.add_argument("--model", default="local-model")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--llm-batches", type=int, default=20)
    ap.add_argument("--llm-per-batch", type=int, default=15)
    ap.add_argument("--fills-per-template", type=int, default=8, help="Faker fills per template")
    ap.add_argument("--locales", default="en_US", help="comma-separated Faker locales")
    ap.add_argument("--neg-ratio", type=float, default=0.15, help="fraction of PII-free examples")
    ap.add_argument("--split", default="0.9,0.05,0.05", help="train,val,test fractions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--to-bio", metavar="TOKENIZER", help="also emit *.bio.jsonl for this HF tokenizer")
    args = ap.parse_args()

    random.seed(args.seed)
    locales = [l.strip() for l in args.locales.split(",") if l.strip()]
    fakers = [Faker(loc) for loc in locales]
    for f in fakers:
        f.seed_instance(args.seed)

    templates = list(SEED_TEMPLATES)
    if not args.no_llm:
        got = llm_templates(args.base_url, args.model, args.api_key, args.llm_batches, args.llm_per_batch)
        sys.stderr.write(f"LLM produced {len(got)} templates\n")
        templates.extend(got)
    templates = list(dict.fromkeys(templates))  # dedup, keep order

    records = []
    rejected = 0
    while len(records) < args.num:
        tmpl = random.choice(templates)
        for _ in range(args.fills_per_template):
            faker = random.choice(fakers)
            text, ents = fill_template(tmpl, faker)
            if text is None or not validate(text, ents):
                rejected += 1
                continue
            records.append({"text": text, "entities": ents})
            if len(records) >= args.num:
                break
    # Negatives
    n_neg = int(len(records) * args.neg_ratio)
    for _ in range(n_neg):
        records.append({"text": random.choice(NEGATIVE_TEMPLATES), "entities": []})

    random.shuffle(records)

    # Split + write
    tr, va, te = (float(x) for x in args.split.split(","))
    n = len(records)
    i_tr, i_va = int(n * tr), int(n * (tr + va))
    splits = {"train": records[:i_tr], "val": records[i_tr:i_va], "test": records[i_va:]}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stem = args.out.with_suffix("")
    for name, recs in splits.items():
        path = args.out if name == "train" and len(splits) == 1 else Path(f"{stem}.{name}.jsonl")
        with open(path, "w") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        sys.stderr.write(f"wrote {len(recs):>6} -> {path}\n")
        if args.to_bio:
            bio = to_bio(recs, args.to_bio)
            bpath = Path(f"{stem}.{name}.bio.jsonl")
            with open(bpath, "w") as fh:
                for r in bio:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            sys.stderr.write(f"wrote {len(bio):>6} -> {bpath}\n")

    # Label stats
    from collections import Counter
    counts = Counter(e["label"] for r in records for e in r["entities"])
    sys.stderr.write(f"\n{n} examples ({n_neg} negatives), {rejected} rejected fills\n")
    sys.stderr.write("label distribution:\n")
    for lab, c in counts.most_common():
        sys.stderr.write(f"  {lab:22s} {c}\n")


if __name__ == "__main__":
    main()
