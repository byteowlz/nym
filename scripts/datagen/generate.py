#!/usr/bin/env python3
"""Synthesize PII token-classification training data with exact labels.

Pipeline:
  1. Sweep a coverage RUBRIC of (language x topic x style) cells (rubric.py). For
     each cell a local LLM (OpenAI-compatible: LM Studio / Ollama / vLLM) writes
     [LABEL]-placeholder templates *in that language and register*. A seed bank
     provides an offline fallback (--no-llm).
  2. Fill placeholders with Faker using the cell's matching locale, building the
     string as segments so every value's character span is exact.
  3. Optionally corrupt a fraction of examples with label-preserving NOISE
     (OCR confusions, typos, spacing, case) — spans stay exact (noise.py).
  4. Mix in PII-free negatives, shuffle, split, write JSONL:
       {"text": "...", "entities": [{"start", "end", "label"}]}
     Optionally emit HF BIO token/tag pairs (--to-bio <tokenizer>).

Offsets are CHARACTER indices (matches HF fast-tokenizer offset_mapping), so BIO
conversion is exact including non-ASCII / non-Latin scripts.

Examples:
  # Offline smoke test (seeds only):
  uv run --with faker scripts/datagen/generate.py --no-llm -n 200 -o data/pii.jsonl

  # Full rubric sweep on a local model, with 20% noisy examples:
  uv run --with faker --with openai scripts/datagen/generate.py \
      -n 20000 -o data/pii.jsonl --base-url http://hp-z8:8080/v1 --model step-3.7-flash \
      --cells 300 --per-cell 12 --noise-ratio 0.2 --to-bio bert-base-multilingual-cased
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
from noise import corrupt_text  # noqa: E402
from rubric import cells as rubric_cells  # noqa: E402
from seeds import NEGATIVE_TEMPLATES, SEED_TEMPLATES  # noqa: E402

PLACEHOLDER_RE = re.compile(r"\[([A-Z_]+)\]")

_FAKER_CACHE: dict = {}


def get_faker(locale: str, seed: int) -> Faker:
    f = _FAKER_CACHE.get(locale)
    if f is None:
        try:
            f = Faker(locale)
        except Exception:
            f = Faker("en_US")
        f.seed_instance(seed)
        _FAKER_CACHE[locale] = f
    return f


# ------------------------------------------------------------------ fill / noise
def fill_segments(template: str, faker: Faker):
    """Return a list of {text, label} segments (label=None for literals), or None
    if the template uses an unknown placeholder label."""
    segs = []
    pos = 0
    for m in PLACEHOLDER_RE.finditer(template):
        label = m.group(1)
        if label not in ALLOWED_LABELS:
            return None
        segs.append({"text": template[pos:m.start()], "label": None})
        segs.append({"text": generate_value(label, faker), "label": label})
        pos = m.end()
    segs.append({"text": template[pos:], "label": None})
    return segs


def corrupt_segments(segs, rng, level):
    return [{"text": corrupt_text(s["text"], rng, level), "label": s["label"]} for s in segs]


def assemble(segs):
    """Build (text, entities) from segments, computing exact char offsets."""
    text = ""
    ents = []
    for s in segs:
        start = len(text)
        text += s["text"]
        if s["label"] and s["text"].strip():
            ents.append({"start": start, "end": len(text), "label": s["label"]})
    return text, ents


def valid(text, ents):
    return all(0 <= e["start"] < e["end"] <= len(text) and text[e["start"]:e["end"]].strip() for e in ents)


# ---------------------------------------------------------------------------- llm
def llm_sweep(base_url, model, api_key, n_cells, per_cell, seed, timeout, max_tokens, concurrency):
    """Query the rubric cells concurrently; return list of (template, locale).

    Requests run in a thread pool (vLLM batches them server-side), each with a
    hard timeout and bounded output so one stalled/runaway generation can't
    freeze the sweep — a failing cell is logged and skipped."""
    try:
        from openai import OpenAI
    except ImportError:
        sys.stderr.write("openai not installed; add --with openai (or use --no-llm)\n")
        return []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = OpenAI(base_url=base_url, api_key=api_key or "not-needed",
                    timeout=timeout, max_retries=2)
    labels_str = ", ".join(ALLOWED_LABELS)
    cells = rubric_cells(n_cells, seed)

    def one(cell):
        sys_prompt = (
            f"You write synthetic training templates for a multilingual PII detector. "
            f"Write the natural language in {cell.language.name}. Domain: {cell.topic}. "
            f"Format: {cell.style}. Insert PLACEHOLDERS in square brackets for every "
            f"piece of personal data, using ONLY these English label names: {labels_str}. "
            f"Example placeholder use: 'Patient [GIVEN_NAME] [SURNAME], DOB [DATE_OF_BIRTH]'. "
            f"Put NO real data in the text — only placeholders and surrounding prose in "
            f"{cell.language.name}. Vary sentence length and which labels appear. "
            f"Return ONLY a JSON array of template strings."
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Generate {per_cell} templates."},
            ],
            temperature=1.0, max_tokens=max_tokens, timeout=timeout,
        )
        return [(t, cell.language.faker) for t in _parse_json_array(resp.choices[0].message.content or "")]

    out = []
    done = failed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(one, c): c for c in cells}
        for fut in as_completed(futs):
            done += 1
            try:
                out.extend(fut.result())
            except Exception as exc:  # noqa: BLE001
                failed += 1
                cell = futs[fut]
                if failed <= 10:
                    sys.stderr.write(f"cell failed ({cell.language.name}/{cell.topic}): {exc}\n")
            if done % 25 == 0:
                sys.stderr.write(f"  ...{done}/{len(cells)} cells, {len(out)} templates, {failed} failed\n")
    sys.stderr.write(f"sweep done: {len(out)} templates from {len(cells)} cells ({failed} failed)\n")
    return out


def _parse_json_array(content):
    m = re.search(r"\[.*\]", content, re.S)
    if not m:
        return []
    try:
        return [t for t in json.loads(m.group(0)) if isinstance(t, str) and "[" in t]
    except json.JSONDecodeError:
        return []


# ---------------------------------------------------------------------------- bio
def to_bio(records, tokenizer_name):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    out = []
    for rec in records:
        enc = tok(rec["text"], return_offsets_mapping=True, truncation=True, max_length=512)
        tags = ["O"] * len(enc["input_ids"])
        for ent in rec["entities"]:
            started = False
            for idx, (a, b) in enumerate(enc["offset_mapping"]):
                if a == b or a >= ent["end"] or b <= ent["start"]:
                    continue
                tags[idx] = ("B-" if not started else "I-") + ent["label"]
                started = True
        out.append({"tokens": tok.convert_ids_to_tokens(enc["input_ids"]), "ner_tags": tags})
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--num", type=int, default=200)
    ap.add_argument("-o", "--out", type=Path, default=Path("pii_data.jsonl"))
    ap.add_argument("--no-llm", action="store_true", help="seeds only (offline)")
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--model", default="local-model")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--cells", type=int, default=120, help="rubric cells to query")
    ap.add_argument("--per-cell", type=int, default=12, help="templates per cell")
    ap.add_argument("--request-timeout", type=float, default=180.0, help="per-LLM-request timeout (s)")
    ap.add_argument("--max-tokens", type=int, default=16384, help="max tokens per LLM response (reasoning models need headroom)")
    ap.add_argument("--concurrency", type=int, default=8, help="parallel LLM requests")
    ap.add_argument("--fills-per-template", type=int, default=8)
    ap.add_argument("--noise-ratio", type=float, default=0.2, help="fraction of examples to corrupt")
    ap.add_argument("--noise-level", choices=["light", "medium", "heavy"], default="medium")
    ap.add_argument("--neg-ratio", type=float, default=0.15)
    ap.add_argument("--split", default="0.9,0.05,0.05")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--to-bio", metavar="TOKENIZER", help="also emit *.bio.jsonl (use a multilingual tokenizer)")
    ap.add_argument("--dump-templates", type=Path, help="write the collected templates for inspection")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # (template, faker_locale) pairs: seeds are English; LLM cells carry their locale.
    templates = [(t, "en_US") for t in SEED_TEMPLATES]
    if not args.no_llm:
        got = llm_sweep(args.base_url, args.model, args.api_key, args.cells, args.per_cell,
                        args.seed, args.request_timeout, args.max_tokens, args.concurrency)
        sys.stderr.write(f"LLM produced {len(got)} templates across the rubric\n")
        templates.extend(got)
    # dedup on template text, keep first locale.
    seen = set()
    uniq = []
    for t, loc in templates:
        if t not in seen:
            seen.add(t)
            uniq.append((t, loc))
    templates = uniq
    if args.dump_templates:
        args.dump_templates.write_text("\n".join(f"[{loc}] {t}" for t, loc in templates))

    records = []
    rejected = 0
    guard = 0
    while len(records) < args.num and guard < args.num * 50 + 1000:
        guard += 1
        tmpl, locale = rng.choice(templates)
        faker = get_faker(locale, args.seed)
        for _ in range(args.fills_per_template):
            segs = fill_segments(tmpl, faker)
            if segs is None:
                rejected += 1
                break
            if rng.random() < args.noise_ratio:
                segs = corrupt_segments(segs, rng, args.noise_level)
            text, ents = assemble(segs)
            if not valid(text, ents):
                rejected += 1
                continue
            records.append({"text": text, "entities": ents})
            if len(records) >= args.num:
                break

    for _ in range(int(len(records) * args.neg_ratio)):
        neg = rng.choice(NEGATIVE_TEMPLATES)
        if rng.random() < args.noise_ratio:
            neg = corrupt_text(neg, rng, args.noise_level)
        records.append({"text": neg, "entities": []})
    n_neg = len(records) - args.num if len(records) > args.num else 0

    rng.shuffle(records)

    tr, va, te = (float(x) for x in args.split.split(","))
    n = len(records)
    i_tr, i_va = int(n * tr), int(n * (tr + va))
    splits = {"train": records[:i_tr], "val": records[i_tr:i_va], "test": records[i_va:]}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stem = args.out.with_suffix("")
    for name, recs in splits.items():
        path = Path(f"{stem}.{name}.jsonl")
        with open(path, "w") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        sys.stderr.write(f"wrote {len(recs):>7} -> {path}\n")
        if args.to_bio and recs:
            bio = to_bio(recs, args.to_bio)
            with open(f"{stem}.{name}.bio.jsonl", "w") as fh:
                for r in bio:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            sys.stderr.write(f"wrote {len(bio):>7} -> {stem}.{name}.bio.jsonl\n")

    from collections import Counter
    counts = Counter(e["label"] for r in records for e in r["entities"])
    sys.stderr.write(f"\n{n} examples ({n_neg} negatives), ~{int(args.noise_ratio*100)}% noisy, {rejected} rejected\n")
    sys.stderr.write("label distribution:\n")
    for lab, c in counts.most_common():
        sys.stderr.write(f"  {lab:22s} {c}\n")


if __name__ == "__main__":
    main()
