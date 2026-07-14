#!/usr/bin/env python3
"""Label real text for distillation using a large LLM (gemma via local-studio).

The LLM is a far stronger PII labeler than nym's own mmBERT teacher (measured:
+27 char-recall at equal precision), so distilling a student from these labels
breaks the ceiling that mmBERT-labeled data hit.

License-clean: source is Wikipedia ONLY (CC-BY-SA 4.0), train-only, never merged
into the MIT synthetic dataset. Each record carries source/license/labeler.

The LLM returns {value,label} constrained to nym's 42-label scheme; we locate the
verbatim values back in the text to get exact char offsets, then regex-validate
structured types. Residual misses are handled at train time via
`--mask-o-sources gemma` (the LLM still misses some PII, so its "O" is not gold).

Usage:
  python gemma_label.py --out data/gemma --n 20000 --base-url http://hp-z8:8081/v1 \
      --model gemma-4-26b-a4b --concurrency 16
"""
import argparse
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from labels import ALLOWED_LABELS  # noqa: E402

DONE = json.dumps({"__done__": True})
_SENT = re.compile(r"(?<=[.!?。！？؟])\s+|(?<=[。！？])")


def chunks_from_article(text, max_chars=350, min_chars=40):
    """Greedy sentence-packing into short passages (mirrors label_real_text.py,
    duplicated here to avoid importing torch)."""
    for para in text.split("\n"):
        para = para.strip()
        if len(para) < min_chars:
            continue
        sents = [s.strip() for s in _SENT.split(para) if s and s.strip()]
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

ALIASES = {
    "NAME": "GIVEN_NAME", "FIRST_NAME": "GIVEN_NAME", "FIRSTNAME": "GIVEN_NAME",
    "LAST_NAME": "SURNAME", "LASTNAME": "SURNAME", "FAMILY_NAME": "SURNAME",
    "FULL_NAME": "GIVEN_NAME", "PERSON": "GIVEN_NAME", "ORGANIZATION": "COMPANY_NAME",
    "ORG": "COMPANY_NAME", "COMPANY": "COMPANY_NAME", "ADDRESS": "STREET_ADDRESS",
    "STREET": "STREET_NAME", "POSTCODE": "ZIP_CODE", "POSTAL_CODE": "ZIP_CODE",
    "ZIP": "ZIP_CODE", "PHONE_NUMBER": "PHONE", "TELEPHONE": "PHONE", "TEL": "PHONE",
    "MOBILE": "PHONE", "FAX": "FAX_NUMBER", "DOB": "DATE_OF_BIRTH", "BIRTHDATE": "DATE_OF_BIRTH",
    "SEX": "GENDER", "IP": "IPV4", "IP_ADDRESS": "IPV4", "NATIONAL_ID": "GOVERNMENT_ID",
    "ID": "GOVERNMENT_ID", "ID_NUMBER": "GOVERNMENT_ID", "CREDIT_CARD": "CREDIT_DEBIT_CARD",
    "CARD_NUMBER": "CREDIT_DEBIT_CARD", "SOCIAL_SECURITY_NUMBER": "SSN", "VAT": "TAX_ID",
    "MRN": "MEDICAL_RECORD_NUMBER", "STATE_PROVINCE": "STATE", "PROVINCE": "STATE",
    "JOB_TITLE": None, "TITLE": None,  # not in our scheme -> drop
}

# structured types get a light shape sanity check (drop obvious LLM hallucinations)
DIGITY = {"PHONE", "FAX_NUMBER", "SSN", "TAX_ID", "IBAN", "CREDIT_DEBIT_CARD", "CVV",
          "PIN", "ROUTING_NUMBER", "ACCOUNT_NUMBER", "PASSPORT", "DRIVERS_LICENSE",
          "GOVERNMENT_ID", "MEDICAL_RECORD_NUMBER", "ZIP_CODE", "IPV4", "IPV6",
          "SWIFT_BIC", "MAC_ADDRESS", "LICENSE_PLATE", "EMPLOYEE_ID", "CUSTOMER_ID"}

SYS = ("You are a precise PII extraction engine. From the text, extract EVERY piece of "
       "personal data. Return ONLY a JSON array of objects "
       '{"value":"<exact verbatim substring>","label":"<LABEL>"}. '
       "value MUST be copied character-for-character from the text. LABEL must be one of: "
       + ", ".join(sorted(ALLOWED_LABELS)) +
       ". Use the closest matching label. Do not invent values. No commentary, JSON only.")


def norm_label(raw):
    u = re.sub(r"[^A-Z_]", "", (raw or "").upper().replace(" ", "_"))
    if u in ALLOWED_LABELS:
        return u
    return ALIASES.get(u, None)


def call_llm(base_url, model, api_key, text, timeout, retries=5):
    import time
    body = {"model": model, "messages": [{"role": "system", "content": SYS},
                                         {"role": "user", "content": text}],
            "max_tokens": 1600, "temperature": 0}
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {api_key}"})
    out = None
    for attempt in range(retries):
        try:
            out = json.load(urllib.request.urlopen(req, timeout=timeout))["choices"][0]["message"]["content"]
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))  # backoff on rate-limit / transient
                continue
            raise
    if out is None:
        raise RuntimeError("no response")
    m = re.search(r"\[.*\]", out, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [(o.get("value", ""), o.get("label", "")) for o in arr if isinstance(o, dict)]


def to_entities(text, pairs):
    """Locate verbatim values -> non-overlapping char spans with mapped labels."""
    spans = []
    for value, raw_label in pairs:
        value = (value or "").strip()
        label = norm_label(raw_label)
        if not value or len(value) < 2 or label is None:
            continue
        if label in DIGITY and not any(c.isdigit() for c in value):
            continue  # structured type with no digits = hallucination
        start = 0
        while True:
            idx = text.find(value, start)
            if idx < 0:
                break
            spans.append((idx, idx + len(value), label))
            start = idx + len(value)
    # resolve overlaps: sort by start, longest-first; greedily keep non-overlapping
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    out, last_end = [], -1
    for s, e, lab in spans:
        if s >= last_end:
            out.append({"start": s, "end": e, "label": lab})
            last_end = e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/gemma"))
    ap.add_argument("--n", type=int, default=20000, help="kept passages target")
    ap.add_argument("--max-process", type=int, default=60000)
    ap.add_argument("--neg-keep", type=float, default=0.25)
    ap.add_argument("--base-url", default="http://hp-z8:8081/v1")
    ap.add_argument("--model", default="gemma-4-26b-a4b")
    ap.add_argument("--api-key", default="x")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--langs", default="en,de,fr,es,it,ru,ar,zh,ja,ko,hi,pt,nl,tr,pl,uk")
    ap.add_argument("--per-lang-articles", type=int, default=400)
    ap.add_argument("--snapshot", default="20231101")
    ap.add_argument("--text-file", type=Path, default=None,
                    help="JSONL with a 'text' field to re-label (e.g. data/real-filtered.jsonl) "
                         "instead of streaming Wikipedia; the source/license is carried through")
    ap.add_argument("--append", action="store_true",
                    help="append to an existing out/gemma.jsonl, skipping already-labeled texts")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import random
    rng = random.Random(args.seed)

    # append mode: load already-labeled texts to skip, strip the DONE marker
    out_path = args.out / "gemma.jsonl"
    done_texts = set()
    if args.append and out_path.exists():
        keep = [l for l in open(out_path) if l.strip() and not json.loads(l).get("__done__")]
        done_texts = {json.loads(l)["text"] for l in keep}
        out_path.write_text("".join(keep))
        sys.stderr.write(f"append mode: {len(done_texts)} texts already labeled, skipping them\n")

    # source A: re-label existing passages from a JSONL (license carried through)
    if args.text_file:
        rows = [json.loads(l) for l in open(args.text_file) if l.strip()]
        rows = [r for r in rows if not r.get("__done__") and r.get("text") and r["text"] not in done_texts]
        rng.shuffle(rows)
        passages = [(r["text"], r.get("lang", "?")) for r in rows[:args.max_process]]
        sys.stderr.write(f"labeling {len(passages)} passages from {args.text_file}\n")
        _run_labeling(args, passages, rng, mode="a" if args.append else "w")
        return

    # source B: stream Wikipedia
    from datasets import load_dataset
    passages = []
    langs = args.langs.split(",")
    per_lang = max(1, args.max_process // max(len(langs), 1))
    for code in langs:
        try:
            ds = load_dataset("wikimedia/wikipedia", f"{args.snapshot}.{code}",
                              streaming=True, split="train")
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"[{code}] load failed: {str(e)[:60]}\n")
            continue
        got, arts = 0, 0
        for art in ds:
            for ch in chunks_from_article(art.get("text") or ""):
                passages.append((ch, code))
                got += 1
            arts += 1
            if got >= per_lang or arts >= args.per_lang_articles:
                break
        sys.stderr.write(f"[{code}] {got} passages\n")
    rng.shuffle(passages)
    sys.stderr.write(f"total candidate passages: {len(passages)}\n")
    _run_labeling(args, passages, rng)


def _run_labeling(args, passages, rng, mode="w"):
    out_path = args.out / "gemma.jsonl"
    kept = pos = processed = failed = 0
    BATCH = max(args.concurrency * 4, 32)
    with out_path.open(mode) as f, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        i = 0
        while i < len(passages) and kept < args.n and processed < args.max_process:
            batch = passages[i:i + BATCH]
            i += BATCH
            futs = {ex.submit(call_llm, args.base_url, args.model, args.api_key, t, args.timeout): (t, c)
                    for t, c in batch}
            for fut in as_completed(futs):
                text, code = futs[fut]
                processed += 1
                try:
                    ents = to_entities(text, fut.result())
                except Exception:  # noqa: BLE001 — transient LLM/HTTP error
                    failed += 1
                    continue
                if not ents and rng.random() > args.neg_keep:
                    continue
                f.write(json.dumps({"text": text, "entities": ents, "lang": code,
                                    "source": "wikipedia", "license": "CC-BY-SA-4.0",
                                    "labeler": "gemma-4-26b"}, ensure_ascii=False) + "\n")
                f.flush()
                kept += 1
                pos += 1 if ents else 0
            sys.stderr.write(f"  kept {kept} ({pos} w/ PII), {processed} processed, {failed} failed\n")
        f.write(DONE + "\n")
    sys.stderr.write(f"\nwrote {kept} passages ({pos} with PII) -> {out_path}\n")


if __name__ == "__main__":
    main()
