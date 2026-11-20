#!/usr/bin/env python3
"""Assemble a 40-class coverage-preserving replay + trace + payment + OpenPII corpus.

Combines:
- struct: exact synthetic structured/prose records across all 40 classes (hard O + positive)
- trace_synth: family-partitioned, audited 10-language synthetic agent traces (hard O + positive)
- payment_synth: Horizon-style programmatic card/CVV/expiry/IBAN/BIC snippets with fine-grained nym labels
- wikipedia: script-balanced weak-labeled Wikipedia passages (O-masked in hard CE; teacher KD supervises O)
- openpii: pinned 30-language OpenPII fine-grained replay (O-masked in hard CE; teacher KD supervises O)

Enforces:
- 81-label frozen taxonomy compatibility and 40-class coverage in every partition
- zero token truncation under the target tokenizer (splits records > max_tokens at safe boundaries)
- zero NFKC-casefold-whitespace text overlap across partitions or with reserved evaluation hashes
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, namedtuple
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

IBAN_SPEC = {
    "DE": 18, "ES": 20, "FR": 23, "IT": 23, "NL": 14, "GB": 18,
    "PL": 24, "BE": 12, "AT": 16, "CH": 17, "PT": 21, "SE": 20,
}
NAMES = [
    ("Anna", "Schmidt"), ("Luca", "Rossi"), ("Marie", "Dubois"), ("Carlos", "García"),
    ("Sofia", "Nowak"), ("James", "O'Brien"), ("Yuki", "Tanaka"), ("Min-jun", "Kim"),
    ("Wei", "Zhang"), ("Olga", "Ivanova"), ("Fatima", "Zahra"), ("Priya", "Sharma"),
    ("Tomás", "Silva"), ("Emma", "de Vries"), ("Noah", "Müller"), ("Chloé", "Martin"),
    ("Mateo", "López"), ("Aylin", "Yılmaz"), ("Lars", "Nilsson"), ("Grace", "Okafor"),
]
KW = {
    "en": {"train": [("Card number", "Exp", "CVV"), ("Visa", "valid thru", "CVC"), ("card no.", "expiry", "3-digit code")],
           "eval": [("Credit card", "good thru", "card verification code"), ("Mastercard", "exp.", "CSC")]},
    "de": {"train": [("Kartennummer", "gültig bis", "Prüfnummer"), ("Kreditkarte", "Ablaufdatum", "CVC")],
           "eval": [("Karte", "Verfall", "Sicherheitscode")]},
    "fr": {"train": [("Numéro de carte", "expire fin", "cryptogramme"), ("carte bancaire", "date d'expiration", "CVV")],
           "eval": [("CB", "valable jusqu'au", "code de sécurité")]},
    "es": {"train": [("Número de tarjeta", "caduca", "CVV"), ("tarjeta", "vencimiento", "código de seguridad")],
           "eval": [("Tarjeta de crédito", "válida hasta", "CVC")]},
    "it": {"train": [("Numero carta", "scadenza", "CVV")], "eval": [("carta di credito", "scade", "codice di sicurezza")]},
    "pt": {"train": [("Número do cartão", "validade", "CVV")], "eval": [("cartão", "vence em", "código de segurança")]},
    "nl": {"train": [("Kaartnummer", "geldig tot", "CVC")], "eval": [("creditcard", "vervaldatum", "beveiligingscode")]},
    "pl": {"train": [("Numer karty", "ważna do", "CVV")], "eval": [("karta", "data ważności", "kod CVC")]},
    "ru": {"train": [("Номер карты", "срок действия", "CVV")], "eval": [("карта", "действует до", "код безопасности")]},
    "ja": {"train": [("カード番号", "有効期限", "セキュリティコード")], "eval": [("クレジットカード", "期限", "CVV")]},
    "ko": {"train": [("카드번호", "유효기간", "CVC")], "eval": [("신용카드", "만료일", "보안코드")]},
    "zh": {"train": [("卡号", "有效期", "安全码")], "eval": [("信用卡", "到期", "CVV")]},
}
IBAN_KW = {
    "en": ["IBAN", "bank account"], "de": ["IBAN", "Konto"], "fr": ["IBAN", "RIB"],
    "es": ["IBAN", "cuenta"], "it": ["IBAN", "conto"], "pt": ["IBAN", "NIB"],
    "nl": ["IBAN", "rekeningnummer"], "pl": ["IBAN", "numer konta"], "ru": ["IBAN", "счёт"],
    "ja": ["IBAN", "口座"], "ko": ["IBAN", "계좌"], "zh": ["IBAN", "账户"],
}
TRAIN_TPL = [
    "{ck}: {CARD}, {ek} {EXP}, {vk} {CVV}.",
    "{ck} {CARD} ({ek} {EXP}) {vk}: {CVV}",
    "Hi, it's {GIVEN} {SUR}. {ck} {CARD}, {vk} {CVV}, {ek} {EXP}. Total {AMT}.",
    '{{"card": "{CARD}", "exp": "{EXP}", "cvc": "{CVV}", "amount": "{AMT}"}}',
    "{GIVEN} {SUR} <{EMAIL}> paid {AMT} at {TIME} with {ck} {CARD}.",
    "{ik}: {IBAN}, BIC {BIC}, {GIVEN} {SUR}",
    "Please transfer {AMT} to {ik} {IBAN} (BIC: {BIC}).",
]
EVAL_TPL = [
    "Payment details: {ck} {CARD}; {ek} {EXP}; {vk} {CVV}. Thanks, {GIVEN} {SUR}",
    "{GIVEN} {SUR} wants to update the {ck} on file to {CARD} (exp {EXP}, {vk} {CVV}).",
    "Refund {AMT} to {ik}: {IBAN} (SWIFT {BIC}).",
    "order total {AMT} at {TIME} | {ck} {CARD} | {vk} {CVV}",
]
SCRIPT_PATTERNS = [
    ("ja", r"[\u3040-\u30ff]"), ("ko", r"[\uac00-\ud7af]"), ("zh", r"[\u3400-\u9fff]"),
    ("ar", r"[\u0600-\u06ff]"), ("hi", r"[\u0900-\u097f]"), ("el", r"[\u0370-\u03ff]"),
    ("uk", r"[іїєґІЇЄҐ]"), ("ru", r"[\u0400-\u052f]"),
]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def norm_hash(text: str) -> str:
    norm = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def luhn_complete(rng: random.Random, prefix: str, length: int) -> str:
    digits = [int(c) for c in prefix] + [rng.randint(0, 9) for _ in range(length - len(prefix) - 1)]
    total = 0
    for idx, val in enumerate(reversed(digits)):
        if idx % 2 == 0:
            val = val * 2 - 9 if val * 2 > 9 else val * 2
        total += val
    return "".join(map(str, digits)) + str((10 - total % 10) % 10)


def make_card(rng: random.Random) -> str:
    brand = rng.choice(["visa", "mc", "amex", "discover"])
    if brand == "amex":
        num = luhn_complete(rng, rng.choice(["34", "37"]), 15)
        groups = [num[:4], num[4:10], num[10:]]
    else:
        prefix = {"visa": "4", "mc": str(rng.randint(51, 55)), "discover": "6011"}[brand]
        num = luhn_complete(rng, prefix, 16)
        groups = [num[i:i + 4] for i in range(0, 16, 4)]
    sep = rng.choice([" ", "-", ""])
    return sep.join(groups)


def make_iban(rng: random.Random) -> str:
    country = rng.choice(sorted(IBAN_SPEC))
    length = IBAN_SPEC[country]
    if country in ("NL", "GB"):
        bban = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(4))
        bban += "".join(str(rng.randint(0, 9)) for _ in range(length - 4))
    else:
        bban = "".join(str(rng.randint(0, 9)) for _ in range(length))
    num = "".join(str(int(ch, 36)) for ch in bban + country + "00")
    raw = f"{country}{98 - int(num) % 97:02d}{bban}"
    return " ".join(raw[i:i + 4] for i in range(0, len(raw), 4)) if rng.random() < 0.6 else raw


def fill_payment(rng: random.Random, tpl: str, lang: str, is_eval: bool) -> dict:
    kset = "eval" if is_eval else "train"
    ck, ek, vk = rng.choice(KW[lang][kset])
    given, sur = rng.choice(NAMES)
    ascii_name = f"{given}.{sur}".lower().replace(" ", ".").replace("'", "")
    email = ascii_name.encode("ascii", "ignore").decode() + rng.choice(["@example.com", "@mail.de", "@example.org"])
    month, year = rng.randint(1, 12), rng.randint(26, 34)
    exp = rng.choice([f"{month:02d}/{year}", f"{month:02d}/20{year}", f"{month:02d}-{year}"])
    bic = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(4)) + rng.choice(["DE", "FR", "ES", "GB"]) + "FFXXX"
    vals = {
        "CARD": (make_card(rng), "CREDIT_DEBIT_CARD"),
        "EXP": (exp, "DATE"),
        "CVV": ("".join(str(rng.randint(0, 9)) for _ in range(rng.choice([3, 3, 4]))), "CVV"),
        "IBAN": (make_iban(rng), "IBAN"),
        "BIC": (bic, "SWIFT_BIC"),
        "GIVEN": (given, "GIVEN_NAME"),
        "SUR": (sur, "SURNAME"),
        "EMAIL": (email, "EMAIL"),
        "AMT": (f"€{rng.randint(10, 900)}.{rng.randint(0, 99):02d}", None),
        "TIME": (f"{rng.randint(0, 23):02d}:{rng.choice(['00', '15', '30', '45'])}", None),
    }
    kws = {"ck": ck, "ek": ek, "vk": vk, "ik": rng.choice(IBAN_KW[lang])}
    safe = tpl.replace("{{", "\x01").replace("}}", "\x02")
    expanded = safe.format(**kws, **{k: f"\x00{k}\x00" for k in vals}).replace("\x01", "{").replace("\x02", "}")
    text, entities = "", []
    for idx, part in enumerate(expanded.split("\x00")):
        if idx % 2 == 0:
            text += part
        else:
            val, label = vals[part]
            if label:
                entities.append({"start": len(text), "end": len(text) + len(val), "label": label})
            text += val
    return {"text": text, "entities": entities, "source": "payment_synth", "language": lang}


def generate_payments(count: int, seed: int, is_eval: bool, blocked: set[str]) -> list[dict]:
    rng = random.Random(seed)
    templates = EVAL_TPL if is_eval else TRAIN_TPL
    langs = sorted(KW) + ["en", "de"] * 2
    rows = []
    while len(rows) < count:
        row = fill_payment(rng, rng.choice(templates), rng.choice(langs), is_eval)
        digest = norm_hash(row["text"])
        if digest not in blocked:
            blocked.add(digest)
            rows.append(row)
    return rows


def script_bucket(text: str) -> str:
    for name, pattern in SCRIPT_PATTERNS:
        if re.search(pattern, text):
            return name
    return "latin"


def interleave_by_script(rows: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[script_bucket(row["text"])].append(row)
    ordered = []
    max_len = max((len(v) for v in buckets.values()), default=0)
    for idx in range(max_len):
        for key in sorted(buckets):
            if idx < len(buckets[key]):
                ordered.append(buckets[key][idx])
    return ordered


SplitContext = namedtuple("SplitContext", ("tokenizer", "label2id", "max_tokens", "blocked"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _choose_split_end(text: str, start: int, offsets: list[tuple[int, int]],
                      entities: list[dict], ctx: SplitContext) -> int:
    ceiling = start + offsets[min(ctx.max_tokens - 3, len(offsets)) - 1][1]
    cands = [start + b for _, b in offsets if start < start + b <= ceiling
             and not any(s["start"] < start + b < s["end"] for s in entities)]
    if not cands:
        raise ValueError("entity span exceeds token budget")
    lines = [c for c in cands if text[c - 1] == "\n" and c > start + (ceiling - start) // 2]
    end = max(lines or cands)
    while len(ctx.tokenizer(text[start:end], truncation=False)["input_ids"]) > ctx.max_tokens:
        cands = [c for c in cands if c < end]
        if not cands:
            raise ValueError("cannot split without truncating an entity")
        end = max(cands)
    return end


def split_bounded_record(text: str, entities: list[dict], ctx: SplitContext) -> list[tuple[str, list[dict]]]:
    """Split text at newline/token boundaries without cutting any entity span."""
    if ctx.max_tokens < 8:
        raise ValueError("token budget too small")
    pieces, start = [], 0
    while start < len(text):
        encoded = ctx.tokenizer(text[start:], truncation=False, return_offsets_mapping=True)
        offsets = [(a, b) for a, b in encoded["offset_mapping"] if a < b]
        end = (len(text) if len(encoded["input_ids"]) <= ctx.max_tokens
               else _choose_split_end(text, start, offsets, entities, ctx))
        sub_ents = [{**s, "start": s["start"] - start, "end": s["end"] - start}
                    for s in entities if start <= s["start"] and s["end"] <= end]
        pieces.append((text[start:end], sub_ents))
        start = end
    return pieces


def split_and_validate(row: dict, source: str, ctx: SplitContext) -> list[dict]:
    lang = row.get("language") or row.get("lang")
    pieces = []
    for text, ents in split_bounded_record(row["text"], [dict(e) for e in row["entities"]], ctx):
        prev = 0
        for ent in sorted(ents, key=lambda e: e["start"]):
            a, b, label = ent["start"], ent["end"], ent["label"]
            if not (0 <= prev <= a < b <= len(text)) or f"B-{label}" not in ctx.label2id:
                raise ValueError(f"invalid entity span or label in {source}")
            prev = b
        if len(ctx.tokenizer(text, truncation=False)["input_ids"]) > ctx.max_tokens:
            raise ValueError(f"record exceeds {ctx.max_tokens} tokens after splitting")
        clean = {"text": text, "entities": ents, "source": source}
        if lang:
            clean["language"] = lang
        pieces.append(clean)
    return pieces


def sample_source(path: Path, source: str, target_docs: int, seed: int, ctx: SplitContext) -> list[dict]:
    raw = read_jsonl(path)
    random.Random(seed).shuffle(raw)
    accepted, docs = [], 0
    for row in interleave_by_script(raw):
        doc_hash = norm_hash(row["text"])
        if doc_hash in ctx.blocked:
            continue
        pieces = split_and_validate(row, source, ctx)
        piece_hashes = [norm_hash(p["text"]) for p in pieces]
        if any(h in ctx.blocked for h in piece_hashes) or len(set(piece_hashes)) != len(piece_hashes):
            continue
        ctx.blocked.add(doc_hash)
        ctx.blocked.update(piece_hashes)
        accepted.extend(pieces)
        docs += 1
        if docs == target_docs:
            break
    if docs != target_docs:
        raise ValueError(f"insufficient disjoint records in {path}: got {docs} < {target_docs}")
    return accepted


def write_jsonl(path: Path, rows: list[dict]) -> str:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return sha256(path)


def _cover_language_quota(pool: list[dict], quota: int, seed: int) -> list[dict]:
    random.Random(seed).shuffle(pool)
    chosen, covered = [], set()
    for row in pool:
        found = {e["label"] for e in row["entities"]}
        if found - covered:
            chosen.append(row)
            covered.update(found)
    used = {id(r) for r in chosen}
    chosen.extend(r for r in pool if id(r) not in used)
    return chosen[:quota]


def _append_unique_trace_pieces(rows: list[dict], prior_blocked: set[str],
                                ctx: SplitContext, out: list[dict]) -> None:
    for row in rows:
        for piece in split_and_validate(row, "trace_synth", ctx):
            digest = norm_hash(piece["text"])
            if digest in prior_blocked:
                raise ValueError("trace_synth collided with reserved or cross-partition text")
            if digest not in ctx.blocked:
                ctx.blocked.add(digest)
                out.append(piece)


def select_trace_train(path: Path, required_types: set[str], seed: int, ctx: SplitContext) -> list[dict]:
    prior_blocked = set(ctx.blocked)
    by_lang = defaultdict(list)
    for row in read_jsonl(path):
        by_lang[row["language"]].append(row)
    quota = min(len(v) for v in by_lang.values())
    selected = []
    for _, pool in sorted(by_lang.items()):
        _append_unique_trace_pieces(_cover_language_quota(pool, quota, seed), prior_blocked, ctx, selected)
    if {e["label"] for r in selected for e in r["entities"]} != required_types:
        raise ValueError("trace_synth train failed full entity-type coverage")
    return selected


def load_trace_eval(path: Path, required_types: set[str], ctx: SplitContext) -> list[dict]:
    prior_blocked = set(ctx.blocked)
    out = []
    _append_unique_trace_pieces(read_jsonl(path), prior_blocked, ctx, out)
    if {e["label"] for r in out for e in r["entities"]} != required_types:
        raise ValueError(f"trace eval missing entity types in {path}")
    return out


def summarize_partition(rows: list[dict], digest: str) -> dict:
    return {
        "rows": len(rows),
        "sha256": digest,
        "sources": dict(Counter(r["source"] for r in rows)),
        "classes": len({e["label"] for r in rows for e in r["entities"]}),
        "entities": dict(Counter(e["label"] for r in rows for e in r["entities"])),
    }


def build_replay_trace(args, tokenizer) -> dict:
    if args.output.exists():
        raise ValueError(f"refusing to overwrite existing output directory: {args.output}")
    config = json.loads(args.label_config.read_text(encoding="utf-8"))
    label2id = config["label2id"]
    required_types = {k[2:] for k in label2id if k.startswith("B-")}
    reserved = set(json.loads(args.reserved_hashes.read_text(encoding="utf-8")))
    ctx = SplitContext(tokenizer, label2id, args.max_tokens, set(reserved))

    holdout = load_trace_eval(args.trace_dir / "trace.test.jsonl", required_types, ctx)
    holdout.extend(sample_source(args.openpii_dir / "holdout.jsonl", "openpii",
                                 args.openpii_eval, args.seed + 2, ctx))
    holdout.extend(generate_payments(args.payment_eval, args.seed + 20, True, ctx.blocked))

    selection = load_trace_eval(args.trace_dir / "trace.val.jsonl", required_types, ctx)
    selection.extend(sample_source(args.openpii_dir / "selection.jsonl", "openpii",
                                   args.openpii_eval, args.seed + 1, ctx))
    selection.extend(generate_payments(args.payment_eval, args.seed + 10, True, ctx.blocked))

    train = select_trace_train(args.trace_dir / "trace.train.jsonl", required_types, args.seed, ctx)
    train.extend(sample_source(args.struct_train, "struct", args.struct_count, args.seed, ctx))
    train.extend(sample_source(args.gemma_train, "wikipedia", args.wiki_each, args.seed, ctx))
    train.extend(sample_source(args.real_train, "wikipedia", args.wiki_each, args.seed + 3, ctx))
    train.extend(sample_source(args.openpii_dir / "optimizer.jsonl", "openpii",
                               args.openpii_train, args.seed + 4, ctx))
    train.extend(generate_payments(args.payment_train, args.seed, False, ctx.blocked))
    random.Random(args.seed).shuffle(train)

    args.output.mkdir(parents=True)
    parts = {}
    for name, rows in (("train", train), ("selection", selection), ("holdout", holdout)):
        if {e["label"] for r in rows for e in r["entities"]} != required_types:
            raise ValueError(f"{name} does not cover all {len(required_types)} required entity types")
        digest = write_jsonl(args.output / f"{name}.jsonl", rows)
        parts[name] = summarize_partition(rows, digest)
    report = {
        "schema": "nym.replay-trace.qa.v1",
        "max_tokens": args.max_tokens,
        "truncated_records": 0,
        "reserved_hash_overlaps": 0,
        "cross_partition_overlaps": 0,
        "required_entity_types": len(required_types),
        "partitions": parts,
        "script_sha256": sha256(Path(__file__)),
    }
    (args.output / "qa.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--struct-train", type=Path, required=True)
    ap.add_argument("--gemma-train", type=Path, required=True)
    ap.add_argument("--real-train", type=Path, required=True)
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--openpii-dir", type=Path, required=True)
    ap.add_argument("--reserved-hashes", type=Path, required=True)
    ap.add_argument("--label-config", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--struct-count", type=int, default=6000)
    ap.add_argument("--wiki-each", type=int, default=2000)
    ap.add_argument("--openpii-train", type=int, default=6000)
    ap.add_argument("--openpii-eval", type=int, default=400)
    ap.add_argument("--payment-train", type=int, default=2000)
    ap.add_argument("--payment-eval", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=49)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tok.backend_tokenizer.no_truncation()
    tok.backend_tokenizer.no_padding()
    print(json.dumps(build_replay_trace(args, tok), indent=2))


if __name__ == "__main__":
    main()
