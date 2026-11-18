#!/usr/bin/env python3
"""Prepare a bounded, pinned OpenPII continuation/distillation experiment.

Only explicit fine-label aliases are accepted. Reject an ENTIRE example if any
annotation is unsupported, overlapping or disagrees with its Unicode offsets;
never make an unknown name/address into O or guess a finer label. Upstream
annotations remain upstream synthetic labels, not human-reviewed complete gold.

Official validation never trains. All its document IDs, normalized texts and
entity-delexicalized contexts block training, including unselected validation
examples. Selection/holdout also share none of these signatures. Hash-priority
sampling scans the whole input, rather than its geographically ordered prefix.
EN/DE receive weight four; other languages receive weight one. The heldout output
is sealed for evaluation after the training recipe has been frozen.

uv run scripts/prepare_openpii.py --raw RAW_DIR --output NEW_DIR \
    --label-config CHECKPOINT/config.json
Outputs are immutable; an existing output directory is an error. No downloads.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import heapq
import json
import math
from pathlib import Path
import unicodedata

REVISION = "a785eb528e28be2693c3718a27e066970de5dadb"
INPUT_HASHES = {
    "train.jsonl": "91ae11b0e1b18793c45ddd11038315fc2c2a98c61926f08dde0ad4fa3df87689",
    "validation.jsonl": "efc7012475c55e76e0930d23d4643931b37424efb9b976dbc3cac9a649a17a5b",
}
ALIASES = {
    "DATE": "DATE", "AGE": "AGE", "GENDER": "GENDER", "SEX": "GENDER",
    "GIVENNAME": "GIVEN_NAME", "SURNAME": "SURNAME", "EMAIL": "EMAIL",
    "CITY": "CITY", "COUNTRY": "COUNTRY", "STREET": "STREET_NAME",
    "BUILDINGNUM": "BUILDING_NUMBER", "ZIPCODE": "ZIP_CODE",
    "TAXNUM": "TAX_ID", "SOCIALNUM": "SSN", "IDCARDNUM": "GOVERNMENT_ID",
    "PASSPORTNUM": "PASSPORT", "CREDITCARDNUMBER": "CREDIT_DEBIT_CARD",
    "TELEPHONENUM": "PHONE", "DRIVERLICENSENUM": "DRIVERS_LICENSE",
    "TIME": "TIME", "URL": "URL", "ORGANISATION": "COMPANY_NAME",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).digest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def annotations(row):
    text = row["source_text"]
    if not isinstance(text, str) or not text or len(text) > 100000:
        raise ValueError("invalid text")
    spans = row["privacy_mask"]
    if not isinstance(spans, list):
        raise ValueError("invalid annotation collection")
    previous = 0
    ordered = sorted(spans, key=lambda s: s["start"])
    for span in ordered:
        a, b = span["start"], span["end"]
        if (type(a) is not int or type(b) is not int
                or not previous <= a < b <= len(text)
                or text[a:b] != span["value"] or not isinstance(span["label"], str)):
            raise ValueError("overlapping or inconsistent codepoint annotation")
        previous = b
    return ordered


def signatures(row, spans=None):
    """Keep namespaces separate; include the provider's source and region in UID."""
    identity = json.dumps([row["source_dataset"], row["region"], row["uid"]],
                          ensure_ascii=False, separators=(",", ":"))
    result = {b"i" + digest(identity), b"t" + digest(normalized(row["source_text"]))}
    if spans is not None:
        parts, end = [], 0
        for span in spans:
            parts.extend((row["source_text"][end:span["start"]], f"<{span['label']}>"))
            end = span["end"]
        parts.append(row["source_text"][end:])
        result.add(b"c" + digest(normalized("".join(parts))))
    return result


def convert(row, spans, label2id):
    if not spans:
        raise ValueError("unreviewed empty annotations")
    if any(s["label"] not in ALIASES for s in spans):
        raise ValueError("unsupported annotation; whole document excluded")
    entities = [{"start": s["start"], "end": s["end"], "label": ALIASES[s["label"]]}
                for s in spans]
    if any(f"B-{s['label']}" not in label2id or f"I-{s['label']}" not in label2id for s in entities):
        raise ValueError("alias absent from frozen taxonomy")
    return {"text": row["source_text"], "entities": entities, "source": "openpii",
            "unit_id": digest(json.dumps([row["source_dataset"], row["region"], row["uid"]],
                                        ensure_ascii=False)).hex(),
            "language": row["language"]}


def priority(row):
    weight = 4 if row["language"] in ("en", "de") else 1
    u = (int.from_bytes(digest("nym-openpii-v1:" + row["text"])[:8], "big") + 1) / (2**64 + 1)
    return -math.log(u) / weight


def offer(heap, limit, record, sigs, serial):
    entry = (-priority(record), serial, record, sigs)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif entry[0] > heap[0][0]:
        heapq.heapreplace(heap, entry)


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for serial, line in enumerate(stream):
            if len(line) > 4 * 1024 * 1024:
                raise ValueError("oversized upstream record")
            yield serial, json.loads(line)


def prepare(raw, output, label2id, train_limit, eval_limit):
    if output.exists():
        raise ValueError("refusing to overwrite an experiment dataset")
    if train_limit < 1 or eval_limit < 1:
        raise ValueError("sample limits must be positive")
    for name, expected in INPUT_HASHES.items():
        if sha256(raw / name) != expected:
            raise ValueError(f"wrong pinned upstream bytes: {name}")
    blocked, evaluation_seen = set(), {"selection": set(), "holdout": set()}
    heaps = {s: [] for s in ("train", "selection", "holdout")}
    stats, unsupported = Counter(), Counter()
    for upstream in ("validation", "train"):
        for serial, row in read_rows(raw / f"{upstream}.jsonl"):
            stats[f"{upstream}_read"] += 1
            if row["split"] != upstream:
                raise ValueError("upstream split marker disagrees with file")
            spans = None
            try:
                spans = annotations(row)
            except (ValueError, KeyError, TypeError):
                stats[f"{upstream}_invalid_offsets"] += 1
            sigs = signatures(row, spans)
            if upstream == "validation":
                blocked.update(sigs)
            elif blocked.intersection(sigs):
                stats["train_validation_overlap_excluded"] += 1
                continue
            if spans is None:
                continue
            unsupported.update(s["label"] for s in spans if s["label"] not in ALIASES)
            try:
                record = convert(row, spans, label2id)
            except ValueError:
                stats[f"{upstream}_unsupported_or_empty"] += 1
                continue
            split = "train"
            if upstream == "validation":
                split = "selection" if digest(record["unit_id"])[0] % 2 == 0 else "holdout"
                other = "holdout" if split == "selection" else "selection"
                if evaluation_seen[other].intersection(sigs):
                    stats["evaluation_cross_partition_duplicate_excluded"] += 1
                    continue
                evaluation_seen[split].update(sigs)
            offer(heaps[split], train_limit if split == "train" else eval_limit, record, sigs, serial)
            stats[f"{split}_eligible"] += 1
            if serial and serial % 100000 == 0:
                print(json.dumps({"phase": upstream, "read": serial, "counts": dict(stats)}), flush=True)
    final, used = {}, {}
    for split, heap in heaps.items():
        records, seen = [], set()
        for _, _, row, sigs in sorted(heap, key=lambda e: (-e[0], e[1])):
            if not seen.intersection(sigs):
                records.append(row)
                seen.update(sigs)
        if not records:
            raise ValueError(f"empty resulting {split} partition")
        final[split], used[split] = records, seen
    if any(used[a].intersection(used[b]) for a, b in
           (("train", "selection"), ("train", "holdout"), ("selection", "holdout"))):
        raise ValueError("cross-partition signature overlap")
    output.mkdir(parents=True)
    partitions = {}
    for split, records in final.items():
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        partitions[split] = {"rows": len(records), "sha256": sha256(path),
                             "languages": dict(Counter(r["language"] for r in records)),
                             "entities": dict(Counter(e["label"] for r in records for e in r["entities"]))}
    report = {"source": "ai4privacy/pii-masking-openpii-1.5m", "revision": REVISION,
              "raw_sha256": INPUT_HASHES, "script_sha256": sha256(__file__),
              "counts": dict(stats), "unsupported_annotations_excluded": dict(unsupported),
              "partitions": partitions, "cross_partition_signature_overlap": 0,
              "limits": {"train": train_limit, "each_evaluation": eval_limit},
              "annotation_basis": "upstream synthetic; not independently complete-reviewed",
              "split_rule": "official validation excluded from train by UID, normalized text and delexicalized context; hash-priority samples; EN/DE weight four"}
    (output / "qa.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label-config", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=32000)
    parser.add_argument("--eval-limit", type=int, default=1024)
    args = parser.parse_args()
    mapping = json.loads(args.label_config.read_text())["label2id"]
    prepare(args.raw, args.output, mapping, args.train_limit, args.eval_limit)


if __name__ == "__main__":
    main()
