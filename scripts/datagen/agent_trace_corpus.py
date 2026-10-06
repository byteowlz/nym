#!/usr/bin/env python3
"""Offline exact-labelled trace corpus. No private logs, model, Faker or network.

uv run --offline --no-project scripts/datagen/agent_trace_corpus.py \
  --label-config /local/checkpoint/config.json --output-dir /tmp/nym-corpus

Pass the EXISTING 81-label checkpoint config: labels.py has 42 generators, so
silently deriving a new head from it would break compatibility. IDs are kept
in EXACT supplied order, not alphabetized. Training must use that frozen map;
we cover every frozen class in a train-only private-record family. An ONNX-only
public model is not a trainable checkpoint; config is enough for this generator.
Character offsets feed train_ner.py; byte offsets prove UTF-8 integrity.
All O tokens supervise, including entire negative records: never mask this
source with --mask-o-sources. Every scenario family stays in one split.
Known multilingual datasets must be replayed separately, not called synthetic
gold. No old benchmark is read. This is NOT evidence of trained model quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

SOURCE = "agent_trace_exact_v1"
# Generator vocabulary, NOT a replacement for an existing checkpoint's map.
VALUES = {
    "ACCOUNT_NUMBER": "672849103256", "AGE": "37", "API_KEY": "sk-trace-fake-79b26e41c8a03d65",
    "BUILDING_NUMBER": "219", "CITY": "Testville", "COMPANY_NAME": "Fictional Orchard Ltd",
    "COUNTRY": "Testland", "CREDIT_DEBIT_CARD": "4111111111111111", "CUSTOMER_ID": "CID-726304",
    "CVV": "729", "DATE": "2031-07-23", "DATE_OF_BIRTH": "1987-04-19",
    "DRIVERS_LICENSE": "T72849361", "EMAIL": "zoe.trace@example.invalid", "EMPLOYEE_ID": "EMP-72946",
    "FAX_NUMBER": "+1-202-555-0198", "GENDER": "non-binary", "GIVEN_NAME": "Zoë",
    "GOVERNMENT_ID": "726304819", "IBAN": "DE89370400440532013000", "IPV4": "192.0.2.47",
    "IPV6": "2001:db8::47", "LICENSE_PLATE": "TST-7294", "MAC_ADDRESS": "02:00:00:00:00:47",
    "MEDICAL_RECORD_NUMBER": "MRN726304", "PASSPORT": "T72849360", "PASSWORD": "TraceOnly!79xQ",
    "PHONE": "+1-202-555-0147", "PIN": "7392", "ROUTING_NUMBER": "021000021",
    "SECONDARY_ADDRESS": "Apt 219", "SSN": "900-12-3478", "STATE": "Testshire",
    "STREET_ADDRESS": "219 Example Lane", "STREET_NAME": "Example Lane", "SURNAME": "Márquez",
    "SWIFT_BIC": "TESTDEFFXXX", "TAX_ID": "92-7263048", "TIME": "07:23:41",
    "URL": "https://docs.example.invalid/project", "USERNAME": "trace_user_47", "ZIP_CODE": "72946",
}
TRANSPORTS = ("code", "log", "config", "diff", "shell", "tool")
# Independent semantic scenario templates, not train/holdout rewrites of one
# fixture. Contrasts and repeated occurrences ALWAYS inherit this family split.
# Slots [[O:TYPE]] are explicitly benign; [[TYPE]] are exact private spans.
SCENARIOS = {
    "allocator": ("code", "const slab_bytes = [[O:PIN]]; // allocation only", "// user's door access PIN: [[PIN]]"),
    "compiler": ("log", "compile emitted at [[O:DATE_OF_BIRTH]]; version 2.8.1; bash --release", "support log: user's birth date [[DATE_OF_BIRTH]]"),
    "dependency": ("config", "public_documentation = '[[O:URL]]'", "private_reset_link = '[[URL]]'"),
    "git-author": ("diff", "-// API example identifier: [[O:GIVEN_NAME]]", "+// Author: [[GIVEN_NAME]] [[SURNAME]] <[[EMAIL]]>"),
    "curl-auth": ("shell", "printf '%s' '[[O:API_KEY]]' # published parser fixture, not an active key", "curl -H 'Authorization: Bearer [[API_KEY]]' https://api.example.invalid"),
    "review-comment": ("tool", "public syntax tutorial: enum [[O:SURNAME]] { Ready }", "// reviewer who supplied the private report: [[GIVEN_NAME]] [[SURNAME]]"),
    "socket-port": ("config", "listen_port = [[O:PIN]] # transport setting, not a PIN", "login_recovery_pin = '[[PIN]]'"),
    "artifact-time": ("shell", "touch -t 203107230723 artifact; printf 'build time [[O:TIME]]'", "printf 'patient appointment time [[TIME]]'"),
    "public-origin": ("log", "GET [[O:URL]] # public package manual, status 200", "patient log: personal secret share URL [[URL]]"),
    "dotenv": ("code", "const parser_checksum = '[[O:PASSWORD]]'; // public test vector", "const liveCredentials = { password: '[[PASSWORD]]', key: '[[API_KEY]]' };"),
    "tool-name": ("tool", "tool=bash model=example-encoder gpu=ExampleGPU flags=--offline path=/tmp/build hash=a901fed2", "mail inspection tool: private mailbox owner [[GIVEN_NAME]] [[SURNAME]], delivery address [[EMAIL]]"),
    "wheel-date": ("diff", "-release_date = '[[O:DATE_OF_BIRTH]]' # distribution artifact", "+account_birth_date = '[[DATE_OF_BIRTH]]'"),
    "retry-budget": ("shell", "export RETRY_BUDGET=[[O:PIN]]; echo 'not a PIN; attempt count'", "export CUSTOMER_BANK_PIN='[[PIN]]'"),
    "calendar-build": ("tool", "builder output: reproducible timestamp [[O:DATE_OF_BIRTH]]T07:23:41Z", "tool response: applicant date of birth [[DATE_OF_BIRTH]]"),
    "link-index": ("code", "// indexed public documentation URL: [[O:URL]]", "// private tenant access URL: [[URL]]"),
    "commit-review": ("log", "parser snapshot: [[O:GIVEN_NAME]] is the name of a public enum, not a person", "audit: private change requested by [[GIVEN_NAME]] [[SURNAME]]"),
    "secret-rotation": ("diff", "-test_vector = '[[O:API_KEY]]' # publicly documented sentinel", "+deployment_token = '[[API_KEY]]'"),
    "maintainer-env": ("config", "example_handle = '[[O:USERNAME]]' # public CLI usage fixture", "# .env read from the maintainer's private checkout\nMAINTAINER_LOGIN=[[USERNAME]]\nMAINTAINER_EMAIL=[[EMAIL]]\nAPI_KEY=[[API_KEY]]\nPASSWORD=[[PASSWORD]]"),
}
MARKER = re.compile(r"\[\[(O:)?([A-Z][A-Z_0-9]*)\]\]")


def load_label_map(config: dict) -> list[str]:
    """Fail rather than invent labels, change IDs or silently shrink the head."""
    mapping = config.get("id2label")
    if not isinstance(mapping, dict) or set(mapping) != {str(i) for i in range(81)}:
        raise ValueError("existing config must contain exactly 81 contiguous id2label entries")
    labels = [mapping[str(i)] for i in range(81)]
    types = sorted({label[2:] for label in labels if isinstance(label, str) and label.startswith("B-")})
    expected = {"O"} | {f"{prefix}-{label}" for label in types for prefix in ("B", "I")}
    if (not all(isinstance(label, str) for label in labels) or set(labels) != expected
            or len(types) != 40 or not set(types) <= VALUES.keys()):
        raise ValueError("frozen BIO map is unsupported, incomplete or contains duplicate labels")
    if "label2id" in config and config["label2id"] != {label: i for i, label in enumerate(labels)}:
        raise ValueError("config label maps disagree")
    required = {match[2] for pair in SCENARIOS.values() for text in pair[1:] for match in MARKER.finditer(text)}
    if not required <= set(types):
        raise ValueError("checkpoint lacks a required trace label")
    return labels


def family_splits(seed: int) -> dict[str, str]:
    """Deterministic, stratified family allocation; literals never choose a split."""
    result = {"private-record": "train"}
    for transport in TRANSPORTS:
        names = [name for name, pair in SCENARIOS.items() if pair[0] == transport]
        names.sort(key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).digest())
        for name, split in zip(names, ("train", "selection", "holdout"), strict=True):
            result[name] = split
    return result


def render(template: str, values: dict[str, str]) -> tuple[str, list, list]:
    """Build spans while appending, never find/deduplicate a literal after rendering."""
    parts, entities, benign, cursor, length = [], [], [], 0, 0
    for match in MARKER.finditer(template):
        prefix = template[cursor:match.start()]
        parts.append(prefix)
        length += len(prefix)
        label, is_o = match[2], bool(match[1])
        value = values[label]
        span = {"start": length, "end": length + len(value), "label": "O" if is_o else label, "value": value}
        if is_o:
            span["contrast_label"] = label
        (benign if is_o else entities).append(span)
        parts.append(value)
        length += len(value)
        cursor = match.end()
    parts.append(template[cursor:])
    text = "".join(parts)
    if "[[" in text or "]]" in text:
        raise ValueError("unresolved annotation marker")
    for span in entities + benign:
        span["byte_start"] = len(text[:span["start"]].encode("utf-8"))
        span["byte_end"] = len(text[:span["end"]].encode("utf-8"))
    return text, entities, benign


def validate_row(row: dict, labels: list[str]) -> None:
    text = row["text"]
    if not isinstance(text, str) or not text or row["supervision"] != "full":
        raise ValueError("invalid full-supervision row")
    occupied = set()
    for span in row["entities"] + row["benign"]:
        a, b = span["start"], span["end"]
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(text):
            raise ValueError("invalid character offsets")
        if span["label"] != "O" and f'B-{span["label"]}' not in labels:
            raise ValueError("unfrozen label")
        if text[a:b] != span["value"] or not span["value"]:
            raise ValueError("annotation text disagrees")
        if (span["byte_start"], span["byte_end"]) != (len(text[:a].encode()), len(text[:b].encode())):
            raise ValueError("UTF-8 offsets disagree")
        positions = set(range(a, b))
        if occupied & positions:
            raise ValueError("overlapping annotations")
        occupied |= positions


def generate(labels: list[str], seed: int = 47, replicas: int = 8) -> list[dict]:
    load_label_map({"id2label": {str(i): label for i, label in enumerate(labels)}})
    if replicas < 1:
        raise ValueError("replicas must be positive")
    splits, rows = family_splits(seed), []
    for replica in range(replicas):
        values = dict(VALUES)
        # Literal pools deliberately recur across splits and contrast polarities;
        # scenario-family identity, NOT private values, defines independence.
        values.update(PIN=str(7301 + replica), DATE_OF_BIRTH=f"1987-04-{replica % 28 + 1:02d}",
                      URL=f"https://docs.example.invalid/project/{replica}",
                      API_KEY=f"sk-trace-fake-{seed:08x}{replica:08x}",
                      GIVEN_NAME=("Zoë", "Renée", "李", "Ирина")[replica % 4])
        for family, (transport, negative, positive) in SCENARIOS.items():
            variants = {"negative": negative, "positive": positive,
                        "mixed": negative + "\n" + positive,
                        "negated-with-secret": "Not a PIN: the buffer size is 2048. Actual credentials follow.\n" + positive}
            for variant, template in variants.items():
                text, entities, benign = render(template + f"\n# synthetic run {replica}", values)
                rows.append(dict(text=text, entities=entities, benign=benign, source=SOURCE,
                                 language="en", supervision="full", scenario_family=family,
                                 transport=transport, split=splits[family], variant=variant,
                                 contrast_id=f"{family}:{replica}"))
    # Full vocabulary coverage, training only. Benign literals here are explicitly
    # publicly documented parser snapshots, NOT denials of active credentials.
    for label in (item[2:] for item in labels if item.startswith("B-")):
        for variant, template in (
            ("positive", f"private_{label.lower()} = '[[{label}]]'"),
            ("negative", f"// public lexer snapshot for {label.lower()}: '[[O:{label}]]'; not a live personal record"),
        ):
            text, entities, benign = render(template, VALUES)
            rows.append(dict(text=text, entities=entities, benign=benign, source=SOURCE,
                             language="en", supervision="full", scenario_family="private-record",
                             transport="config", split="train", variant=variant, contrast_id=f"coverage:{label}"))
    seen = set()
    for row in rows:
        validate_row(row, labels)
        if row["text"] in seen:
            raise ValueError("duplicate corpus text")
        seen.add(row["text"])
    covered = {span["label"] for row in rows if row["split"] == "train" for span in row["entities"]}
    if {"O"} | {f"{prefix}-{label}" for label in covered for prefix in ("B", "I")} != set(labels):
        raise ValueError("training corpus does not cover every frozen class")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--replicas", type=int, default=8)
    args = parser.parse_args()
    labels = load_label_map(json.loads(args.label_config.read_text()))
    rows = generate(labels, args.seed, args.replicas)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for split in ("train", "selection", "holdout"):
        with (args.output_dir / f"agent-trace.{split}.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                if row["split"] == split:
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {"source": SOURCE, "seed": args.seed, "replicas": args.replicas,
                "families": family_splits(args.seed), "counts": dict(Counter(row["split"] for row in rows)),
                "id2label": {str(i): label for i, label in enumerate(labels)},
                "label2id": {label: i for i, label in enumerate(labels)},
                "multilingual_replay": "separate known dataset; not synthetic gold",
                "supervision": "full O; never mask source", "trained_or_evaluated": False}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"counts": manifest["counts"], "bio_labels": len(labels)}))


if __name__ == "__main__":
    main()
