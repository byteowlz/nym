"""Complete occurrence-level NER gold data; never turn uncertainty into O.

Character offsets are Unicode codepoints. This schema is distinct from vocabulary
approvals: a corpus-wide literal decision is not an occurrence annotation.
"""
import hashlib
import json
import re
from pathlib import Path
from json_safe import unique_object as unique_keys

SCHEMA = "nym.ner.gold.v1"
BUNDLE_SCHEMA = "nym.ner.gold-bundle.v1"
SPLITS = ("train", "selection", "holdout")


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _span(span, length, labelled):
    keys = {"start", "end", "label"} if labelled else {"start", "end"}
    if not isinstance(span, dict) or set(span) != keys:
        raise ValueError("invalid gold span fields")
    a, b = span["start"], span["end"]
    if type(a) is not int or type(b) is not int or not 0 <= a < b <= length:
        raise ValueError("invalid gold codepoint range")
    if labelled and (not isinstance(span["label"], str)
                     or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", span["label"])
                     or span["label"].upper() in {"O", "UNCERTAIN"}):
        raise ValueError("invalid gold entity type; uncertainty requires masked_spans")
    return a, b


def validate_row(row, allow_incomplete=False):
    required = {"schema", "source_id", "unit_id", "text", "text_sha256",
                "annotation_complete", "entities", "masked_spans"}
    if not isinstance(row, dict) or set(row) != required or row["schema"] != SCHEMA:
        raise ValueError("invalid complete-gold schema")
    for key in ("source_id", "unit_id"):
        if not isinstance(row[key], str) or not row[key] or len(row[key]) > 256:
            raise ValueError("gold requires bounded source and unit identities")
    if not isinstance(row["text"], str) or len(row["text"].encode("utf-8")) > 1024 * 1024:
        raise ValueError("invalid or oversized gold text")
    if row["text_sha256"] != digest(row["text"]):
        raise ValueError("gold original-text checksum mismatch")
    if type(row["annotation_complete"]) is not bool:
        raise ValueError("gold completeness must be explicit boolean")
    if not allow_incomplete and not row["annotation_complete"]:
        raise ValueError("incomplete annotations cannot supervise training")
    spans = []
    for key, labelled in (("entities", True), ("masked_spans", False)):
        if not isinstance(row[key], list) or len(row[key]) > 1000:
            raise ValueError("invalid gold span collection")
        spans.extend(_span(span, len(row["text"]), labelled) for span in row[key])
    spans.sort()
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("overlapping known or unknown gold regions")
    return row


def validate_partitions(parts, require_training_mix=True):
    if set(parts) != set(SPLITS) or any(not isinstance(parts[s], list) or not parts[s] for s in SPLITS):
        raise ValueError("nonempty train, selection and holdout partitions required")
    source_split, text_split, identities = {}, {}, set()
    for split in SPLITS:
        for row in parts[split]:
            validate_row(row)
            identity = (row["source_id"], row["unit_id"])
            if identity in identities:
                raise ValueError("duplicate gold unit identity")
            identities.add(identity)
            source = row["source_id"]
            if source in source_split and source_split[source] != split:
                raise ValueError("gold source leaked across partitions")
            source_split[source] = split
            text_hash = row["text_sha256"]
            if text_hash in text_split and text_split[text_hash] != split:
                raise ValueError("exact gold text leaked across partitions")
            text_split[text_hash] = split
    if require_training_mix:
        if not any(row["entities"] for row in parts["train"]):
            raise ValueError("training requires completely annotated sensitive positives")
        if not any(not row["entities"] and not row["masked_spans"] for row in parts["train"]):
            raise ValueError("training requires completely reviewed negative examples")
    return parts


def prepare(rows, source_splits, label_types):
    if not isinstance(source_splits, dict) or any(v not in SPLITS for v in source_splits.values()):
        raise ValueError("invalid frozen source split map")
    if not isinstance(label_types, list) or not label_types or len(set(label_types)) != len(label_types):
        raise ValueError("explicit unique taxonomy required")
    for label in label_types:
        _span({"start": 0, "end": 1, "label": label}, 1, True)
    parts = {s: [] for s in SPLITS}
    excluded = 0
    seen = set()
    for row in rows:
        validate_row(row, allow_incomplete=True)
        identity = (row["source_id"], row["unit_id"])
        if identity in seen:
            raise ValueError("duplicate input gold unit")
        seen.add(identity)
        if row["source_id"] not in source_splits:
            raise ValueError("gold source missing from frozen split map")
        if any(e["label"] not in label_types for e in row["entities"]):
            raise ValueError("gold entity missing from declared taxonomy")
        if not row["annotation_complete"]:
            excluded += 1
            continue
        parts[source_splits[row["source_id"]]].append(row)
    validate_partitions(parts)
    payload = {"schema": BUNDLE_SCHEMA, "label_types": label_types,
               "source_splits": source_splits, "excluded_incomplete": excluded, **parts}
    payload["payload_sha256"] = bundle_digest(payload)
    return payload


def bundle_digest(bundle):
    value = {k: v for k, v in bundle.items() if k != "payload_sha256"}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def load_bundle(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(128 * 1024 * 1024 + 1)
    if len(raw) > 128 * 1024 * 1024:
        raise ValueError("gold bundle limit exceeded")
    bundle = json.loads(raw, object_pairs_hook=unique_keys)
    keys = {"schema", "label_types", "source_splits", "excluded_incomplete",
            "payload_sha256", *SPLITS}
    if not isinstance(bundle, dict) or set(bundle) != keys or bundle["schema"] != BUNDLE_SCHEMA:
        raise ValueError("invalid gold bundle schema")
    if bundle["payload_sha256"] != bundle_digest(bundle):
        raise ValueError("gold bundle checksum mismatch")
    rebuilt = prepare([row for s in SPLITS for row in bundle[s]],
                      bundle["source_splits"], bundle["label_types"])
    if any(rebuilt[s] != bundle[s] for s in SPLITS):
        raise ValueError("gold split assignment mismatch")
    if type(bundle["excluded_incomplete"]) is not int or bundle["excluded_incomplete"] < 0:
        raise ValueError("invalid incomplete exclusion count")
    return bundle


def label_offset(offset, row, label2id, weak=False):
    """A token intersecting any unknown region contributes no hard-label loss."""
    a, b = offset
    if a == b or any(a < m["end"] and b > m["start"] for m in row.get("masked_spans", [])):
        return -100
    entities = [e for e in row["entities"] if a < e["end"] and b > e["start"]]
    if not entities:
        return -100 if weak else label2id["O"]
    if row.get("schema") == SCHEMA and (len(entities) != 1
            or a < entities[0]["start"] or b > entities[0]["end"]):
        return -100
    entity = min(entities, key=lambda e: e["start"])
    tag = ("B-" if a <= entity["start"] else "I-") + entity["label"]
    return label2id[tag]


def align_record(row, tokenizer, label2id, max_length, weak=False):
    if "annotation_complete" in row and row["annotation_complete"] is not True:
        raise ValueError("incomplete annotations cannot supervise training")
    gold = row.get("schema") == SCHEMA
    if gold:
        validate_row(row)
    for mask in row.get("masked_spans", []):
        _span(mask, len(row["text"]), False)
    encoded = tokenizer(row["text"], truncation=not gold, max_length=max_length,
                        return_offsets_mapping=True)
    if gold and len(encoded["input_ids"]) > max_length:
        raise ValueError("gold unit exceeds token limit; split and re-annotate rather than truncate")
    encoded["labels"] = [label_offset(offset, row, label2id, weak)
                         for offset in encoded.pop("offset_mapping")]
    return encoded


def supervised_selection(encoded):
    result = [row for row in encoded if any(label != -100 for label in row["labels"])]
    if not result:
        raise ValueError("gold selection has no evaluable supervised tokens")
    return result


def supervised_training(encoded, o_id):
    result = supervised_selection(encoded)
    if not any(label not in (-100, o_id) for row in result for label in row["labels"]):
        raise ValueError("gold optimizer has no supervised sensitive tokens")
    return result
