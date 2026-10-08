#!/usr/bin/env python3
"""Validate complete NER annotations and publish one private source-split bundle.

uv run --no-project python scripts/prepare_ner_gold.py --input gold.jsonl \
  --source-splits splits.json --label-types labels.json --output gold-bundle.json

No inference, training, automatic labels, or model activation is performed.
"""
import argparse
import json
import sys
from pathlib import Path

from gold_ner import prepare, unique_keys
from json_safe import publish


def read_json(path):
    with path.open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("gold metadata size limit exceeded")
    return json.loads(raw, object_pairs_hook=unique_keys)


def read_rows(paths):
    rows = []
    total_bytes = 0
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            while line := stream.readline(1024 * 1024 + 1):
                size = len(line.encode("utf-8"))
                total_bytes += size
                if size > 1024 * 1024 or total_bytes > 128 * 1024 * 1024:
                    raise ValueError("gold input size limit exceeded")
                if line.strip():
                    rows.append(json.loads(line, object_pairs_hook=unique_keys))
                if len(rows) > 100_000:
                    raise ValueError("gold row limit exceeded")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--source-splits", type=Path, required=True)
    parser.add_argument("--label-types", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        bundle = prepare(read_rows(args.input), read_json(args.source_splits), read_json(args.label_types))
        publish(args.output, bundle, args.force)
    except (ValueError, OSError, TypeError, UnicodeError):
        # Parser payloads, paths and private values are never reflected to stderr.
        sys.stderr.write("error: gold preparation failed; validate complete annotations, source splits, taxonomy and destination\n")
        return 1
    print(json.dumps({"status": "prepared_not_trained", "excluded_incomplete": bundle["excluded_incomplete"],
                      "counts": {s: len(bundle[s]) for s in ("train", "selection", "holdout")},
                      "payload_sha256": bundle["payload_sha256"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
