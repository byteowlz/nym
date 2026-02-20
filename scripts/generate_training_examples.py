#!/usr/bin/env python3
"""
Generate PII training examples from the ai4privacy dataset.

This script downloads examples from the ai4privacy/open-pii-masking-500k-ai4privacy
dataset and formats them according to the nym PII entities schema for LLM training.

Each example is output as an individual JSON file containing:
- input_text: The text to analyze
- expected_json: The expected PII entities following the nym schema

Usage:
    uv run scripts/generate_training_examples.py [--count N] [--output-dir DIR] [--languages LANGS]

Examples:
    uv run scripts/generate_training_examples.py --count 20
    uv run scripts/generate_training_examples.py --count 50 --output-dir training_data
    uv run scripts/generate_training_examples.py --languages en,de,fr --count 30
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from urllib.error import URLError

# Dataset configuration
DATASET_NAME = "ai4privacy/open-pii-masking-500k-ai4privacy"
DATASET_SPLIT = "train"
HF_API_URL = "https://datasets-server.huggingface.co/rows"

# Map dataset labels to our schema entity names
LABEL_MAPPING = {
    # Names
    "GIVENNAME": "first_name",
    "GIVENNAME1": "first_name",
    "GIVENNAME2": "first_name",
    "FIRSTNAME": "first_name",
    "LASTNAME": "last_name",
    "LASTNAME1": "last_name",
    "LASTNAME2": "last_name",
    "LASTNAME3": "last_name",
    "SURNAME": "last_name",
    "MIDDLENAME": "first_name",
    "PREFIX": "person",
    "TITLE": "person",
    # Contact
    "EMAIL": "email",
    "TEL": "phone_intl",
    "TELEPHONENUM": "phone_intl",
    "PHONE": "phone_intl",
    # Location
    "CITY": "city",
    "STATE": "state",
    "COUNTRY": "country",
    "STREET": "street_address",
    "STREETADDRESS": "street_address",
    "SECADDRESS": "street_address",
    "BUILDINGNUM": "street_address",
    "POSTCODE": "zip_code",
    "ZIPCODE": "zip_code",
    # Identity
    "SOCIALNUMBER": "ssn",
    "SOCIALNUM": "ssn",
    "SSN": "ssn",
    "IDCARD": "eu_id",
    "IDCARDNUM": "eu_id",
    "DRIVERLICENSE": "drivers_license",
    "DRIVERLICENSENUM": "drivers_license",
    "PASSPORT": "passport_us",
    "PASSPORTNUM": "passport_us",
    # Financial
    "CREDITCARDNUMBER": "credit_card",
    "CREDITCARD": "credit_card",
    "IBAN": "iban",
    "ACCOUNTNUM": "iban",
    # Date/Time
    "DATE": "date",
    "DATEOFBIRTH": "date_of_birth",
    "DOB": "date_of_birth",
    "TIME": "time",
    # Network
    "IP": "ipv4",
    "IP_ADDRESS": "ipv4",
    "IPV4": "ipv4",
    "IPV6": "ipv6",
    # Other
    "USERNAME": "username",
    "JOBAREA": "organization",
    "JOBTITLE": "organization",
    "USERAGENT": "other",
    "URL": "social_url",
    "SEX": "other",
    "GENDER": "other",
    "AGE": "other",
    "AMOUNT": "other",
    "CURRENCY": "other",
    "CURRENCYCODE": "other",
    "CURRENCYNAME": "other",
    "CURRENCYSYMBOL": "other",
    "ORDINALDIRECTION": "other",
    "NEARBYGPSCOORD": "geocoord",
    "MASKEDNUMBER": "other",
    "SECONDARYADDRESS": "street_address",
    "BUILDINGNUMBER": "street_address",
    "COUNTY": "state",
    "VEHICLEVIN": "other",
    "VEHICLEVRM": "other",
    "COMPANYNAME": "organization",
    "JOBTYPE": "other",
    "MAC": "mac",
    "LITECOINADDRESS": "other",
    "BITCOINADDRESS": "other",
    "ETHEREUMADDRESS": "other",
    "BIC": "iban",
    "IBAN_CODE": "iban",
    "PIN": "api_key",
    "PASSWORD": "api_key",
}

# Category mapping for our schema
CATEGORY_MAPPING = {
    "email": "contact",
    "phone_intl": "contact",
    "phone_us": "contact",
    "first_name": "identity",
    "last_name": "identity",
    "person": "identity",
    "ssn": "identity",
    "passport_us": "identity",
    "drivers_license": "identity",
    "eu_id": "identity",
    "date_of_birth": "identity",
    "credit_card": "financial",
    "iban": "financial",
    "street_address": "location",
    "city": "location",
    "state": "location",
    "country": "location",
    "zip_code": "location",
    "geocoord": "location",
    "ipv4": "network",
    "ipv6": "network",
    "mac": "network",
    "date": "temporal",
    "time": "temporal",
    "username": "social",
    "social_url": "social",
    "organization": "other",
    "api_key": "authentication",
    "other": "other",
}


def fetch_dataset_rows(offset: int = 0, length: int = 100) -> list[dict[str, Any]]:
    """Fetch rows from the HuggingFace dataset API."""
    url = (
        f"{HF_API_URL}?dataset={DATASET_NAME}"
        f"&config=default&split={DATASET_SPLIT}"
        f"&offset={offset}&length={length}"
    )

    try:
        with urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("rows", [])
    except URLError as e:
        print(f"Error fetching dataset: {e}", file=sys.stderr)
        return []


def parse_privacy_mask(privacy_mask: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse the privacy_mask field into our entity format."""
    entities = []

    for mask in privacy_mask:
        label = mask.get("label", "").upper()
        value = mask.get("value", "")
        start = mask.get("start", 0)
        end = mask.get("end", 0)

        # Skip empty or unknown labels
        if not label or not value:
            continue

        # Map to our entity type
        entity_type = LABEL_MAPPING.get(label, None)
        if entity_type is None:
            # Skip unmapped labels
            continue
        if entity_type == "other":
            # Skip generic "other" category
            continue

        category = CATEGORY_MAPPING.get(entity_type, "other")

        entities.append(
            {
                "type": entity_type,
                "value": value,
                "start": start,
                "end": end,
                "category": category,
                "original_label": label,
            }
        )

    return entities


def format_training_example(row: dict[str, Any]) -> dict[str, Any] | None:
    """Format a dataset row as a training example."""
    row_data = row.get("row", {})

    # Get the source text (unmasked)
    source_text = row_data.get("source_text", "")
    if not source_text:
        return None

    # Get language
    language = row_data.get("language", "unknown")

    # Parse entities from privacy_mask
    privacy_mask = row_data.get("privacy_mask", [])
    entities = parse_privacy_mask(privacy_mask)

    # Skip examples with no relevant entities
    if not entities:
        return None

    # Deduplicate entities by (type, value)
    seen = set()
    unique_entities = []
    for entity in entities:
        key = (entity["type"], entity["value"])
        if key not in seen:
            seen.add(key)
            unique_entities.append(entity)

    return {
        "text": source_text,
        "language": language,
        "entities": unique_entities,
        "entity_count": len(unique_entities),
    }


def get_diverse_examples(
    count: int = 20,
    languages: list[str] | None = None,
    fetch_limit: int = 500,
    min_length: int = 0,
) -> list[dict[str, Any]]:
    """
    Fetch diverse examples from the dataset.

    Tries to get a mix of:
    - Different languages
    - Different entity types
    - Different entity counts
    """
    print(f"Fetching {fetch_limit} rows from HuggingFace...", file=sys.stderr)

    # Fetch more rows than needed to allow for filtering and diversity
    all_rows = fetch_dataset_rows(offset=0, length=min(fetch_limit, 100))

    # If we need more, fetch additional pages
    if fetch_limit > 100:
        for offset in range(100, fetch_limit, 100):
            batch = fetch_dataset_rows(
                offset=offset, length=min(100, fetch_limit - offset)
            )
            if not batch:
                break
            all_rows.extend(batch)

    print(f"Fetched {len(all_rows)} rows", file=sys.stderr)

    # Parse all examples
    examples = []
    for row in all_rows:
        example = format_training_example(row)
        if example:
            # Filter by language if specified
            if languages and example["language"].lower() not in [
                l.lower() for l in languages
            ]:
                continue
            # Filter by minimum text length
            if len(example["text"]) < min_length:
                continue
            examples.append(example)

    print(f"Parsed {len(examples)} valid examples", file=sys.stderr)

    if len(examples) <= count:
        return examples

    # Ensure diversity by entity types
    entity_type_buckets: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        for entity in example["entities"]:
            entity_type = entity["type"]
            if entity_type not in entity_type_buckets:
                entity_type_buckets[entity_type] = []
            entity_type_buckets[entity_type].append(example)

    # Select diverse examples
    selected = []
    selected_texts = set()

    # First, try to get at least one example per entity type
    for entity_type, bucket in sorted(entity_type_buckets.items()):
        if len(selected) >= count:
            break
        random.shuffle(bucket)
        for example in bucket:
            if example["text"] not in selected_texts:
                selected.append(example)
                selected_texts.add(example["text"])
                break

    # Fill remaining slots with random diverse examples
    remaining = [e for e in examples if e["text"] not in selected_texts]
    random.shuffle(remaining)

    for example in remaining:
        if len(selected) >= count:
            break
        selected.append(example)
        selected_texts.add(example["text"])

    return selected[:count]


def format_single_example(example: dict[str, Any]) -> dict[str, Any]:
    """Format a single example for output as input_text + expected_json.

    Output format follows schemas/training-example.schema.json:
    {
        "input_text": "...",
        "expected_json": {
            "entities": [
                {"type": "email", "value": "user@example.com"},
                {"type": "phone_intl", "value": "+1-555-123-4567"}
            ]
        }
    }

    This flat structure is ideal for LLM training as models can reliably
    output entity type and value pairs without needing byte offsets.
    """
    entities = [{"type": e["type"], "value": e["value"]} for e in example["entities"]]

    return {
        "input_text": example["text"],
        "expected_json": {
            "entities": entities,
        },
    }


def write_examples_to_files(
    examples: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    """Write each example to an individual JSON file and return statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect statistics
    entity_stats: dict[str, int] = {}
    language_stats: dict[str, int] = {}

    for idx, example in enumerate(examples, start=1):
        lang = example["language"]
        language_stats[lang] = language_stats.get(lang, 0) + 1
        for entity in example["entities"]:
            etype = entity["type"]
            entity_stats[etype] = entity_stats.get(etype, 0) + 1

        # Format and write individual file
        formatted = format_single_example(example)
        filename = output_dir / f"example_{idx:04d}.json"
        filename.write_text(
            json.dumps(formatted, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return {
        "total_examples": len(examples),
        "languages": language_stats,
        "entity_types": dict(sorted(entity_stats.items(), key=lambda x: -x[1])),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate PII training examples from the ai4privacy dataset"
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=20,
        help="Number of examples to generate (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="training_examples",
        help="Output directory for individual JSON files (default: training_examples)",
    )
    parser.add_argument(
        "--languages",
        "-l",
        type=str,
        default=None,
        help="Comma-separated list of language codes to filter (e.g., en,de,fr)",
    )
    parser.add_argument(
        "--fetch-limit",
        "-f",
        type=int,
        default=500,
        help="Number of rows to fetch for diversity (default: 500)",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--min-length",
        "-m",
        type=int,
        default=0,
        help="Minimum character length for input_text (default: 0)",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    languages = None
    if args.languages:
        languages = [l.strip() for l in args.languages.split(",")]

    # Fetch and format examples
    examples = get_diverse_examples(
        count=args.count,
        languages=languages,
        fetch_limit=args.fetch_limit,
        min_length=args.min_length,
    )

    if not examples:
        print("Error: No examples found matching criteria", file=sys.stderr)
        sys.exit(1)

    # Write individual files
    output_dir = Path(args.output_dir)
    stats = write_examples_to_files(examples, output_dir)

    # Print summary to stderr
    print(
        f"\nWrote {stats['total_examples']} examples to {output_dir}/", file=sys.stderr
    )
    print(f"  Languages: {stats['languages']}", file=sys.stderr)
    print(f"  Entity types: {len(stats['entity_types'])}", file=sys.stderr)


if __name__ == "__main__":
    main()
