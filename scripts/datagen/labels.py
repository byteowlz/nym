"""Label vocabulary + Faker value generators for PII training-data synthesis.

The label names match nym's token-classification backend mapping
(`src/engine/ner_token.rs::label_to_pattern`), so a model trained on this data
integrates cleanly (nice placeholders / fake replacement) rather than falling
back to the generic `ner_entity`.

Each generator takes a `Faker` instance and returns a realistic, format-correct,
locale-aware value. Providers that don't exist for a given locale fall back to a
sensible default so generation never crashes mid-run.
"""
from __future__ import annotations

import random
import secrets
import string
from typing import Callable, Dict

from faker import Faker


def _digits(n: int) -> str:
    return "".join(random.choices(string.digits, k=n))


def _safe(fn: Callable[[], str], fallback: Callable[[], str]) -> str:
    try:
        v = fn()
        if v is None:
            raise ValueError("no value")
        return str(v)
    except Exception:
        return fallback()


# label -> generator(faker) -> value
GENERATORS: Dict[str, Callable[[Faker], str]] = {
    # Names / identity
    "GIVEN_NAME": lambda f: f.first_name(),
    "SURNAME": lambda f: f.last_name(),
    "USERNAME": lambda f: f.user_name(),
    "AGE": lambda f: str(random.randint(1, 99)),
    "GENDER": lambda f: random.choice(["male", "female", "non-binary", "M", "F"]),
    "DATE_OF_BIRTH": lambda f: f.date_of_birth(minimum_age=0, maximum_age=99).strftime("%m/%d/%Y"),
    # Government / record identifiers
    "SSN": lambda f: _safe(f.ssn, lambda: f"{_digits(3)}-{_digits(2)}-{_digits(4)}"),
    "TAX_ID": lambda f: f"{_digits(2)}-{_digits(7)}",
    "MEDICAL_RECORD_NUMBER": lambda f: "MRN" + _digits(random.randint(6, 9)),
    "PASSPORT": lambda f: _safe(getattr(f, "passport_number", None) or (lambda: None),
                                lambda: random.choice(string.ascii_uppercase) + _digits(8)),
    "DRIVERS_LICENSE": lambda f: random.choice(string.ascii_uppercase) + _digits(random.randint(6, 8)),
    "GOVERNMENT_ID": lambda f: _digits(random.randint(8, 11)),
    "ACCOUNT_NUMBER": lambda f: _safe(f.bban, lambda: _digits(12)),
    "CUSTOMER_ID": lambda f: "CID-" + _digits(6),
    "EMPLOYEE_ID": lambda f: "EMP-" + _digits(5),
    # Contact
    "EMAIL": lambda f: f.email(),
    "PHONE": lambda f: f.phone_number(),
    "FAX_NUMBER": lambda f: f.phone_number(),
    # Location
    "STREET_NAME": lambda f: f.street_name(),
    "BUILDING_NUMBER": lambda f: _safe(f.building_number, lambda: str(random.randint(1, 9999))),
    "STREET_ADDRESS": lambda f: f.street_address().replace("\n", " "),
    "SECONDARY_ADDRESS": lambda f: _safe(f.secondary_address, lambda: f"Apt {random.randint(1, 999)}"),
    "CITY": lambda f: f.city(),
    "STATE": lambda f: _safe(f.state, f.city),
    "COUNTRY": lambda f: f.country(),
    "ZIP_CODE": lambda f: _safe(f.postcode, lambda: _digits(5)),
    # Financial
    "CREDIT_DEBIT_CARD": lambda f: f.credit_card_number(),
    "CVV": lambda f: _safe(f.credit_card_security_code, lambda: _digits(3)),
    "PIN": lambda f: _digits(random.choice([4, 6])),
    "ROUTING_NUMBER": lambda f: _safe(f.aba, lambda: _digits(9)),
    "SWIFT_BIC": lambda f: _safe(f.swift, lambda: "".join(random.choices(string.ascii_uppercase, k=8))),
    "IBAN": lambda f: _safe(f.iban, lambda: "DE" + _digits(20)),
    # Network / technical
    "IPV4": lambda f: f.ipv4(),
    "IPV6": lambda f: f.ipv6(),
    "MAC_ADDRESS": lambda f: f.mac_address(),
    "URL": lambda f: f.url(),
    # Credentials
    "API_KEY": lambda f: "sk-" + secrets.token_hex(16),
    "PASSWORD": lambda f: f.password(length=random.randint(8, 16)),
    # Organization / temporal / vehicle
    "COMPANY_NAME": lambda f: f.company(),
    "DATE": lambda f: f.date(pattern="%m/%d/%Y"),
    "TIME": lambda f: f.time(pattern="%I:%M %p"),
    "LICENSE_PLATE": lambda f: _safe(f.license_plate, lambda: "".join(random.choices(string.ascii_uppercase, k=3)) + _digits(4)),
}

ALLOWED_LABELS = sorted(GENERATORS.keys())


def generate_value(label: str, faker: Faker) -> str:
    gen = GENERATORS.get(label.upper())
    if gen is None:
        # Unknown label: emit a generic token so the pipeline is robust.
        return "".join(random.choices(string.ascii_letters, k=6))
    return gen(faker)
