"""Label vocabulary + Faker value generators for PII training-data synthesis.

The label names match nym's token-classification backend mapping
(`src/engine/ner_token.rs::label_to_pattern`), so a model trained on this data
integrates cleanly (nice placeholders / fake replacement) rather than falling
back to the generic `ner_entity`.

Every value is produced through `_try`, which resolves the Faker provider by name
inside a try/except. Providers vary across the ~23 locales in the rubric (e.g.
`iban`, `state`, `secondary_address`, `passport_number` don't exist everywhere);
`_try` falls back to a synthetic value so generation never crashes mid-run.
"""
from __future__ import annotations

import random
import secrets
import string
from typing import Callable, Dict

from faker import Faker


def _digits(n: int) -> str:
    return "".join(random.choices(string.digits, k=n))


def _try(faker: Faker, provider: str, fallback: Callable[[], str], **kwargs) -> str:
    """Call `faker.<provider>(**kwargs)`, falling back if the provider is missing
    for this locale, errors, or returns None/empty."""
    try:
        fn = getattr(faker, provider)
        v = fn(**kwargs)
        if v is None or (isinstance(v, str) and not v.strip()):
            raise ValueError("empty")
        return str(v).replace("\n", " ")
    except Exception:
        return fallback()


def _passport(f: Faker) -> str:
    return _try(f, "passport_number",
                lambda: random.choice(string.ascii_uppercase) + _digits(8))


# label -> generator(faker) -> value
GENERATORS: Dict[str, Callable[[Faker], str]] = {
    # Names / identity (person providers exist for every locale)
    "GIVEN_NAME": lambda f: _try(f, "first_name", lambda: "Alex"),
    "SURNAME": lambda f: _try(f, "last_name", lambda: "Smith"),
    "USERNAME": lambda f: _try(f, "user_name", lambda: "user" + _digits(3)),
    "AGE": lambda f: str(random.randint(1, 99)),
    "GENDER": lambda f: random.choice(["male", "female", "non-binary", "M", "F"]),
    "DATE_OF_BIRTH": lambda f: _try(f, "date_of_birth", lambda: "01/01/1980"),
    # Government / record identifiers
    "SSN": lambda f: _try(f, "ssn", lambda: f"{_digits(3)}-{_digits(2)}-{_digits(4)}"),
    "TAX_ID": lambda f: f"{_digits(2)}-{_digits(7)}",
    "MEDICAL_RECORD_NUMBER": lambda f: "MRN" + _digits(random.randint(6, 9)),
    "PASSPORT": _passport,
    "DRIVERS_LICENSE": lambda f: random.choice(string.ascii_uppercase) + _digits(random.randint(6, 8)),
    "GOVERNMENT_ID": lambda f: _digits(random.randint(8, 11)),
    "ACCOUNT_NUMBER": lambda f: _try(f, "bban", lambda: _digits(12)),
    "CUSTOMER_ID": lambda f: "CID-" + _digits(6),
    "EMPLOYEE_ID": lambda f: "EMP-" + _digits(5),
    # Contact
    "EMAIL": lambda f: _try(f, "email", lambda: "user@example.com"),
    "PHONE": lambda f: _try(f, "phone_number", lambda: "+1-555-" + _digits(4)),
    "FAX_NUMBER": lambda f: _try(f, "phone_number", lambda: "+1-555-" + _digits(4)),
    # Location
    "STREET_NAME": lambda f: _try(f, "street_name", lambda: "Main Street"),
    "BUILDING_NUMBER": lambda f: _try(f, "building_number", lambda: str(random.randint(1, 9999))),
    "STREET_ADDRESS": lambda f: _try(f, "street_address", lambda: str(random.randint(1, 999)) + " Main St"),
    "SECONDARY_ADDRESS": lambda f: _try(f, "secondary_address", lambda: f"Apt {random.randint(1, 999)}"),
    "CITY": lambda f: _try(f, "city", lambda: "Springfield"),
    "STATE": lambda f: _try(f, "state", lambda: _try(f, "city", lambda: "Springfield")),
    "COUNTRY": lambda f: _try(f, "country", lambda: "Nowhere"),
    "ZIP_CODE": lambda f: _try(f, "postcode", lambda: _digits(5)),
    # Financial
    "CREDIT_DEBIT_CARD": lambda f: _try(f, "credit_card_number", lambda: _digits(16)),
    "CVV": lambda f: _try(f, "credit_card_security_code", lambda: _digits(3)),
    "PIN": lambda f: _digits(random.choice([4, 6])),
    "ROUTING_NUMBER": lambda f: _try(f, "aba", lambda: _digits(9)),
    "SWIFT_BIC": lambda f: _try(f, "swift", lambda: "".join(random.choices(string.ascii_uppercase, k=8))),
    "IBAN": lambda f: _try(f, "iban", lambda: "DE" + _digits(20)),
    # Network / technical
    "IPV4": lambda f: _try(f, "ipv4", lambda: ".".join(_digits(2) for _ in range(4))),
    "IPV6": lambda f: _try(f, "ipv6", lambda: "2001:db8::" + _digits(3)),
    "MAC_ADDRESS": lambda f: _try(f, "mac_address", lambda: ":".join("%02x" % random.randint(0, 255) for _ in range(6))),
    "URL": lambda f: _try(f, "url", lambda: "https://example.com"),
    # Credentials
    "API_KEY": lambda f: "sk-" + secrets.token_hex(16),
    "PASSWORD": lambda f: _try(f, "password", lambda: secrets.token_urlsafe(10)),
    # Organization / temporal / vehicle
    "COMPANY_NAME": lambda f: _try(f, "company", lambda: "Acme Inc"),
    "DATE": lambda f: _try(f, "date", lambda: "01/01/2020", pattern="%m/%d/%Y"),
    "TIME": lambda f: _try(f, "time", lambda: "12:00 PM", pattern="%I:%M %p"),
    "LICENSE_PLATE": lambda f: _try(f, "license_plate",
                                    lambda: "".join(random.choices(string.ascii_uppercase, k=3)) + _digits(4)),
}

ALLOWED_LABELS = sorted(GENERATORS.keys())


def generate_value(label: str, faker: Faker) -> str:
    gen = GENERATORS.get(label.upper())
    if gen is None:
        return "".join(random.choices(string.ascii_letters, k=6))
    return gen(faker)
