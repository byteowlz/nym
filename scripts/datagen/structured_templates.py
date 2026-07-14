#!/usr/bin/env python3
"""Generate structured-format PII templates (JSON, CSV, key-value, XML, table, EDI).

The gemma-distilled student over-flags on structured documents — it labels JSON
keys, delimiters, and field names as PII because it only trained on prose. These
templates put PII *values* inside realistic structured scaffolding with the keys/
delimiters as literal (O) text, so the model learns "the key `email` is O, the
value is EMAIL". Exact labels by construction (same [LABEL] placeholder scheme as
the rest of datagen); fills via generate.py --templates-file.

Usage:
  python structured_templates.py --n 500 -o data/templates.structured.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from labels import ALLOWED_LABELS  # noqa: E402

# realistic field-key names per PII label (keys stay literal O text)
KEYS = {
    "GIVEN_NAME": ["first_name", "firstName", "given_name", "fname", "vorname", "prenom", "nombre"],
    "SURNAME": ["last_name", "lastName", "surname", "family_name", "lname", "nachname", "apellido"],
    "EMAIL": ["email", "e_mail", "email_address", "mail", "contact_email"],
    "PHONE": ["phone", "phone_number", "tel", "telephone", "mobile", "contact_phone"],
    "FAX_NUMBER": ["fax", "fax_number"],
    "CITY": ["city", "town", "ville", "stadt", "ciudad", "locality"],
    "STATE": ["state", "province", "region"],
    "COUNTRY": ["country", "nation", "pays"],
    "ZIP_CODE": ["zip", "zip_code", "postcode", "postal_code", "plz"],
    "STREET_ADDRESS": ["address", "street_address", "addr", "adresse"],
    "STREET_NAME": ["street", "street_name"],
    "BUILDING_NUMBER": ["building", "house_number", "no"],
    "SECONDARY_ADDRESS": ["address2", "apt", "suite", "unit"],
    "DATE_OF_BIRTH": ["dob", "date_of_birth", "birth_date", "birthdate"],
    "DATE": ["date", "created", "updated", "timestamp", "issued"],
    "TIME": ["time", "created_at"],
    "AGE": ["age"],
    "GENDER": ["gender", "sex"],
    "SSN": ["ssn", "social_security", "national_id"],
    "TAX_ID": ["tax_id", "vat", "vat_number", "tin"],
    "PASSPORT": ["passport", "passport_no", "passport_number"],
    "DRIVERS_LICENSE": ["drivers_license", "license_no", "dl"],
    "GOVERNMENT_ID": ["gov_id", "id_number", "identity"],
    "IBAN": ["iban", "account_iban"],
    "SWIFT_BIC": ["swift", "bic", "swift_bic"],
    "ACCOUNT_NUMBER": ["account", "account_number", "acct", "acct_no"],
    "ROUTING_NUMBER": ["routing", "routing_number", "aba"],
    "CREDIT_DEBIT_CARD": ["card", "card_number", "cc", "pan"],
    "CVV": ["cvv", "cvc", "security_code"],
    "PIN": ["pin"],
    "CUSTOMER_ID": ["customer_id", "cust_id", "client_id", "customerNumber"],
    "EMPLOYEE_ID": ["employee_id", "emp_id", "staff_id", "badge"],
    "MEDICAL_RECORD_NUMBER": ["mrn", "record_number", "patient_id"],
    "COMPANY_NAME": ["company", "organization", "employer", "vendor", "firm"],
    "USERNAME": ["username", "user", "login", "handle", "user_id"],
    "PASSWORD": ["password", "passwd", "pwd"],
    "API_KEY": ["api_key", "apikey", "token", "secret"],
    "IPV4": ["ip", "ip_address", "ipv4"],
    "IPV6": ["ipv6"],
    "MAC_ADDRESS": ["mac", "mac_address"],
    "URL": ["url", "website", "link", "homepage"],
    "LICENSE_PLATE": ["plate", "license_plate", "reg"],
}
FIELD_LABELS = [l for l in KEYS if l in ALLOWED_LABELS]


def ph(label):
    return f"[{label}]"


def key(label, rng):
    return rng.choice(KEYS[label])


def json_flat(fields, rng):
    parts = [f'"{key(l, rng)}": "{ph(l)}"' for l in fields]
    return "{" + ", ".join(parts) + "}"


def json_nested(fields, rng):
    # split fields into a couple of nested groups
    rng.shuffle(fields)
    mid = max(1, len(fields) // 2)
    g1, g2 = fields[:mid], fields[mid:]
    def obj(fs):
        return "{" + ", ".join(f'"{key(l, rng)}": "{ph(l)}"' for l in fs) + "}"
    grp = rng.choice(["person", "customer", "client", "record", "user"])
    sub = rng.choice(["contact", "details", "address", "info"])
    return '{"%s": {"id": "%s", "%s": %s}}' % (grp, ph(rng.choice([f for f in fields])), sub, obj(g2)) \
        if g2 else obj(g1)


def csv_rows(fields, rng):
    header = ",".join(key(l, rng) for l in fields)
    row = ",".join(ph(l) for l in fields)
    return f"{header}\n{row}"


def kv_lines(fields, rng):
    sep = rng.choice([": ", ": ", " = ", ":\t"])
    return "\n".join(f"{key(l, rng)}{sep}{ph(l)}" for l in fields)


def xml_block(fields, rng):
    root = rng.choice(["record", "person", "customer", "entry"])
    inner = "".join(f"<{key(l, rng)}>{ph(l)}</{key(l, rng)}>" for l in fields)
    return f"<{root}>{inner}</{root}>"


def table_md(fields, rng):
    header = "| " + " | ".join(key(l, rng) for l in fields) + " |"
    sep = "|" + "|".join("---" for _ in fields) + "|"
    row = "| " + " | ".join(ph(l) for l in fields) + " |"
    return f"{header}\n{sep}\n{row}"


def yaml_block(fields, rng):
    return "\n".join(f"{key(l, rng)}: {ph(l)}" for l in fields)


def edi_like(fields, rng):
    seg = rng.choice(["NAD", "COM", "DTM", "RFF"])
    return seg + "+" + "+".join(ph(l) for l in fields) + "'"


STRUCTS = [json_flat, json_nested, csv_rows, kv_lines, xml_block, table_md, yaml_block, edi_like]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", type=Path, default=Path("data/templates.structured.jsonl"))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    # locale pool so Faker fills with varied values; keys stay English-ish (realistic for structured data)
    locales = ["en_US", "de_DE", "fr_FR", "es_ES", "it_IT", "ru_RU", "ja_JP", "zh_CN",
               "ko_KR", "ar_AA", "pt_BR", "nl_NL", "tr_TR", "pl_PL", "hi_IN"]
    seen, out = set(), []
    guard = 0
    while len(out) < args.n and guard < args.n * 20:
        guard += 1
        struct = rng.choice(STRUCTS)
        nf = rng.randint(2, 7)
        fields = rng.sample(FIELD_LABELS, min(nf, len(FIELD_LABELS)))
        tmpl = struct(list(fields), rng)
        if tmpl in seen:
            continue
        seen.add(tmpl)
        out.append({"template": tmpl, "locale": rng.choice(locales)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    sys.stderr.write(f"wrote {len(out)} structured templates -> {args.out}\n")
    # show a few
    for o in out[:3]:
        sys.stderr.write("  " + o["template"][:90].replace("\n", "\\n") + "\n")


if __name__ == "__main__":
    main()
