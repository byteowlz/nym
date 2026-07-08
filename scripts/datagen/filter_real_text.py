#!/usr/bin/env python3
"""Filter teacher-labeled real text down to a high-precision training subset.

Conservative rule: if ANY span in a passage fails its gate, the WHOLE passage is
dropped. We never strip a span and keep the passage — that would teach the
student "this text is O" when the teacher merely disagreed with a validator.
Purity beats volume: the synthetic corpus supplies volume; this corpus supplies
trustworthy real-prose signal (mostly names/places/dates + clean negatives).

Gates:
  1. confidence: contextual types >= --conf (default 0.85),
     structured types >= --conf-structured (default 0.95)
  2. shape validation for structured types (a teacher "SSN" on Wikipedia that
     has no digits is a hallucination -> passage dropped)

Usage:
  python filter_real_text.py data/real/real.*.jsonl -o data/real-filtered.jsonl
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

# ---- shape validators for structured types ------------------------------
LUHN_STRIP = re.compile(r"[ -]")


def luhn_ok(s):
    digits = [int(c) for c in LUHN_STRIP.sub("", s) if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def ndigits(s):
    return sum(c.isdigit() for c in s)


VALIDATORS = {
    "EMAIL": lambda s: re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", s.strip()) is not None,
    "URL": lambda s: ("://" in s or s.strip().lower().startswith("www.")
                      or re.search(r"\.[a-z]{2,6}(/|$)", s.strip().lower()) is not None),
    "IP_ADDRESS": lambda s: re.fullmatch(r"(\d{1,3}\.){3}\d{1,3}", s.strip()) is not None
                            or ":" in s and re.fullmatch(r"[0-9a-fA-F:]{3,45}", s.strip()) is not None,
    "MAC_ADDRESS": lambda s: re.fullmatch(r"([0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}", s.strip()) is not None,
    "IBAN": lambda s: re.fullmatch(r"[A-Z]{2}\d{2}[ ]?([A-Z0-9][ ]?){10,32}", s.strip()) is not None,
    "SWIFT_BIC": lambda s: re.fullmatch(r"[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?", s.strip()) is not None,
    "CREDIT_DEBIT_CARD": luhn_ok,
    "PHONE": lambda s: ndigits(s) >= 7,
    "FAX_NUMBER": lambda s: ndigits(s) >= 7,
    "SSN": lambda s: ndigits(s) >= 6,
    "TAX_ID": lambda s: ndigits(s) >= 5,
    "ROUTING_NUMBER": lambda s: ndigits(s) >= 8,
    "ACCOUNT_NUMBER": lambda s: ndigits(s) >= 5,
    "MEDICAL_RECORD_NUMBER": lambda s: ndigits(s) >= 4,
    "EMPLOYEE_ID": lambda s: ndigits(s) >= 2,
    "CUSTOMER_ID": lambda s: ndigits(s) >= 2,
    "GOVERNMENT_ID": lambda s: ndigits(s) >= 4,
    "PASSPORT": lambda s: ndigits(s) >= 4,
    "DRIVERS_LICENSE": lambda s: ndigits(s) >= 3,
    "LICENSE_PLATE": lambda s: ndigits(s) >= 1 and len(s.strip()) >= 4,
    "ZIP_CODE": lambda s: ndigits(s) >= 3 and len(s.strip()) <= 12,
    "PIN": lambda s: s.strip().isdigit() and 3 <= len(s.strip()) <= 8,
    "CVV": lambda s: s.strip().isdigit() and 3 <= len(s.strip()) <= 4,
    "API_KEY": lambda s: len(s.strip()) >= 16 and ndigits(s) >= 2 and " " not in s.strip(),
    "PASSWORD": lambda s: len(s.strip()) >= 6 and " " not in s.strip(),
}
STRUCTURED = set(VALIDATORS)
# contextual types: conf gate only
CONTEXTUAL = {"GIVEN_NAME", "SURNAME", "CITY", "STATE", "COUNTRY", "COMPANY_NAME",
              "DATE", "TIME", "AGE", "GENDER", "DATE_OF_BIRTH", "STREET_ADDRESS",
              "STREET_NAME", "BUILDING_NUMBER", "SECONDARY_ADDRESS", "USERNAME"}
# USERNAME is contextual-shaped but hallucination-prone on encyclopedic text:
HIGH_BAR = {"USERNAME", "PASSWORD", "API_KEY"}


# Bibliographic identifiers are the classic Wikipedia false-positive for ID-type
# labels (ISBN/ISSN/DOI look exactly like SSNs/account numbers, teacher conf is
# high). Reject any structured span that is ISBN-shaped or directly preceded by
# a bibliographic prefix.
BIBLIO_PREFIX = re.compile(r"(ISBN|ISSN|e-?ISSN|DOI|OCLC|LCCN|PMID|arXiv)[:\s]*$", re.I)
ISBN_SHAPE = re.compile(r"^97[89][ -]?(\d[ -]?){9}\d$|^(\d[ -]?){9}[\dXx]$")


def biblio_fp(text, start, span):
    return (ISBN_SHAPE.match(span.strip()) is not None
            or BIBLIO_PREFIX.search(text[max(0, start - 12):start]) is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.85)
    ap.add_argument("--conf-structured", type=float, default=0.95)
    ap.add_argument("--conf-high-bar", type=float, default=0.98)
    args = ap.parse_args()

    stats = collections.Counter()
    drop_reasons = collections.Counter()
    kept_labels = collections.Counter()
    with args.out.open("w") as out:
        for path in args.inputs:
            for line in open(path):
                r = json.loads(line)
                if r.get("__done__"):
                    continue
                stats["read"] += 1
                ok = True
                for e in r["entities"]:
                    lab, conf = e["label"], e.get("conf", 0.0)
                    val = r["text"][e["start"]:e["end"]]
                    if lab in HIGH_BAR:
                        gate = args.conf_high_bar
                    elif lab in STRUCTURED:
                        gate = args.conf_structured
                    else:
                        gate = args.conf
                    if conf < gate:
                        drop_reasons[f"lowconf:{lab}"] += 1
                        ok = False
                        break
                    v = VALIDATORS.get(lab)
                    if v is not None and not v(val):
                        drop_reasons[f"shape:{lab}"] += 1
                        ok = False
                        break
                    if lab in STRUCTURED and biblio_fp(r["text"], e["start"], val):
                        drop_reasons[f"biblio:{lab}"] += 1
                        ok = False
                        break
                if not ok:
                    stats["dropped"] += 1
                    continue
                for e in r["entities"]:
                    e.pop("conf", None)
                    kept_labels[e["label"]] += 1
                stats["kept"] += 1
                if r["entities"]:
                    stats["kept_pos"] += 1
                out.write(json.dumps(r, ensure_ascii=False) + "\n")

    k, rd = stats["kept"], stats["read"]
    sys.stderr.write(f"read {rd:,} -> kept {k:,} ({k*100//max(rd,1)}%), "
                     f"{stats['kept_pos']:,} with PII\n")
    sys.stderr.write("top drop reasons: "
                     f"{drop_reasons.most_common(10)}\n")
    sys.stderr.write(f"kept labels: {kept_labels.most_common(12)}\n")


if __name__ == "__main__":
    main()
