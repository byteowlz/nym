#!/usr/bin/env python3
"""Build a license-clean real-name corpus for datagen name fills.

Sources (all permissive, no scraped/leaked data):
  - US Census 2010 surnames  — PUBLIC DOMAIN (US gov work). ~160k surnames incl.
    the rare long tail Faker lacks. Latin script, frequency-weighted.
  - Wikidata given/family names — CC0 1.0 (public-domain dedication). Multilingual,
    native scripts (Arabic/CJK/Cyrillic/Devanagari/Greek/...). No attribution required.

Every record carries source + license for provenance. Output feeds generate.py's
name fills as a Faker replacement (--name-corpus). We sample name STRINGS to fill
SYNTHETIC templates — no real person's record is stored.

Usage:
  python name_corpus.py --out data/names --per-lang 3000
"""
import argparse
import csv
import io
import json
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

UA = "nym-namecorpus/1.0 (https://github.com/byteowlz/nym; research; PII detector training)"
CENSUS_URL = "https://www2.census.gov/topics/genealogy/2010surnames/names.zip"
WDQS = "https://query.wikidata.org/sparql"
# label languages -> our locale-ish tag; native scripts first (the point), then Latin tail
WD_LANGS = ["ar", "zh", "ja", "ko", "ru", "hi", "el", "uk", "he", "fa",
            "en", "de", "fr", "es", "it", "pt", "nl", "pl", "sv", "cs", "ro", "tr", "fi", "da"]
# Q202444 = given name, Q101352 = family name
WD_KINDS = {"given": "Q202444", "family": "Q101352"}


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=timeout).read()


def census_surnames(min_count):
    """US Census 2010 surnames (public domain). Yields (name, count)."""
    raw = fetch(CENSUS_URL, timeout=120)
    zf = zipfile.ZipFile(io.BytesIO(raw))
    fn = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
    text = zf.read(fn).decode("latin-1")
    rows = csv.DictReader(io.StringIO(text))
    for r in rows:
        name = (r.get("name") or "").strip()
        if not name or name == "ALL OTHER NAMES":
            continue
        try:
            count = int(r.get("count", "0"))
        except ValueError:
            count = 0
        if count >= min_count:
            # census stores ALLCAPS; title-case for realistic surface form
            yield name.title(), count


def wikidata_names(kind_qid, lang, limit, retries=3):
    """Wikidata name-entity labels in one language (CC0). Yields name strings."""
    q = (f'SELECT DISTINCT ?l WHERE {{ ?n wdt:P31 wd:{kind_qid}. '
         f'?n rdfs:label ?l. FILTER(lang(?l)="{lang}") }} LIMIT {limit}')
    url = f"{WDQS}?format=json&query={urllib.parse.quote(q)}"
    req = urllib.request.Request(
        url, headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
    data = None
    for attempt in range(retries):
        try:
            data = json.loads(urllib.request.urlopen(req, timeout=90).read().decode("utf-8"))
            break
        except Exception:  # noqa: BLE001 — transient WDQS truncation/timeout
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    for b in data["results"]["bindings"]:
        v = b["l"]["value"].strip()
        # skip labels that are clearly not a bare name (disambiguation, parens, digits)
        if v and "(" not in v and not any(c.isdigit() for c in v) and len(v) <= 40:
            yield v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/names"))
    ap.add_argument("--per-lang", type=int, default=3000, help="Wikidata names per (kind, language)")
    ap.add_argument("--census-min-count", type=int, default=100,
                    help="drop surnames rarer than this (100 = keep the real long tail, ~160k)")
    ap.add_argument("--sources", default="census,wikidata")
    ap.add_argument("--wd-langs", default=",".join(WD_LANGS))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sources = set(args.sources.split(","))
    out_path = args.out / "corpus.jsonl"
    seen = set()
    counts = {}
    with out_path.open("w") as f:
        def emit(name, kind, script, source, license):
            key = (name, kind)
            if key in seen:
                return
            seen.add(key)
            f.write(json.dumps({"name": name, "kind": kind, "script": script,
                                "source": source, "license": license}, ensure_ascii=False) + "\n")
            counts[(kind, source)] = counts.get((kind, source), 0) + 1

        if "census" in sources:
            sys.stderr.write("US Census 2010 surnames (public domain)...\n")
            n = 0
            for name, _c in census_surnames(args.census_min_count):
                emit(name, "family", "latin", "us-census-2010", "public-domain")
                n += 1
            sys.stderr.write(f"  census surnames: {n}\n")

        if "wikidata" in sources:
            for lang in args.wd_langs.split(","):
                for kind, qid in WD_KINDS.items():
                    try:
                        got = 0
                        for name in wikidata_names(qid, lang, args.per_lang):
                            emit(name, kind, lang, "wikidata", "CC0-1.0")
                            got += 1
                        sys.stderr.write(f"  wikidata {kind}/{lang}: {got}\n")
                    except Exception as exc:  # noqa: BLE001
                        sys.stderr.write(f"  wikidata {kind}/{lang} FAILED: {str(exc)[:70]}\n")
                    time.sleep(1.0)  # be polite to WDQS

    total = sum(counts.values())
    sys.stderr.write(f"\nwrote {total} unique names -> {out_path}\n")
    for (kind, source), n in sorted(counts.items()):
        sys.stderr.write(f"  {kind:8s} {source:16s} {n}\n")
    # provenance note next to the corpus
    (args.out / "SOURCES.md").write_text(
        "# Name corpus provenance\n\n"
        "- **US Census 2010 surnames** — public domain (U.S. Census Bureau).\n"
        "- **Wikidata given/family names** — CC0 1.0 (public-domain dedication).\n\n"
        "Aggregated name strings only; used to fill synthetic templates. No real "
        "individual's record is stored or reproduced.\n")


if __name__ == "__main__":
    main()
