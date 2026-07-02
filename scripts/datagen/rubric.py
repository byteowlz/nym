"""Coverage rubric: sweep (language x topic x style) cells systematically.

Instead of asking the LLM for random templates, we enumerate cells across three
axes and request a few templates per cell. This guarantees breadth — many
languages, domains, and registers — rather than whatever the model gravitates to.

Each language carries a Faker locale so the *filled* values match the language of
the surrounding text (a German template gets German names/addresses/IDs).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Language:
    name: str          # human name for the LLM prompt
    faker: str         # Faker locale for value generation
    script: str        # "latin" | "nonlatin" (hint: use a multilingual tokenizer)


LANGUAGES: List[Language] = [
    Language("English (US)", "en_US", "latin"),
    Language("English (UK)", "en_GB", "latin"),
    Language("German", "de_DE", "latin"),
    Language("French", "fr_FR", "latin"),
    Language("Spanish", "es_ES", "latin"),
    Language("Italian", "it_IT", "latin"),
    Language("Portuguese (Brazil)", "pt_BR", "latin"),
    Language("Dutch", "nl_NL", "latin"),
    Language("Polish", "pl_PL", "latin"),
    Language("Swedish", "sv_SE", "latin"),
    Language("Czech", "cs_CZ", "latin"),
    Language("Romanian", "ro_RO", "latin"),
    Language("Turkish", "tr_TR", "latin"),
    Language("Finnish", "fi_FI", "latin"),
    Language("Danish", "da_DK", "latin"),
    Language("Greek", "el_GR", "nonlatin"),
    Language("Russian", "ru_RU", "nonlatin"),
    Language("Ukrainian", "uk_UA", "nonlatin"),
    Language("Japanese", "ja_JP", "nonlatin"),
    Language("Chinese (Simplified)", "zh_CN", "nonlatin"),
    Language("Korean", "ko_KR", "nonlatin"),
    Language("Arabic", "ar_AA", "nonlatin"),
    Language("Hindi", "hi_IN", "nonlatin"),
]

TOPICS: List[str] = [
    "hospital / clinical notes",
    "banking and wire transfers",
    "health insurance claims",
    "human resources / onboarding",
    "legal contracts and filings",
    "travel booking and immigration",
    "e-commerce orders and shipping",
    "customer support chat",
    "personal email correspondence",
    "government and tax forms",
    "university enrollment records",
    "real estate and rental agreements",
    "telecom and utility bills",
    "social media posts and DMs",
    "IT system and access logs",
    "job applications and resumes",
    "pharmacy prescriptions",
    "hotel and restaurant reservations",
    "ride-share and food delivery",
    "loan and credit applications",
]

STYLES: List[str] = [
    "a casual chat / SMS message",
    "a structured form with 'Field: value' lines",
    "an email with a greeting and signature",
    "a call-center or interview transcript",
    "a terse log line or database row",
    "a handwritten-style note with abbreviations",
    "a bulleted list of details",
    # Long-form / multi-sentence — matches the 200-word inference window.
    "a multi-paragraph formal letter or report spanning several sentences",
    "a detailed case / incident report with labeled sections",
    "a two-person message thread (several turns)",
    "a narrative paragraph telling a short story",
    "a full clinical / official note with history and details",
]

# A fourth axis: rotating persona / tone / length / ADVERSARIAL twists. Same
# (language, topic, style) cell produces different output per flavor, and the
# adversarial ones inject the hard cases a detector must handle.
FLAVORS: List[str] = [
    "first-person perspective",
    "third-person, written by an agent about someone else",
    "terse shorthand with abbreviations",
    "verbose, formal and bureaucratic",
    "an emotional complaint tone",
    "casual and conversational",
    "span multiple sentences across at least two paragraphs",
    "very short: one or two lines with only 1-2 PII items",
    "dense: pack in 5 to 8 different PII items",
    "put at least one PII value glued to punctuation, inside a URL, or in an email",
    "place two entities of the SAME type adjacent (e.g. two people, or two dates)",
    "also include a realistic NON-PII lookalike (order number, SKU, tracking or model number) and do NOT bracket it",
    "use an unusual or region-specific format for a date, phone number, or ID",
    "include quoted speech or a signature block",
    "mix 'Field: value' lines together with running prose",
    "include an abbreviation or acronym next to a name or place",
]


@dataclass(frozen=True)
class Cell:
    language: Language
    topic: str
    style: str
    flavor: str


def cells(n: int, seed: int = 0) -> List[Cell]:
    """Return `n` distinct (language, topic, style, flavor) cells, shuffled but
    deterministic for a given seed. The full space is ~90k combinations, so we
    build cells lazily by sampling each axis rather than materializing it."""
    rng = random.Random(seed)
    out: List[Cell] = []
    seen = set()
    # Sample distinct combos; fall back to allowing repeats if n is very large.
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        c = Cell(rng.choice(LANGUAGES), rng.choice(TOPICS), rng.choice(STYLES), rng.choice(FLAVORS))
        key = (c.language.faker, c.topic, c.style, c.flavor)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    while len(out) < n:  # only if the space is exhausted
        out.append(Cell(rng.choice(LANGUAGES), rng.choice(TOPICS), rng.choice(STYLES), rng.choice(FLAVORS)))
    return out
