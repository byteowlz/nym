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
    "a formal letter or document",
    "a casual chat / SMS message",
    "a structured form with 'Field: value' lines",
    "an email with a greeting and signature",
    "a call-center or interview transcript",
    "a terse log line or database row",
    "a handwritten-style note with abbreviations",
    "a bulleted list of details",
]


@dataclass(frozen=True)
class Cell:
    language: Language
    topic: str
    style: str


def cells(n: int, seed: int = 0) -> List[Cell]:
    """Return `n` distinct (language, topic, style) cells, shuffled but
    deterministic for a given seed, cycling so every language recurs."""
    rng = random.Random(seed)
    full = [Cell(l, t, s) for l in LANGUAGES for t in TOPICS for s in STYLES]
    rng.shuffle(full)
    if n <= len(full):
        return full[:n]
    # Repeat the shuffled space if more cells are requested than exist.
    out: List[Cell] = []
    while len(out) < n:
        out.extend(full)
    return out[:n]
