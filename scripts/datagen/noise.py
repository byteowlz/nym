"""Label-preserving text corruption to emulate real-world noisy inputs.

Applied per-segment during generation so entity spans stay exact after the text
length changes (see generate.py). Corrupting entity values too (a smudged name,
an OCR'd account number) is deliberate — it teaches the model to still tag them.

Noise kinds, roughly in order of realism for scanned/OCR'd and typed text:
  - OCR character confusions (o/0, l/1/I, S/5, rn/m, cl/d, ...)
  - keyboard typos (delete / insert / transpose / adjacent-key substitution)
  - spacing glitches (double/'missing spaces, stray newlines)
  - case flips
"""
from __future__ import annotations

import random
from typing import List

# Single-char OCR confusions (bidirectional-ish; applied one way per hit).
OCR_CHAR = {
    "o": "0", "O": "0", "0": "O",
    "l": "1", "I": "1", "1": "l",
    "S": "5", "5": "S", "s": "5",
    "B": "8", "8": "B",
    "g": "9", "9": "g",
    "Z": "2", "2": "Z",
    "b": "6", "6": "b",
    "G": "6", "q": "9",
    "|": "l", "!": "1",
}
# Multi-char OCR confusions.
OCR_DIGRAPH = {"rn": "m", "m": "rn", "cl": "d", "vv": "w", "nn": "m", "ii": "ll"}

# Rough QWERTY neighbors for typo substitution.
_NEIGHBORS = {
    "a": "sqwz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgcv", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}

LEVELS = {"light": 0.02, "medium": 0.05, "heavy": 0.10}


def _ocr(s: str, rng: random.Random, p: float) -> str:
    # Digraph passes first.
    for a, b in OCR_DIGRAPH.items():
        if a in s and rng.random() < p * 3:
            s = s.replace(a, b, 1)
    out = []
    for ch in s:
        if ch in OCR_CHAR and rng.random() < p:
            out.append(OCR_CHAR[ch])
        else:
            out.append(ch)
    return "".join(out)


def _typos(s: str, rng: random.Random, p: float) -> str:
    out: List[str] = []
    i = 0
    chars = list(s)
    while i < len(chars):
        ch = chars[i]
        r = rng.random()
        if r < p and ch.isalpha():
            kind = rng.random()
            if kind < 0.3:  # delete
                i += 1
                continue
            if kind < 0.55 and ch.lower() in _NEIGHBORS:  # substitute neighbor
                sub = rng.choice(_NEIGHBORS[ch.lower()])
                out.append(sub.upper() if ch.isupper() else sub)
                i += 1
                continue
            if kind < 0.8 and i + 1 < len(chars):  # transpose
                out.append(chars[i + 1])
                out.append(ch)
                i += 2
                continue
            # duplicate / insert
            out.append(ch)
            out.append(ch)
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _spacing(s: str, rng: random.Random, p: float) -> str:
    out = []
    for ch in s:
        if ch == " " and rng.random() < p * 2:
            out.append("" if rng.random() < 0.5 else "  ")
        else:
            out.append(ch)
        if ch != " " and rng.random() < p * 0.4:
            out.append(" ")  # stray space
    return "".join(out)


def _case(s: str, rng: random.Random, p: float) -> str:
    return "".join(
        (c.upper() if c.islower() else c.lower()) if (c.isalpha() and rng.random() < p) else c
        for c in s
    )


def corrupt_text(s: str, rng: random.Random, level: str = "medium") -> str:
    """Apply a random subset of noise kinds at the given intensity level."""
    if not s:
        return s
    p = LEVELS.get(level, 0.05)
    # Randomly pick which noise families to apply this time.
    if rng.random() < 0.7:
        s = _ocr(s, rng, p)
    if rng.random() < 0.6:
        s = _typos(s, rng, p)
    if rng.random() < 0.4:
        s = _spacing(s, rng, p)
    if rng.random() < 0.25:
        s = _case(s, rng, p)
    return s
