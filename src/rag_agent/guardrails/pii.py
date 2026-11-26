"""Detect and mask personal data and secrets (regex + checksums, no LLM).

Why mask *before* the LLM: the user's message is sent to Groq, possibly to a web
search engine, and stored in chat history. A pasted email, phone number, card
or API key has no business in any of those. Masking keeps the request useful
("my card [CARD_1] was declined") while the raw value never leaves the server.

Card numbers are validated with the Luhn checksum and Aadhaar numbers with the
Verhoeff checksum, so ordinary long numbers (order IDs, byte counts like
52428800) are not mistaken for them.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# --- checksums -------------------------------------------------------------------------


def luhn_ok(digits: str) -> bool:
    total, double = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if double:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        double = not double
    return total % 10 == 0


_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_ok(digits: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


# --- patterns --------------------------------------------------------------------------
# Order matters: secrets first (they can contain digit runs), then IDs, then contact info.


@dataclass(frozen=True)
class PIIType:
    name: str
    pattern: re.Pattern
    sensitive: bool  # True: always redacted in answers too (not just masked in input)
    validate: object = None  # optional callable(match_text) -> bool


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


PII_TYPES: list[PIIType] = [
    PIIType(
        "API_KEY",
        re.compile(
            r"\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{20,}|gsk_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}"
            r"|gh[pousr]_[A-Za-z0-9]{30,}|xox[abprs]-[A-Za-z0-9-]{10,}|tvly-[A-Za-z0-9]{20,}"
            r"|lsv2_[a-z]{2}_[A-Za-z0-9_]{20,}|AIza[0-9A-Za-z_-]{35})\b"
        ),
        sensitive=True,
    ),
    PIIType(
        "PASSWORD",
        re.compile(r"(?i)\b(?:password|passwd|pwd|secret)\s*[:=]\s*\S{4,}"),
        sensitive=True,
    ),
    PIIType(
        "CARD",
        re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
        sensitive=True,
        validate=lambda m: 13 <= len(_digits(m)) <= 19 and luhn_ok(_digits(m)),
    ),
    PIIType(
        "AADHAAR",
        re.compile(r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b"),
        sensitive=True,
        validate=lambda m: verhoeff_ok(_digits(m)),
    ),
    PIIType("PAN", re.compile(r"\b[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]\b"), sensitive=True),
    PIIType(
        "SSN", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"), sensitive=True
    ),
    PIIType(
        "EMAIL",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        sensitive=False,
    ),
    PIIType(
        "PHONE",
        re.compile(
            r"(?<![\w.])(?:\+91[ -]?)?[6-9]\d{4}[ -]?\d{5}\b"  # Indian mobile
            r"|(?<![\w.])\+\d{1,3}[ -]?\(?\d{2,4}\)?[ -]?\d{3,4}[ -]?\d{3,4}\b"  # international
        ),
        sensitive=False,
    ),
]


@dataclass
class PIIResult:
    text: str
    found: Counter = field(default_factory=Counter)

    @property
    def summary(self) -> str:
        return ", ".join(
            f"{n} {t.lower().replace('_', ' ')}{'s' if n > 1 else ''}"
            for t, n in self.found.items()
        )


def mask_pii(text: str, only_sensitive: bool = False, style: str = "mask") -> PIIResult:
    """Replace PII with placeholders.

    ``style="mask"``   -> numbered placeholders ``[EMAIL_1]`` (input side: keeps
                          the sentence meaningful for the LLM).
    ``style="redact"`` -> ``[REDACTED CARD]`` (output side).
    ``only_sensitive`` -> skip contact info (emails/phones), which legitimately
                          appears in the user's own documents and answers.
    """
    found: Counter = Counter()
    for pii in PII_TYPES:
        if only_sensitive and not pii.sensitive:
            continue

        def repl(m: re.Match, pii=pii) -> str:
            if pii.validate and not pii.validate(m.group(0)):
                return m.group(0)
            found[pii.name] += 1
            if style == "redact":
                return f"[REDACTED {pii.name.replace('_', ' ')}]"
            return f"[{pii.name}_{found[pii.name]}]"

        text = pii.pattern.sub(repl, text)
    return PIIResult(text=text, found=found)
