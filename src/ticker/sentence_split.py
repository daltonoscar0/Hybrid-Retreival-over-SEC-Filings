"""Provisional sentence splitter.

Splits on sentence-final punctuation, then re-merges any split that landed
right after a known abbreviation or a mid-number decimal point, so
"U.S. GAAP", "Item No. 3", and "$1.5 million" survive as one sentence.

This is a simple first cut so one filing can be ingested end to end. The
tuned financial-prose splitter -- validated against dollar amounts, "Inc.",
"Corp.", numbered list items, decimal percentages, "e.g."/"i.e.", and
tabular fragments -- is Phase 1's job.
"""

from __future__ import annotations

import re

_ABBREVIATIONS = {
    "inc", "corp", "co", "ltd", "llc", "lp", "u.s", "u.k", "no", "vs",
    "mr", "mrs", "ms", "dr", "jr", "sr", "st", "e.g", "i.e", "etc",
    "fig", "figs", "approx", "misc", "dept", "gov", "vol", "art", "sec",
}

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d\"'(])")
_TRAILING_DECIMAL = re.compile(r"\d\.$")


def _last_token(fragment: str) -> str:
    stripped = fragment.rstrip(".!?")
    tail = re.split(r"[\s(]", stripped)[-1]
    return tail.lower().rstrip(".")


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    pieces = _SENTENCE_BOUNDARY.split(normalized)
    sentences: list[str] = []
    buffer = ""
    for piece in pieces:
        buffer = f"{buffer} {piece}".strip() if buffer else piece
        if _last_token(buffer) in _ABBREVIATIONS or _TRAILING_DECIMAL.search(buffer):
            continue
        sentences.append(buffer)
        buffer = ""
    if buffer:
        sentences.append(buffer)
    return sentences
