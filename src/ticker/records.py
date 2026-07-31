"""Frozen record types mirroring the DuckDB schema rows, one class per table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True, slots=True)
class Filing:
    accession: str
    cik: int
    ticker: str
    form: str
    filed_at: datetime
    period_end: date | None
    url: str


@dataclass(frozen=True, slots=True)
class Section:
    section_id: str
    accession: str
    item: str
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class Sentence:
    sentence_id: str
    section_id: str
    ordinal: int
    text: str
    filed_at: datetime


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    section_id: str
    sentence_ids: tuple[str, ...]
    text: str
