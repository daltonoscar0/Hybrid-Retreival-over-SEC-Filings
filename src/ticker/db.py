"""The data layer: schema, inserts, and time-filtered reads.

Convention enforced here, not by caller discipline: every datetime that
crosses this module's boundary (`filed_at` on insert, `as_of` on read) must be
timezone-aware. A naive datetime compared against DuckDB's TIMESTAMPTZ column
would otherwise be silently reinterpreted, which is exactly the kind of bug
this module exists to make impossible.

`sentences.filed_at` is denormalized from `filings.filed_at` on purpose. Every
novelty-facing query filters on the sentence row's own `filed_at`, not a join
back to `filings`, so the time filter cannot be dropped by refactoring away a
join.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb

from ticker.records import Chunk, Filing, Section, Sentence

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS filings (
    accession   VARCHAR PRIMARY KEY,
    cik         BIGINT NOT NULL,
    ticker      VARCHAR NOT NULL,
    sector      VARCHAR NOT NULL,
    form        VARCHAR NOT NULL,
    filed_at    TIMESTAMPTZ NOT NULL,
    period_end  DATE,
    url         VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS sections (
    section_id  VARCHAR PRIMARY KEY,
    accession   VARCHAR NOT NULL REFERENCES filings(accession),
    item        VARCHAR NOT NULL,
    text        VARCHAR NOT NULL,
    char_start  INTEGER NOT NULL,
    char_end    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sentences (
    sentence_id VARCHAR PRIMARY KEY,
    section_id  VARCHAR NOT NULL REFERENCES sections(section_id),
    ordinal     INTEGER NOT NULL,
    text        VARCHAR NOT NULL,
    filed_at    TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     VARCHAR PRIMARY KEY,
    section_id   VARCHAR NOT NULL REFERENCES sections(section_id),
    sentence_ids VARCHAR[] NOT NULL,
    text         VARCHAR NOT NULL
);
"""


def connect(path: str | Path = ":memory:") -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(path))


def connect_readonly(path: str | Path) -> duckdb.DuckDBPyConnection:
    """Read-only handle, for every tool that only queries.

    DuckDB takes an exclusive file lock on a read-write connection, so one
    reader opened read-write locks every other process out of the corpus.
    That is not a theoretical concern here: the Phase 4 scoring pass runs for
    tens of minutes, and it must not be able to block a judging session, or a
    judging session it. Read-only handles share.

    Not the default for `connect`, because `scripts/ingest_corpus.py` writes
    and an in-memory database cannot be opened read-only at all.
    """
    return duckdb.connect(str(path), read_only=True)


def create_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA_SQL)


def _require_aware(value: datetime, label: str) -> None:
    if value.tzinfo is None:
        raise ValueError(
            f"{label} must be a timezone-aware datetime, got naive {value!r}. "
            "Ticker's convention is aware datetimes everywhere filed_at or "
            "as_of is used; a naive value here would compare against "
            "TIMESTAMPTZ silently and wrongly."
        )


def insert_filing(con: duckdb.DuckDBPyConnection, filing: Filing) -> None:
    _require_aware(filing.filed_at, "Filing.filed_at")
    con.execute(
        "INSERT INTO filings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            filing.accession,
            filing.cik,
            filing.ticker,
            filing.sector,
            filing.form,
            filing.filed_at,
            filing.period_end,
            filing.url,
        ],
    )


def insert_section(con: duckdb.DuckDBPyConnection, section: Section) -> None:
    con.execute(
        "INSERT INTO sections VALUES (?, ?, ?, ?, ?, ?)",
        [
            section.section_id,
            section.accession,
            section.item,
            section.text,
            section.char_start,
            section.char_end,
        ],
    )


def insert_sentence(con: duckdb.DuckDBPyConnection, sentence: Sentence) -> None:
    _require_aware(sentence.filed_at, "Sentence.filed_at")
    con.execute(
        "INSERT INTO sentences VALUES (?, ?, ?, ?, ?)",
        [
            sentence.sentence_id,
            sentence.section_id,
            sentence.ordinal,
            sentence.text,
            sentence.filed_at,
        ],
    )


def insert_chunk(con: duckdb.DuckDBPyConnection, chunk: Chunk) -> None:
    con.execute(
        "INSERT INTO chunks VALUES (?, ?, ?, ?)",
        [chunk.chunk_id, chunk.section_id, list(chunk.sentence_ids), chunk.text],
    )


_SENTENCE_SELECT = "s.sentence_id, s.section_id, s.ordinal, s.text, s.filed_at"


def _rows_to_sentences(rows: list[tuple]) -> list[Sentence]:
    return [Sentence(*row) for row in rows]


def prior_sentences(
    con: duckdb.DuckDBPyConnection, cik: int, as_of: datetime
) -> list[Sentence]:
    """Sentences for `cik` filed strictly before `as_of`.

    No default for `as_of`: a firm language model fit on documents at or
    after the one it is meant to score is a leak, not a feature. Strict `<`,
    never `<=` -- a filing at exactly `as_of` must not leak into its own
    background.
    """
    _require_aware(as_of, "as_of")
    rows = con.execute(
        f"""
        SELECT {_SENTENCE_SELECT}
        FROM sentences s
        JOIN sections sec ON s.section_id = sec.section_id
        JOIN filings f ON sec.accession = f.accession
        WHERE f.cik = ? AND s.filed_at < ?
        ORDER BY s.filed_at, s.sentence_id
        """,
        [cik, as_of],
    ).fetchall()
    return _rows_to_sentences(rows)


def background_sentences(
    con: duckdb.DuckDBPyConnection, exclude_cik: int, as_of: datetime
) -> list[Sentence]:
    """Sentences filed strictly before `as_of`, excluding `exclude_cik`.

    Corpus-wide minus the target firm. `filings.sector` exists as of Phase 1
    but this function does not filter on it; sector-scoped background (peers
    only, per PLAN.md phase 4) is that phase's job, not this one's.
    """
    _require_aware(as_of, "as_of")
    rows = con.execute(
        f"""
        SELECT {_SENTENCE_SELECT}
        FROM sentences s
        JOIN sections sec ON s.section_id = sec.section_id
        JOIN filings f ON sec.accession = f.accession
        WHERE f.cik != ? AND s.filed_at < ?
        ORDER BY s.filed_at, s.sentence_id
        """,
        [exclude_cik, as_of],
    ).fetchall()
    return _rows_to_sentences(rows)
