"""End-to-end ingestion of a single filing: filings -> sections -> sentences -> chunks.

Section extraction here is provisional: one section spanning the whole
document body, labeled PROVISIONAL_FULL_TEXT. The real item-boundary
extractor, with per-item regression tests against hand-checked filings, is
Phase 1's job and does not belong in this module.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from ticker import db
from ticker.chunker import chunk_sentences
from ticker.download import DEFAULT_CACHE_DIR, fetch_latest_filing
from ticker.records import Filing, Section, Sentence
from ticker.sentence_split import split_sentences

PROVISIONAL_ITEM = "PROVISIONAL_FULL_TEXT"


def ingest_latest_filing(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    form: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Filing:
    downloaded = fetch_latest_filing(ticker, form, cache_dir=cache_dir)

    filing = Filing(
        accession=downloaded.accession,
        cik=downloaded.cik,
        ticker=downloaded.ticker,
        form=downloaded.form,
        filed_at=downloaded.filed_at,
        period_end=downloaded.period_end,
        url=downloaded.url,
    )
    db.insert_filing(con, filing)

    section_id = f"{filing.accession}#{PROVISIONAL_ITEM}"
    section = Section(
        section_id=section_id,
        accession=filing.accession,
        item=PROVISIONAL_ITEM,
        text=downloaded.text,
        char_start=0,
        char_end=len(downloaded.text),
    )
    db.insert_section(con, section)

    sentences = [
        Sentence(
            sentence_id=f"{section_id}#{i}",
            section_id=section_id,
            ordinal=i,
            text=sentence_text,
            filed_at=filing.filed_at,
        )
        for i, sentence_text in enumerate(split_sentences(downloaded.text))
    ]
    for sentence in sentences:
        db.insert_sentence(con, sentence)

    for chunk in chunk_sentences(sentences):
        db.insert_chunk(con, chunk)

    return filing
