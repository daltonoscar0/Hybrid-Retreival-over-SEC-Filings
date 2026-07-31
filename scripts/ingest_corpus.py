#!/usr/bin/env python3
"""Rebuild data/ticker.duckdb from the on-disk filing cache.

filings -> sections (ticker.sections.extract_sections) -> sentences
(ticker.sentence_split, filed_at denormalized per row) -> chunks
(ticker.chunker, 4-sentence window / stride 2), for every cached filing that
belongs to the locked universe (ticker.universe).

The DuckDB file is disposable: this script deletes and recreates it every
run rather than migrating, per PLAN.md's Phase 0 schema note.

Section extraction is a different agent's module (src/ticker/sections.py)
and is imported lazily so this script's cache-loading and filing-insertion
logic can be smoke-tested before that module exists. When it is missing,
this script ingests filings only and reports how many sections/sentences/
chunks it could not build, rather than crashing.

Usage: .venv/bin/python3 scripts/ingest_corpus.py
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import duckdb

from ticker import db
from ticker.chunker import chunk_sentences
from ticker.download import DEFAULT_CACHE_DIR, DownloadedFiling, iter_cached_filings
from ticker.records import Filing, Section, Sentence
from ticker.sentence_split import split_sentences
from ticker.universe import UNIVERSE

DEFAULT_DB_PATH = Path("data/ticker.duckdb")

_UNIVERSE_CIKS = {c.cik for c in UNIVERSE}


def _load_section_extractor():
    """Lazy import so a missing ticker.sections does not crash the script."""
    try:
        from ticker.sections import extract_sections
    except ImportError:
        return None
    return extract_sections


def _ingest_filing(
    con: duckdb.DuckDBPyConnection,
    downloaded: DownloadedFiling,
    extract_sections,
    counts: Counter,
    failures: list[tuple[str, str, str]],
) -> None:
    filing = Filing(
        accession=downloaded.accession,
        cik=downloaded.cik,
        ticker=downloaded.ticker,
        sector=downloaded.sector,
        form=downloaded.form,
        filed_at=downloaded.filed_at,
        period_end=downloaded.period_end,
        url=downloaded.url,
    )
    db.insert_filing(con, filing)
    counts["filings"] += 1

    if extract_sections is None:
        counts["filings_blocked_on_sections"] += 1
        return

    extraction = extract_sections(downloaded.text, downloaded.form)

    for item, reason in extraction.failures:
        failures.append((downloaded.accession, item, reason))
        counts["section_failures"] += 1

    for item, char_start, char_end in extraction.sections:
        section_id = f"{filing.accession}#{item}"
        section_text = downloaded.text[char_start:char_end]
        section = Section(
            section_id=section_id,
            accession=filing.accession,
            item=item,
            text=section_text,
            char_start=char_start,
            char_end=char_end,
        )
        db.insert_section(con, section)
        counts["sections"] += 1

        sentences = [
            Sentence(
                sentence_id=f"{section_id}#{i}",
                section_id=section_id,
                ordinal=i,
                text=sentence_text,
                filed_at=filing.filed_at,
            )
            for i, sentence_text in enumerate(split_sentences(section_text))
        ]
        for sentence in sentences:
            db.insert_sentence(con, sentence)
        counts["sentences"] += len(sentences)

        chunks = chunk_sentences(sentences)
        for chunk in chunks:
            db.insert_chunk(con, chunk)
        counts["chunks"] += len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="ingest at most this many cached filings (smoke testing)",
    )
    args = parser.parse_args()

    if args.db.exists():
        os.remove(args.db)
    args.db.parent.mkdir(parents=True, exist_ok=True)

    con = db.connect(args.db)
    db.create_schema(con)

    extract_sections = _load_section_extractor()
    if extract_sections is None:
        print("ticker.sections not importable -- ingesting filings only, "
              "sections/sentences/chunks blocked until it lands.")

    filings = [f for f in iter_cached_filings(args.cache_dir) if f.cik in _UNIVERSE_CIKS]
    filings.sort(key=lambda f: f.filed_at)
    if args.limit is not None:
        filings = filings[: args.limit]

    counts: Counter = Counter()
    failures: list[tuple[str, str, str]] = []
    for downloaded in filings:
        _ingest_filing(con, downloaded, extract_sections, counts, failures)

    con.close()

    print(f"filings in cache matching universe: {len(filings)}")
    for key in ("filings", "filings_blocked_on_sections", "sections",
                "section_failures", "sentences", "chunks"):
        print(f"  {key}: {counts.get(key, 0)}")

    if failures:
        print(f"\n{len(failures)} section extraction failures (accession, item, reason):")
        for accession, item, reason in failures[:20]:
            print(f"  {accession} {item}: {reason}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")


if __name__ == "__main__":
    main()
