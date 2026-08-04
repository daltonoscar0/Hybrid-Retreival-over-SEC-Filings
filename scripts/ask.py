#!/usr/bin/env python3
"""Ad hoc search with per-sentence novelty highlighting.

The Phase 7 interface, as a CLI rather than a web app. `scripts/search.py`
runs the fixed query set into run files for evaluation; this runs one typed
query and prints something a person reads.

Novelty is a toggle, off by default, exactly as PLAN section 6 requires. With
it off, ranking is pure relevance and novelty only colours the text. With
`--novelty` the results are reordered by relevance rank blended with novelty
rank, and the header says so, because a reordering the reader cannot see is
the silent-term-in-the-ranking-function mistake that section warns about.

Colour bands are within-document z-scores of the raw contrast, per invariant
3. The number printed by `--show-scores` is the raw contrast, because that is
the quantity that means something across documents.

Usage:
  uv run python scripts/ask.py "goodwill impairment charge"
  uv run python scripts/ask.py "supply chain disruption" --novelty
  uv run python scripts/ask.py "credit loss provision" --ticker CMA --form 10-K
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

from ticker.display import (
    BAND_HIGH,
    BAND_MID,
    build_sentence_views,
    document_novelty,
    load_novelty,
    rerank_by_novelty,
)

DEFAULT_DB = Path("data/ticker.duckdb")

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
YELLOW = "\033[1;33m"
ORANGE = "\033[33m"
OFF = "\033[0m"

BAND_COLOUR = {BAND_HIGH: YELLOW, BAND_MID: ORANGE}

RETRIEVERS = {
    "bm25": "ticker.retrieval.bm25:load_retriever",
    "dense": "ticker.retrieval.dense:load_retriever",
}


def _load_retriever(spec: str):
    module_path, _, attr = spec.partition(":")
    from importlib import import_module

    return getattr(import_module(module_path), attr)()


def _chunk_rows(con: duckdb.DuckDBPyConnection, chunk_ids: list[str]) -> dict:
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" for _ in chunk_ids)
    rows = con.execute(
        f"""
        SELECT c.chunk_id, c.sentence_ids, sec.item, f.ticker, f.form, f.filed_at
        FROM chunks c
        JOIN sections sec ON c.section_id = sec.section_id
        JOIN filings f ON sec.accession = f.accession
        WHERE c.chunk_id IN ({placeholders})
        """,
        chunk_ids,
    ).fetchall()
    return {r[0]: r[1:] for r in rows}


def _sentence_texts(con: duckdb.DuckDBPyConnection, sentence_ids: list[str]) -> dict:
    if not sentence_ids:
        return {}
    placeholders = ",".join("?" for _ in sentence_ids)
    return dict(
        con.execute(
            f"SELECT sentence_id, text FROM sentences WHERE sentence_id IN ({placeholders})",
            sentence_ids,
        ).fetchall()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("query")
    parser.add_argument("--retriever", default="dense", choices=sorted(RETRIEVERS))
    parser.add_argument("-k", type=int, default=5, help="results to show")
    parser.add_argument(
        "--novelty",
        action="store_true",
        help="rerank by relevance blended with novelty; off by default",
    )
    parser.add_argument("--weight", type=float, default=0.5, help="novelty weight when reranking")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--form", default=None)
    parser.add_argument("--show-scores", action="store_true", help="print raw contrasts")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()

    con = duckdb.connect(str(args.db), read_only=True)
    novelty = load_novelty()
    if not novelty:
        print(f"{DIM}no novelty scores found; showing relevance only{OFF}", file=sys.stderr)

    retriever = _load_retriever(RETRIEVERS[args.retriever])
    # Over-fetch so the filters and the rerank have something to work with.
    depth = max(args.k * 20, 100)
    hits = [(h.chunk_id, h.score) for h in retriever.search(args.query, depth)]

    meta = _chunk_rows(con, [c for c, _ in hits])
    if args.ticker:
        hits = [h for h in hits if meta.get(h[0]) and meta[h[0]][2] == args.ticker.upper()]
    if args.form:
        hits = [h for h in hits if meta.get(h[0]) and meta[h[0]][3] == args.form]

    if args.novelty:
        chunk_novelty = {
            chunk_id: document_novelty(novelty.get(s) for s in meta[chunk_id][0])
            for chunk_id, _ in hits
            if chunk_id in meta
        }
        hits = rerank_by_novelty(hits, chunk_novelty, weight=args.weight)

    hits = hits[: args.k]

    mode = (
        f"relevance blended with novelty, weight {args.weight}"
        if args.novelty
        else "relevance only"
    )
    print()
    print(f"{BOLD}{CYAN}{args.query}{OFF}")
    print(f"{DIM}{args.retriever}, ranked by {mode}{OFF}")
    print(f"{DIM}colour is within-document novelty: {YELLOW}high{OFF}{DIM} {ORANGE}elevated{OFF}{OFF}")

    for rank, (chunk_id, score) in enumerate(hits, 1):
        if chunk_id not in meta:
            continue
        sentence_ids, item, ticker, form, filed_at = meta[chunk_id]
        texts = _sentence_texts(con, list(sentence_ids))
        views = build_sentence_views(
            list(sentence_ids), [texts.get(s, "") for s in sentence_ids], novelty
        )
        doc = document_novelty(v.raw for v in views)
        doc_str = f"  novelty {doc:+.3f}" if doc is not None else "  novelty n/a"

        print()
        print(f"{BOLD}{rank}.{OFF} {ticker} {form} item {item}  filed {filed_at.date()}"
              f"  {DIM}score {score:.4f}{doc_str}{OFF}")
        for view in views:
            if not view.text:
                continue
            colour = BAND_COLOUR.get(view.band)
            body = f"{colour}{view.text}{OFF}" if colour else view.text
            suffix = f" {DIM}[{view.raw:+.2f}]{OFF}" if args.show_scores and view.raw is not None else ""
            print(f"   {body}{suffix}")

    print()
    con.close()


if __name__ == "__main__":
    main()
