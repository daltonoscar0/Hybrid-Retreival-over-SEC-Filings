"""Judging pool construction: standard TREC ad hoc pooling.

Why pool instead of judging every chunk
----------------------------------------
The corpus has 146,449 chunks. Sixty queries against the full corpus is 8.8
million (query, chunk) pairs; nobody judges that and nobody expects it. TREC
ad hoc pooling (Sparck Jones & van Rijsbergen) instead takes the union of the
top-K results from several qualitatively different retrieval systems and
judges only that union, on the working assumption that a chunk no system
placed near the top of its ranking is, with high probability, not relevant
enough to move the metrics computed over this pool.

Pooling three sources rather than one lexical ranker matters because the
sources fail differently. BM25 misses paraphrase and synonymy. A dense
retriever misses exact numeric and proper-noun matches it was never trained
to weight highly. The keyword seed here is a deliberately dumb substring
match with no learned weighting at all: it is a floor under both learned
systems, because "both the lexical and the dense model underweight this
term" is exactly the failure mode neither model's top-20 alone would catch.

The pool is a ceiling on measurable recall for every system this project
evaluates, not just a one-off floor for this document. Every system on the
Phase 2/3 ablation ladder (BM25, dense, RRF, weighted fusion) re-ranks a
candidate set drawn from BM25's results union dense's results -- exactly the
two learned sources pooled here -- so no evaluated system can surface a
chunk outside the judged pool. A future system built on a materially
different signal would need its own top-K folded into the pool before it can
be evaluated fairly; scoring it against this pool alone would silently
undercount its recall. This is the standard caveat on TREC pooling and it is
the reason `pool_query` takes a list of retrievers rather than hard-coding
two.

Judging order is shuffled per query with a seed derived from (base seed,
query_id), so a rerun with the same seed reproduces the same order, but the
order itself carries no information about which system ranked a chunk or how
highly it ranked there. A judge who noticed "position 1 is always BM25's top
hit" would start trusting position 1 without reading it.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Sequence

import duckdb

from ticker.retrieval_types import RankedChunk, Retriever

DEFAULT_K = 20
DEFAULT_KEYWORD_K = 20

# Grammatical function words only. Domain-relevant adjectives such as "new"
# or "current" are deliberately not here -- they carry meaning for queries
# like "newly disclosed litigation" and stripping them would weaken the
# keyword seed's independence from the learned rankers.
_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "and", "or", "to", "with",
    "by", "is", "are", "this", "that", "its", "from", "as", "at", "into",
    "about", "over", "under", "per", "vs", "was", "were", "be", "been",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _keywords(query_text: str) -> list[str]:
    """Significant terms for the keyword-seed substring search.

    Deliberately naive: lowercase, strip punctuation, drop short function
    words, dedupe while preserving order. No stemming and no synonym
    expansion -- this source is supposed to be dumb so its failures are
    independent of the two learned systems'.
    """
    seen: set[str] = set()
    keywords: list[str] = []
    for token in _TOKEN_RE.findall(query_text.lower()):
        if len(token) < 3 or token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords


def keyword_seed_search(
    con: duckdb.DuckDBPyConnection, query_text: str, k: int = DEFAULT_KEYWORD_K
) -> list[RankedChunk]:
    """Rank chunks by how many distinct query keywords appear as a substring.

    A raw ILIKE match count, not a ranking model, and available with no
    index and no trained retriever -- see the module docstring for why this
    crude source still belongs in the pool. Returns [] if the query has no
    keyword after stopword removal (e.g. a query that is nothing but
    function words, which none of ours are, but the empty case should not
    crash a caller).
    """
    keywords = _keywords(query_text)
    if not keywords:
        return []

    # contains(lower(text), ?) rather than `text ILIKE ?`: same case-insensitive
    # substring semantics, but roughly 2-3x faster on this corpus (measured),
    # because ILIKE's pattern engine is doing needless work for a plain
    # substring check with no wildcards of its own.
    case_terms = " + ".join(
        "CASE WHEN contains(lower(text), ?) THEN 1 ELSE 0 END" for _ in keywords
    )
    params: list[object] = list(keywords)
    sql = f"""
        SELECT chunk_id, matches FROM (
            SELECT chunk_id, ({case_terms}) AS matches FROM chunks
        ) scored
        WHERE matches > 0
        ORDER BY matches DESC, chunk_id
        LIMIT ?
    """
    rows = con.execute(sql, params + [k]).fetchall()
    return [RankedChunk(chunk_id=chunk_id, score=float(matches)) for chunk_id, matches in rows]


@dataclass(frozen=True, slots=True)
class PooledQuery:
    query_id: str
    chunk_ids: tuple[str, ...]
    sources: dict[str, int]  # retriever/seed name -> distinct chunks it contributed


def _derive_seed(base_seed: int, query_id: str) -> int:
    """Deterministic per-query seed so pooling is reproducible without every
    query sharing one shuffle."""
    digest = hashlib.sha256(f"{base_seed}:{query_id}".encode()).hexdigest()
    return int(digest[:16], 16)


def pool_query(
    con: duckdb.DuckDBPyConnection,
    query_id: str,
    query_text: str,
    retrievers: Sequence[Retriever] = (),
    *,
    k: int = DEFAULT_K,
    keyword_k: int = DEFAULT_KEYWORD_K,
    seed: int = 0,
) -> PooledQuery:
    """Union top-k from every retriever plus the keyword seed, dedupe by
    chunk_id, shuffle with a seed tied to (seed, query_id)."""
    contributed: dict[str, set[str]] = {}
    pooled: set[str] = set()

    for retriever in retrievers:
        ids = {rc.chunk_id for rc in retriever.search(query_text, k)}
        contributed[retriever.name] = ids
        pooled |= ids

    keyword_ids = {rc.chunk_id for rc in keyword_seed_search(con, query_text, keyword_k)}
    contributed["keyword"] = keyword_ids
    pooled |= keyword_ids

    ordered = sorted(pooled)  # deterministic pre-shuffle order
    random.Random(_derive_seed(seed, query_id)).shuffle(ordered)

    return PooledQuery(
        query_id=query_id,
        chunk_ids=tuple(ordered),
        sources={name: len(ids) for name, ids in contributed.items()},
    )


def pool_queries(
    con: duckdb.DuckDBPyConnection,
    queries: Sequence[tuple[str, str]],
    retrievers: Sequence[Retriever] = (),
    *,
    k: int = DEFAULT_K,
    keyword_k: int = DEFAULT_KEYWORD_K,
    seed: int = 0,
) -> list[PooledQuery]:
    """`queries` is a sequence of (query_id, text) pairs, deliberately not
    tied to the queries.jsonl schema so this stays testable without it."""
    return [
        pool_query(con, query_id, text, retrievers, k=k, keyword_k=keyword_k, seed=seed)
        for query_id, text in queries
    ]
