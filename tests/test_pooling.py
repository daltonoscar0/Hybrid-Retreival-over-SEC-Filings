"""Pool construction against stub rankings -- no real retriever needed.

Standard TREC pooling: dedupe the top-k from each source plus the keyword
seed, shuffle so judging order carries no rank/source signal. See
`ticker.pooling`'s module docstring for the justification.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from ticker import db
from ticker.pooling import DEFAULT_K, keyword_seed_search, pool_queries, pool_query
from ticker.records import Chunk, Filing, Section
from ticker.retrieval_types import RankedChunk

FILED_AT = datetime(2024, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class StubRetriever:
    name: str
    ranking: list[RankedChunk]

    def search(self, query: str, k: int) -> list[RankedChunk]:
        return self.ranking[:k]


@pytest.fixture
def con():
    connection = db.connect(":memory:")
    db.create_schema(connection)
    db.insert_filing(
        connection,
        Filing(
            accession="acc-1",
            cik=1,
            ticker="TEST",
            sector="test_sector",
            form="10-K",
            filed_at=FILED_AT,
            period_end=None,
            url="https://example.com/acc-1",
        ),
    )
    db.insert_section(
        connection,
        Section(
            section_id="acc-1#1A",
            accession="acc-1",
            item="1A",
            text="body",
            char_start=0,
            char_end=4,
        ),
    )
    return connection


def _insert_chunk(con, chunk_id: str, text: str) -> None:
    db.insert_chunk(
        con,
        Chunk(chunk_id=chunk_id, section_id="acc-1#1A", sentence_ids=(), text=text),
    )


def test_pool_query_dedupes_across_sources(con):
    _insert_chunk(con, "c1", "customer concentration risk discussion")
    _insert_chunk(con, "c2", "irrelevant text about widgets")
    _insert_chunk(con, "c3", "more customer concentration language")

    bm25 = StubRetriever("bm25", [RankedChunk("c1", 3.0), RankedChunk("c2", 1.0)])
    dense = StubRetriever("dense", [RankedChunk("c2", 0.9), RankedChunk("c3", 0.5)])

    pooled = pool_query(con, "q1", "customer concentration", [bm25, dense])

    assert set(pooled.chunk_ids) == {"c1", "c2", "c3"}
    assert len(pooled.chunk_ids) == len(set(pooled.chunk_ids))


def test_pool_query_with_no_retrievers_uses_keyword_only(con):
    _insert_chunk(con, "c1", "customer concentration risk")
    pooled = pool_query(con, "q1", "customer concentration", retrievers=())
    assert set(pooled.chunk_ids) == {"c1"}
    assert set(pooled.sources.keys()) == {"keyword"}


def test_pool_query_respects_k_limit(con):
    ranking = [RankedChunk(f"c{i}", float(-i)) for i in range(30)]
    for rc in ranking:
        _insert_chunk(con, rc.chunk_id, "filler text with no keyword overlap")
    bm25 = StubRetriever("bm25", ranking)

    pooled = pool_query(con, "q1", "zzz_no_match_zzz", [bm25], k=DEFAULT_K)

    assert len(pooled.chunk_ids) == DEFAULT_K
    assert pooled.sources["bm25"] == DEFAULT_K


def test_pool_query_shuffle_is_seed_and_query_dependent(con):
    ranking = [RankedChunk(f"c{i}", float(-i)) for i in range(20)]
    for rc in ranking:
        _insert_chunk(con, rc.chunk_id, "filler")
    bm25 = StubRetriever("bm25", ranking)

    same_seed_a = pool_query(con, "q1", "zzz", [bm25], seed=42)
    same_seed_b = pool_query(con, "q1", "zzz", [bm25], seed=42)
    assert same_seed_a.chunk_ids == same_seed_b.chunk_ids

    different_query = pool_query(con, "q2", "zzz", [bm25], seed=42)
    assert different_query.chunk_ids != same_seed_a.chunk_ids
    assert set(different_query.chunk_ids) == set(same_seed_a.chunk_ids)

    different_seed = pool_query(con, "q1", "zzz", [bm25], seed=7)
    assert different_seed.chunk_ids != same_seed_a.chunk_ids


def test_pool_query_order_is_not_rank_order(con):
    # ranking is c0..c19 best-to-worst; the pooled order should not equal it,
    # since that would let a judge read rank straight off judging position.
    ranking = [RankedChunk(f"c{i}", float(-i)) for i in range(20)]
    for rc in ranking:
        _insert_chunk(con, rc.chunk_id, "filler")
    bm25 = StubRetriever("bm25", ranking)

    pooled = pool_query(con, "q1", "zzz", [bm25], seed=1)
    assert pooled.chunk_ids != tuple(rc.chunk_id for rc in ranking)


def test_keyword_seed_search_matches_and_ranks_by_count(con):
    _insert_chunk(con, "c1", "goodwill was reviewed and no charge resulted this quarter")
    _insert_chunk(con, "c2", "goodwill impairment charge recorded this quarter")
    _insert_chunk(con, "c3", "entirely unrelated risk factor language")

    results = keyword_seed_search(con, "goodwill impairment charge", k=10)
    ids = [rc.chunk_id for rc in results]

    assert "c3" not in ids
    assert ids[0] == "c2"  # matches all three keywords vs c1's two ("goodwill", "charge")
    assert set(ids) == {"c1", "c2"}


def test_keyword_seed_search_empty_after_stopword_removal_returns_empty(con):
    _insert_chunk(con, "c1", "some text")
    assert keyword_seed_search(con, "the of and", k=10) == []


def test_pool_queries_processes_every_query_in_order(con):
    _insert_chunk(con, "c1", "customer concentration risk")
    _insert_chunk(con, "c2", "goodwill impairment charge")

    pooled = pool_queries(
        con,
        [("q1", "customer concentration"), ("q2", "goodwill impairment")],
    )

    assert [p.query_id for p in pooled] == ["q1", "q2"]
    assert set(pooled[0].chunk_ids) == {"c1"}
    assert set(pooled[1].chunk_ids) == {"c2"}
