"""BM25 retriever: tokenization on financial text, empty-query handling,
determinism across rebuilds, and chunk_id round-tripping from index position
back to the database.

Index builds here go through `scripts.index.build_index` against a small
on-disk DuckDB (not `:memory:` -- `build_index` opens the db path itself in a
fresh read-only connection, which needs a real file), never against
`data/ticker.duckdb` or anything under `data/qrels`/`data/spotcheck`.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ticker import db
from ticker.records import Chunk, Filing, Section
from ticker.retrieval.bm25 import BM25_VARIANT, load_retriever, tokenize_texts

# scripts/ is not a package; import scripts/index.py by file path.
_SPEC = importlib.util.spec_from_file_location(
    "ticker_scripts_index", Path(__file__).resolve().parents[1] / "scripts" / "index.py"
)
index_script = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = index_script
_SPEC.loader.exec_module(index_script)

FILED_AT = datetime(2024, 1, 1, tzinfo=timezone.utc)


# --- tokenization on financial text ---------------------------------------


def test_tokenize_lowercases_and_drops_stopwords():
    tokens = tokenize_texts(
        ["Microchip Technology Inc. reported net sales for the quarter."],
        show_progress=False,
    )
    strings = list(tokens.vocab.keys())
    assert "microchip" in strings  # lowercased
    assert "inc" in strings  # trailing period stripped by the word-boundary pattern
    assert "for" not in strings  # stopword
    assert "the" not in strings  # stopword


def test_tokenize_splits_dollar_amounts_and_drops_single_char_fragments():
    tokens = tokenize_texts(
        ["Net sales were $1.281 billion for the U.S. market."], show_progress=False
    )
    strings = set(tokens.vocab.keys())
    # "$1.281" -> the leading "1" is a single word-char, below the \w\w+
    # threshold, so only "281" survives; "U.S." similarly loses both
    # single-letter fragments entirely.
    assert "281" in strings
    assert "billion" in strings
    assert "u" not in strings
    assert "s" not in strings


def test_tokenize_all_stopwords_yields_empty_token_list():
    tokens = tokenize_texts(["The of and"], show_progress=False)
    assert tokens.ids == [[]]


# --- index build + retriever ------------------------------------------------


def _seed_corpus(db_path: Path, chunks: list[tuple[str, str]]) -> None:
    con = db.connect(str(db_path))
    try:
        db.create_schema(con)
        db.insert_filing(
            con,
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
            con,
            Section(
                section_id="acc-1#1A",
                accession="acc-1",
                item="1A",
                text="body",
                char_start=0,
                char_end=4,
            ),
        )
        for chunk_id, text in chunks:
            db.insert_chunk(
                con, Chunk(chunk_id=chunk_id, section_id="acc-1#1A", sentence_ids=(), text=text)
            )
    finally:
        con.close()


_CHUNKS = [
    ("z-chunk", "goodwill impairment charge recognized in the current quarter"),
    ("a-chunk", "customer concentration risk in the semiconductor segment"),
    ("m-chunk", "foreign exchange impact on reported revenue for the period"),
]


@pytest.fixture
def built_index(tmp_path):
    db_path = tmp_path / "corpus.duckdb"
    _seed_corpus(db_path, _CHUNKS)
    out_dir = tmp_path / "index"
    manifest = index_script.build_index(db_path, out_dir, show_progress=False)
    return out_dir, manifest


def test_search_returns_ranked_chunks_and_uses_kamphuis_lucene_variant(built_index):
    out_dir, manifest = built_index
    assert manifest["method"] == BM25_VARIANT == "lucene"

    retriever = load_retriever(out_dir)
    results = retriever.search("customer concentration risk", k=3)

    assert results
    assert results[0].chunk_id == "a-chunk"
    # best match first
    assert all(a.score >= b.score for a, b in zip(results, results[1:]))


def test_search_empty_query_does_not_crash(built_index):
    out_dir, _ = built_index
    retriever = load_retriever(out_dir)

    results = retriever.search("", k=3)
    assert len(results) == 3
    assert all(rc.score == 0.0 for rc in results)

    results_all_stopwords = retriever.search("the of and", k=3)
    assert len(results_all_stopwords) == 3
    assert all(rc.score == 0.0 for rc in results_all_stopwords)


def test_search_k_is_capped_to_corpus_size(built_index):
    out_dir, _ = built_index
    retriever = load_retriever(out_dir)
    assert len(retriever) == 3
    results = retriever.search("goodwill impairment", k=1000)
    assert len(results) == 3


def test_scores_deterministic_across_two_index_builds(tmp_path):
    db_path = tmp_path / "corpus.duckdb"
    _seed_corpus(db_path, _CHUNKS)

    out_a = tmp_path / "index_a"
    out_b = tmp_path / "index_b"
    index_script.build_index(db_path, out_a, show_progress=False)
    index_script.build_index(db_path, out_b, show_progress=False)

    retriever_a = load_retriever(out_a)
    retriever_b = load_retriever(out_b)

    for query in ("customer concentration risk", "goodwill impairment", "revenue"):
        results_a = retriever_a.search(query, k=3)
        results_b = retriever_b.search(query, k=3)
        assert [(r.chunk_id, r.score) for r in results_a] == [
            (r.chunk_id, r.score) for r in results_b
        ]


def test_chunk_ids_round_trip_from_index_position_to_database(built_index):
    """A result's chunk_id, however it sorts within bm25s's internal
    (alphabetically-ordered-by-build) position array, must name a real row
    in the corpus this index was built from -- this is the whole point of
    the chunk_ids.json sidecar bm25s itself does not provide."""
    out_dir, manifest = built_index
    retriever = load_retriever(out_dir)

    known_ids = {chunk_id for chunk_id, _ in _CHUNKS}
    assert set(retriever._chunk_ids) == known_ids
    # build order is `ORDER BY chunk_id`, independent of insertion order
    assert retriever._chunk_ids == sorted(known_ids)

    for query in ("customer concentration", "goodwill impairment", "foreign exchange"):
        for result in retriever.search(query, k=3):
            assert result.chunk_id in known_ids
