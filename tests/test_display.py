"""Display layer: within-document banding, raw aggregation, and the novelty
rerank toggle. The invariants under test are 3 (z-scores never leave the
document) and 4 (novelty never enters the relevance score).
"""

from __future__ import annotations

import json
from pathlib import Path

from ticker.display import (
    BAND_HIGH,
    BAND_LOW,
    BAND_MID,
    SentenceView,
    build_sentence_views,
    document_novelty,
    load_novelty,
    novelty_bands,
    rerank_by_novelty,
)


def test_bands_are_relative_to_the_document_not_absolute():
    low_doc = novelty_bands([-10.0, -10.0, -10.0, -2.0])
    high_doc = novelty_bands([2.0, 2.0, 2.0, 10.0])
    # The same band falls out of both, because each is z-scored against its
    # own document. An absolute threshold would call the second document
    # entirely novel and the first entirely boilerplate.
    assert low_doc[3] == high_doc[3] == BAND_HIGH


def test_bands_return_no_floats():
    bands = novelty_bands([-1.0, 0.0, 1.0, 5.0])
    assert all(b is None or isinstance(b, str) for b in bands)


def test_bands_need_at_least_two_scored_sentences():
    assert novelty_bands([None, 1.0, None]) == [None, None, None]


def test_bands_handle_zero_variance_without_dividing_by_zero():
    assert novelty_bands([3.0, 3.0, 3.0]) == [None, None, None]


def test_bands_pass_through_unscored_sentences_as_none():
    bands = novelty_bands([-2.0, None, 4.0])
    assert bands[1] is None


def test_low_and_mid_bands_separate():
    bands = novelty_bands([-5.0, -5.0, -5.0, -5.0, 0.0, 5.0])
    assert BAND_LOW in bands and BAND_HIGH in bands


def test_document_novelty_averages_raw_and_ignores_missing():
    assert document_novelty([-2.0, None, -4.0]) == -3.0


def test_document_novelty_is_none_when_nothing_is_scored():
    assert document_novelty([None, None]) is None


def test_build_sentence_views_pairs_ids_texts_and_scores():
    views = build_sentence_views(
        ["s0", "s1"], ["first", "second"], {"s0": -1.0, "s1": 5.0}
    )
    assert [v.text for v in views] == ["first", "second"]
    assert views[1].raw == 5.0
    assert isinstance(views[0], SentenceView)


def test_rerank_with_zero_weight_is_the_identity():
    ranked = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    out = rerank_by_novelty(ranked, {"a": -5.0, "b": 1.0, "c": 0.0}, weight=0.0)
    assert [c for c, _ in out] == ["a", "b", "c"]


def test_rerank_promotes_the_novel_chunk():
    ranked = [("a", 0.9), ("b", 0.8)]
    out = rerank_by_novelty(ranked, {"a": -5.0, "b": 5.0}, weight=1.0)
    assert [c for c, _ in out] == ["b", "a"]


def test_rerank_preserves_the_original_relevance_scores():
    ranked = [("a", 0.9), ("b", 0.8)]
    out = rerank_by_novelty(ranked, {"a": -5.0, "b": 5.0}, weight=1.0)
    assert dict(out) == {"a": 0.9, "b": 0.8}


def test_a_top_ranked_unscored_chunk_is_not_displaced_by_the_rerank():
    # A corpus scored for one form only must not push every other form to the
    # bottom. An unscored chunk is unknown, not un-novel.
    ranked = [("unscored", 0.9), ("scored_high", 0.8), ("scored_low", 0.7)]
    out = rerank_by_novelty(
        ranked, {"unscored": None, "scored_high": 5.0, "scored_low": -5.0}, weight=1.0
    )
    assert out[0][0] == "unscored"


def test_unscored_chunks_interleave_by_relevance_rather_than_sinking_as_a_block():
    ranked = [(f"c{i}", 1.0 - i / 10) for i in range(6)]
    # Alternating scored and unscored, with novelty running opposite to
    # relevance so a full-weight rerank has to move things.
    chunk_novelty = {"c0": None, "c1": -5.0, "c2": None, "c3": 0.0, "c4": None, "c5": 5.0}
    out = [c for c, _ in rerank_by_novelty(ranked, chunk_novelty, weight=1.0)]
    unscored_positions = [out.index(c) for c in ("c0", "c2", "c4")]
    assert min(unscored_positions) < 3, "unscored chunks were pushed below every scored one"


def test_rerank_of_an_empty_list_is_empty():
    assert rerank_by_novelty([], {}) == []


def test_load_novelty_reads_every_present_file_and_skips_missing(tmp_path: Path):
    (tmp_path / "scores.jsonl").write_text(
        json.dumps({"sentence_id": "s0", "novelty": -1.0}) + "\n"
    )
    (tmp_path / "scores_10-Q.jsonl").write_text(
        json.dumps({"sentence_id": "s1", "novelty": 2.0}) + "\n"
    )
    scores = load_novelty(tmp_path, ("scores.jsonl", "scores_10-Q.jsonl", "absent.jsonl"))
    assert scores == {"s0": -1.0, "s1": 2.0}


def test_load_novelty_on_an_empty_dir_returns_empty(tmp_path: Path):
    assert load_novelty(tmp_path) == {}
