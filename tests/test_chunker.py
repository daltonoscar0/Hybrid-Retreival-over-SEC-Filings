"""4-sentence window / stride-2 chunker contract.

Chunks must carry the exact sentence_ids that built them -- that is the only
thing that lets a per-sentence novelty score map onto a chunk without
recomputing anything -- so every case here checks sentence_ids, not just chunk
count.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ticker.chunker import chunk_sentences
from ticker.records import Sentence

FILED_AT = datetime(2024, 1, 1, tzinfo=timezone.utc)
SECTION_ID = "acc-1#1A"


def _sentences(n: int) -> list[Sentence]:
    return [
        Sentence(
            sentence_id=f"{SECTION_ID}#{i}",
            section_id=SECTION_ID,
            ordinal=i,
            text=f"sentence {i}",
            filed_at=FILED_AT,
        )
        for i in range(n)
    ]


def test_empty_input_returns_no_chunks():
    assert chunk_sentences([]) == []


def test_exactly_one_window_of_sentences_makes_one_chunk():
    chunks = chunk_sentences(_sentences(4))
    assert len(chunks) == 1
    assert chunks[0].sentence_ids == tuple(f"{SECTION_ID}#{i}" for i in range(4))


def test_fewer_than_one_window_still_makes_one_chunk():
    chunks = chunk_sentences(_sentences(2))
    assert len(chunks) == 1
    assert chunks[0].sentence_ids == (f"{SECTION_ID}#0", f"{SECTION_ID}#1")


def test_six_sentences_makes_two_overlapping_chunks_at_stride_two():
    chunks = chunk_sentences(_sentences(6))
    assert [c.sentence_ids for c in chunks] == [
        tuple(f"{SECTION_ID}#{i}" for i in range(0, 4)),
        tuple(f"{SECTION_ID}#{i}" for i in range(2, 6)),
    ]


def test_five_sentences_tail_chunk_is_short():
    chunks = chunk_sentences(_sentences(5))
    assert [c.sentence_ids for c in chunks] == [
        tuple(f"{SECTION_ID}#{i}" for i in range(0, 4)),
        tuple(f"{SECTION_ID}#{i}" for i in range(2, 5)),
    ]


def test_chunk_text_is_space_joined_sentence_text_in_order():
    chunks = chunk_sentences(_sentences(4))
    assert chunks[0].text == "sentence 0 sentence 1 sentence 2 sentence 3"


def test_chunk_id_is_stable_and_derived_from_section_and_start():
    chunks = chunk_sentences(_sentences(6))
    assert chunks[0].chunk_id == f"{SECTION_ID}#chunk0"
    assert chunks[1].chunk_id == f"{SECTION_ID}#chunk2"


def test_chunk_section_id_matches_input_sentences():
    chunks = chunk_sentences(_sentences(4))
    assert all(c.section_id == SECTION_ID for c in chunks)


def test_mixed_section_ids_raises():
    sentences = _sentences(2) + [
        Sentence(
            sentence_id="other-section#0",
            section_id="other-section",
            ordinal=0,
            text="from a different section",
            filed_at=FILED_AT,
        )
    ]
    with pytest.raises(ValueError):
        chunk_sentences(sentences)
