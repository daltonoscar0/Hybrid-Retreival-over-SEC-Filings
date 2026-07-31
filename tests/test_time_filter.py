"""Time-filtering contract for the data layer.

Written before the query code it exercises. If prior_sentences or
background_sentences change shape, this file changes first.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ticker import db
from ticker.records import Filing, Section, Sentence

TARGET_CIK = 320193
OTHER_CIK = 789019

T = datetime(2023, 6, 1, tzinfo=timezone.utc)
T_MINUS_2 = T - timedelta(days=730)
T_MINUS_1 = T - timedelta(days=365)
T_PLUS_1 = T + timedelta(days=365)


def _filing(accession: str, cik: int, filed_at: datetime) -> Filing:
    return Filing(
        accession=accession,
        cik=cik,
        ticker="TEST",
        form="10-K",
        filed_at=filed_at,
        period_end=None,
        url="https://example.com/" + accession,
    )


def _section(accession: str) -> Section:
    section_id = f"{accession}#PROVISIONAL"
    return Section(
        section_id=section_id,
        accession=accession,
        item="PROVISIONAL",
        text="body",
        char_start=0,
        char_end=4,
    )


def _sentence(section_id: str, filed_at: datetime) -> Sentence:
    return Sentence(
        sentence_id=f"{section_id}#0",
        section_id=section_id,
        ordinal=0,
        text=f"sentence for {section_id}",
        filed_at=filed_at,
    )


@pytest.fixture
def con():
    connection = db.connect(":memory:")
    db.create_schema(connection)
    return connection


@pytest.fixture
def populated(con):
    # target firm: one filing each at t-2, t-1, t, t+1
    for label, filed_at in [
        ("m2", T_MINUS_2),
        ("m1", T_MINUS_1),
        ("t0", T),
        ("p1", T_PLUS_1),
    ]:
        accession = f"acc-target-{label}"
        db.insert_filing(con, _filing(accession, TARGET_CIK, filed_at))
        section = _section(accession)
        db.insert_section(con, section)
        db.insert_sentence(con, _sentence(section.section_id, filed_at))

    # a second firm, filed at t-1, used to check background exclusion
    other_accession = "acc-other-m1"
    db.insert_filing(con, _filing(other_accession, OTHER_CIK, T_MINUS_1))
    other_section = _section(other_accession)
    db.insert_section(con, other_section)
    db.insert_sentence(con, _sentence(other_section.section_id, T_MINUS_1))

    return con


def _accessions(sentences) -> set[str]:
    return {s.section_id.split("#")[0] for s in sentences}


def test_prior_sentences_returns_only_t2_and_t1(populated):
    result = db.prior_sentences(populated, TARGET_CIK, T)
    assert _accessions(result) == {"acc-target-m2", "acc-target-m1"}
    assert len(result) == 2


def test_filing_at_exactly_as_of_is_excluded(populated):
    # catches the off-by-one that leaks a document into its own background model
    result = db.prior_sentences(populated, TARGET_CIK, T)
    assert "acc-target-t0" not in _accessions(result)


def test_filing_after_as_of_is_excluded(populated):
    result = db.prior_sentences(populated, TARGET_CIK, T)
    assert "acc-target-p1" not in _accessions(result)


def test_as_of_before_first_filing_returns_empty_without_raising(populated):
    result = db.prior_sentences(populated, TARGET_CIK, T_MINUS_2 - timedelta(days=1))
    assert result == []


def test_prior_sentences_for_unknown_cik_returns_empty(populated):
    result = db.prior_sentences(populated, 999999, T_PLUS_1)
    assert result == []


def test_background_excludes_target_cik(populated):
    result = db.background_sentences(populated, TARGET_CIK, T_PLUS_1 + timedelta(days=1))
    assert _accessions(result) == {"acc-other-m1"}
    assert all("acc-target" not in acc for acc in _accessions(result))


def test_background_still_respects_time_filter(populated):
    # the other firm's filing is at t-1; as_of == t-1 must exclude it too
    result = db.background_sentences(populated, TARGET_CIK, T_MINUS_1)
    assert result == []


def test_naive_as_of_raises_on_prior_sentences(populated):
    naive = datetime(2023, 6, 1)
    with pytest.raises(ValueError):
        db.prior_sentences(populated, TARGET_CIK, naive)


def test_naive_as_of_raises_on_background_sentences(populated):
    naive = datetime(2023, 6, 1)
    with pytest.raises(ValueError):
        db.background_sentences(populated, TARGET_CIK, naive)


def test_naive_filed_at_raises_on_filing_insert(con):
    naive_filing = _filing("acc-naive", TARGET_CIK, datetime(2023, 1, 1))
    with pytest.raises(ValueError):
        db.insert_filing(con, naive_filing)


def test_naive_filed_at_raises_on_sentence_insert(con):
    accession = "acc-naive-sentence"
    db.insert_filing(con, _filing(accession, TARGET_CIK, T_MINUS_1))
    section = _section(accession)
    db.insert_section(con, section)
    naive_sentence = Sentence(
        sentence_id=f"{section.section_id}#0",
        section_id=section.section_id,
        ordinal=0,
        text="naive",
        filed_at=datetime(2023, 1, 1),
    )
    with pytest.raises(ValueError):
        db.insert_sentence(con, naive_sentence)
