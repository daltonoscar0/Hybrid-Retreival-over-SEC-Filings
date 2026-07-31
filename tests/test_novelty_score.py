"""The novelty contrast and the guardrails around it.

Three things are being pinned here. That the score is a difference of two
surprisals and nothing else. That a z-scored value cannot reach a
cross-document number, which invariant 3 requires and which a comment cannot
enforce. And that the background model is sector-matched, excludes the firm,
and stops strictly before `as_of`, which is the join most likely to be wrong
in a way that quietly inflates Phase 5's agreement number.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ticker import db
from ticker.novelty.kneser_ney import fit, shared_vocabulary
from ticker.novelty.score import (
    DisplayNovelties,
    DisplayNovelty,
    DocumentNovelty,
    RawNovelty,
    build_models,
    document_novelty_index,
    novelty_raw,
    novelty_zscored_for_display,
    score_document,
)
from ticker.records import Filing, Section, Sentence

AS_OF = datetime(2024, 1, 1, tzinfo=timezone.utc)
EARLY = AS_OF - timedelta(days=400)
LATER = AS_OF + timedelta(days=10)

SECTOR = "semiconductors"
OTHER_SECTOR = "regional_banks"
FIRM_CIK = 100
PEER_CIK = 200
OUTSIDER_CIK = 300
NEWCOMER_CIK = 400

FIRM_TEXTS = [
    "our wafer fabrication yields improved firmonly",
    "our wafer fabrication capacity expanded firmonly",
]
PEER_TEXTS = ["peer wafer capacity expanded peeronly"]

SCORED = Sentence(
    sentence_id="scored-0",
    section_id="acc-firm-now#ITEM7",
    ordinal=0,
    text="our wafer fabrication yields improved",
    filed_at=AS_OF,
)


def _sentences(texts, filed_at, prefix):
    return [
        Sentence(
            sentence_id=f"{prefix}-{i}",
            section_id=f"{prefix}-section",
            ordinal=i,
            text=text,
            filed_at=filed_at,
        )
        for i, text in enumerate(texts)
    ]


def _model_pair(firm_texts=FIRM_TEXTS, background_texts=PEER_TEXTS, as_of=AS_OF):
    firm_sentences = _sentences(firm_texts, EARLY, "firm")
    background_sentences = _sentences(background_texts, EARLY, "peer")
    vocab = shared_vocabulary(firm_sentences, background_sentences)
    background_lm = fit(background_sentences, as_of=as_of, vocabulary=vocab)
    firm_lm = fit(firm_sentences, as_of=as_of, vocabulary=vocab, parent=background_lm)
    return firm_lm, background_lm


# --- the contrast, invariant 2 ---------------------------------------------


def test_novelty_raw_is_the_difference_of_two_surprisals():
    firm_lm, background_lm = _model_pair()
    assert novelty_raw(SCORED, firm_lm, background_lm) == pytest.approx(
        firm_lm.surprisal(SCORED.text) - background_lm.surprisal(SCORED.text)
    )


def test_novelty_is_exactly_zero_when_the_two_models_are_the_same_object():
    _, background_lm = _model_pair()
    assert novelty_raw(SCORED, background_lm, background_lm) == 0.0


def test_novelty_is_zero_when_the_two_models_are_fit_on_the_same_sentences():
    sentences = _sentences(FIRM_TEXTS, EARLY, "firm")
    vocab = shared_vocabulary(sentences)
    left = fit(sentences, as_of=AS_OF, vocabulary=vocab)
    right = fit(sentences, as_of=AS_OF, vocabulary=vocab)
    assert novelty_raw(SCORED, left, right) == pytest.approx(0.0, abs=1e-12)


def test_a_firm_that_repeats_itself_scores_below_a_firm_that_does_not():
    boilerplate = "our wafer fabrication yields improved firmonly"
    firm_lm, background_lm = _model_pair()
    repeated = Sentence(
        sentence_id="repeat-0",
        section_id="acc-firm-now#ITEM7",
        ordinal=0,
        text=boilerplate,
        filed_at=AS_OF,
    )
    fresh = Sentence(
        sentence_id="fresh-0",
        section_id="acc-firm-now#ITEM7",
        ordinal=1,
        text="peer wafer capacity expanded peeronly",
        filed_at=AS_OF,
    )
    assert novelty_raw(repeated, firm_lm, background_lm) < novelty_raw(
        fresh, firm_lm, background_lm
    )


def test_models_fit_under_different_as_of_are_rejected():
    firm_lm, _ = _model_pair()
    _, other_background = _model_pair(as_of=AS_OF + timedelta(days=1))
    with pytest.raises(ValueError, match="as_of"):
        novelty_raw(SCORED, firm_lm, other_background)


def test_models_over_different_vocabularies_are_rejected():
    firm_lm, _ = _model_pair()
    unrelated = fit(_sentences(["deposit balances grew"], EARLY, "x"), as_of=AS_OF)
    with pytest.raises(ValueError, match="vocabular"):
        novelty_raw(SCORED, firm_lm, unrelated)


def test_scoring_a_sentence_from_before_the_cutoff_is_rejected():
    # A sentence filed before as_of may be in the training data of the model
    # scoring it. That is the document reading its own answer, one layer up
    # from the check inside fit.
    firm_lm, background_lm = _model_pair()
    stale = Sentence(
        sentence_id="stale-0",
        section_id="acc-firm-old#ITEM7",
        ordinal=0,
        text="our wafer fabrication yields improved",
        filed_at=EARLY,
    )
    with pytest.raises(ValueError, match="stale-0"):
        novelty_raw(stale, firm_lm, background_lm)


def test_scoring_a_sentence_filed_exactly_at_the_cutoff_is_allowed():
    firm_lm, background_lm = _model_pair()
    assert isinstance(novelty_raw(SCORED, firm_lm, background_lm), float)


# --- document scoring ------------------------------------------------------


def _document():
    firm_lm, background_lm = _model_pair()
    sentences = [
        Sentence(
            sentence_id=f"doc-{i}",
            section_id="acc-firm-now#ITEM7",
            ordinal=i,
            text=text,
            filed_at=AS_OF,
        )
        for i, text in enumerate(
            [
                "our wafer fabrication yields improved",
                "peer wafer capacity expanded peeronly",
                "litigation was filed against a subsidiary",
            ]
        )
    ]
    return score_document("acc-firm-now", sentences, firm_lm, background_lm)


def test_score_document_returns_one_raw_score_per_sentence_in_order():
    doc = _document()
    assert [s.sentence_id for s in doc.scores] == ["doc-0", "doc-1", "doc-2"]
    assert all(isinstance(s, RawNovelty) for s in doc.scores)
    assert doc.as_of == AS_OF


def test_document_novelty_index_is_the_mean_of_the_raw_contrasts():
    doc = _document()
    expected = sum(s.contrast for s in doc.scores) / len(doc.scores)
    assert document_novelty_index(doc) == pytest.approx(expected)


# --- z-scoring is for display only, invariant 3 ----------------------------


def test_zscores_have_zero_mean_and_unit_deviation_within_the_document():
    display = novelty_zscored_for_display(_document())
    values = [d.z for d in display.values]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    assert mean == pytest.approx(0.0, abs=1e-12)
    assert variance == pytest.approx(1.0)


def test_zscores_are_zero_when_every_sentence_scores_the_same():
    doc = DocumentNovelty(
        document_id="acc-flat",
        as_of=AS_OF,
        scores=(RawNovelty("a", 1.5), RawNovelty("b", 1.5)),
    )
    assert [d.z for d in novelty_zscored_for_display(doc).values] == [0.0, 0.0]


def test_display_values_carry_no_raw_contrast_to_reach_through():
    display = novelty_zscored_for_display(_document())
    assert isinstance(display, DisplayNovelties)
    sample = display.values[0]
    assert isinstance(sample, DisplayNovelty)
    assert not hasattr(sample, "contrast")


def test_aggregating_the_zscored_values_is_a_typeerror():
    # Invariant 3. Feeding within-document z-scores to a cross-document
    # aggregate produces a plausible number that measures nothing, so the
    # aggregator refuses the type rather than averaging it.
    display = novelty_zscored_for_display(_document())
    with pytest.raises(TypeError, match="z-scored"):
        document_novelty_index(display)


def test_aggregating_a_bare_sequence_of_display_values_is_a_typeerror():
    display = novelty_zscored_for_display(_document())
    with pytest.raises(TypeError):
        document_novelty_index(display.values)


def test_zscoring_something_that_is_not_a_document_is_a_typeerror():
    with pytest.raises(TypeError):
        novelty_zscored_for_display([RawNovelty("a", 1.0)])


def test_document_novelty_index_needs_at_least_one_sentence():
    with pytest.raises(ValueError):
        document_novelty_index(DocumentNovelty("acc-empty", AS_OF, ()))


# --- the builder: sector-matched background, firm excluded, strict cutoff --


def _insert(con, accession, cik, sector, filed_at, texts):
    db.insert_filing(
        con,
        Filing(
            accession=accession,
            cik=cik,
            ticker=f"T{cik}",
            sector=sector,
            form="10-K",
            filed_at=filed_at,
            period_end=None,
            url=f"https://example.com/{accession}",
        ),
    )
    section_id = f"{accession}#ITEM7"
    db.insert_section(
        con,
        Section(
            section_id=section_id,
            accession=accession,
            item="ITEM7",
            text=" ".join(texts),
            char_start=0,
            char_end=1,
        ),
    )
    for ordinal, text in enumerate(texts):
        db.insert_sentence(
            con,
            Sentence(
                sentence_id=f"{section_id}#{ordinal}",
                section_id=section_id,
                ordinal=ordinal,
                text=text,
                filed_at=filed_at,
            ),
        )


@pytest.fixture
def con():
    connection = db.connect(":memory:")
    db.create_schema(connection)
    _insert(connection, "acc-firm-early", FIRM_CIK, SECTOR, EARLY, FIRM_TEXTS)
    _insert(connection, "acc-peer-early", PEER_CIK, SECTOR, EARLY, PEER_TEXTS)
    _insert(
        connection, "acc-peer-cutoff", PEER_CIK, SECTOR, AS_OF, ["atcutoff disclosure"]
    )
    _insert(
        connection,
        "acc-outsider",
        OUTSIDER_CIK,
        OTHER_SECTOR,
        EARLY,
        ["deposit balances grew outsideronly"],
    )
    _insert(
        connection,
        "acc-firm-now",
        FIRM_CIK,
        SECTOR,
        AS_OF,
        ["the current filing must not train anything"],
    )
    _insert(
        connection, "acc-newcomer", NEWCOMER_CIK, SECTOR, LATER, ["newcomer first words"]
    )
    return connection


def test_background_contains_sector_peers(con):
    _, background_lm = build_models(con, cik=FIRM_CIK, sector=SECTOR, as_of=AS_OF)
    assert background_lm.raw_unigram_count("peeronly") == 1


def test_background_excludes_the_firm_being_scored(con):
    firm_lm, background_lm = build_models(con, cik=FIRM_CIK, sector=SECTOR, as_of=AS_OF)
    assert firm_lm.raw_unigram_count("firmonly") == 2
    # In the shared vocabulary, so its absence is a count of zero rather than
    # an out-of-vocabulary miss: the firm did not train its own background.
    assert "firmonly" in background_lm.vocabulary
    assert background_lm.raw_unigram_count("firmonly") == 0


def test_background_excludes_other_sectors(con):
    _, background_lm = build_models(con, cik=FIRM_CIK, sector=SECTOR, as_of=AS_OF)
    assert "outsideronly" not in background_lm.vocabulary


def test_background_stops_strictly_before_as_of(con):
    _, background_lm = build_models(con, cik=FIRM_CIK, sector=SECTOR, as_of=AS_OF)
    assert "atcutoff" not in background_lm.vocabulary


def test_firm_model_stops_strictly_before_as_of(con):
    firm_lm, _ = build_models(con, cik=FIRM_CIK, sector=SECTOR, as_of=AS_OF)
    assert firm_lm.raw_unigram_count("must") == 0


def test_builder_wires_the_background_in_as_the_firm_parent(con):
    firm_lm, background_lm = build_models(con, cik=FIRM_CIK, sector=SECTOR, as_of=AS_OF)
    assert firm_lm.parent is background_lm
    assert firm_lm.vocabulary == background_lm.vocabulary
    assert firm_lm.as_of == background_lm.as_of == AS_OF


def test_builder_rejects_a_naive_as_of(con):
    with pytest.raises(ValueError, match="timezone-aware"):
        build_models(con, cik=FIRM_CIK, sector=SECTOR, as_of=datetime(2024, 1, 1))


def test_builder_rejects_a_cik_with_no_filings(con):
    with pytest.raises(ValueError, match="999"):
        build_models(con, cik=999, sector=SECTOR, as_of=AS_OF)


def test_builder_rejects_a_sector_the_firm_does_not_belong_to(con):
    with pytest.raises(ValueError, match=OTHER_SECTOR):
        build_models(con, cik=FIRM_CIK, sector=OTHER_SECTOR, as_of=AS_OF)


def test_firm_with_no_prior_filings_scores_every_sentence_at_zero(con):
    # Nothing of this firm's own exists before its first filing, so the firm
    # model is its own background and the contrast is identically zero.
    # Silence, not a signal, and the sector model still has data.
    firm_lm, background_lm = build_models(
        con, cik=NEWCOMER_CIK, sector=SECTOR, as_of=LATER
    )
    assert firm_lm.token_count == 0
    assert background_lm.token_count > 0
    first = Sentence(
        sentence_id="first-0",
        section_id="acc-newcomer#ITEM7",
        ordinal=0,
        text="newcomer first words",
        filed_at=LATER,
    )
    assert novelty_raw(first, firm_lm, background_lm) == pytest.approx(0.0, abs=1e-12)
