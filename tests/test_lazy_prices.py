"""Contract for the Lazy Prices baseline.

The diff labels here become the silver standard PLAN section 5.1 validates
the surprisal measure against. A label that is wrong in a systematic way
does not fail loudly downstream, it produces a plausible AUC, so the cases
below pin the label boundaries rather than sampling them: an insertion in
the middle of a section is the one that would quietly poison the number, by
shifting every later sentence one position and reading the whole tail as
edited.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ticker import db
from ticker.novelty.lazy_prices import (
    INSERTED,
    MODIFIED,
    UNCHANGED,
    align_prior_section,
    compare_sections,
    diff_sentences,
    normalized_levenshtein_distance,
    tfidf_cosine_similarity,
    tokenize,
)
from ticker.records import Filing, Section

PRIOR_SENTENCES = [
    "The Company operates three semiconductor fabrication facilities in Oregon.",
    "Revenue from our largest customer represented 18.4% of total revenue in fiscal 2021.",
    "We face competition from larger integrated device manufacturers with greater resources.",
    "Our products are sold through a network of distributors and direct sales personnel.",
]

REWRITTEN_SENTENCES = [
    "Regulatory examinations by banking supervisors occur annually.",
    "Deposit balances declined during the fourth quarter.",
    "Branch closures reduced occupancy expense materially.",
    "Credit quality metrics improved across every loan category.",
]


def _labels(diffs):
    return [d.label for d in diffs]


def test_tokenizer_keeps_numbers_whole_and_drops_punctuation():
    assert tokenize("Revenue rose to $1,500.7 million, or 18.4%.") == [
        "revenue", "rose", "to", "1,500.7", "million", "or", "18.4",
    ]


def test_identical_sections_score_zero_distance_and_unit_cosine():
    result = compare_sections(PRIOR_SENTENCES, PRIOR_SENTENCES)
    assert result.levenshtein_distance == 0.0
    assert result.tfidf_cosine_similarity == pytest.approx(1.0)


def test_identical_sections_label_every_sentence_unchanged():
    result = compare_sections(PRIOR_SENTENCES, PRIOR_SENTENCES)
    assert _labels(result.sentence_diffs) == [UNCHANGED] * len(PRIOR_SENTENCES)
    assert [d.best_prior_ordinal for d in result.sentence_diffs] == [0, 1, 2, 3]
    assert all(d.best_similarity == 1.0 for d in result.sentence_diffs)


def test_wholly_rewritten_section_labels_every_sentence_inserted():
    result = compare_sections(REWRITTEN_SENTENCES, PRIOR_SENTENCES)
    assert _labels(result.sentence_diffs) == [INSERTED] * len(REWRITTEN_SENTENCES)


def test_wholly_rewritten_section_scores_near_maximum_change():
    result = compare_sections(REWRITTEN_SENTENCES, PRIOR_SENTENCES)
    assert result.levenshtein_distance > 0.9
    assert result.tfidf_cosine_similarity < 0.1


def test_one_word_edit_labels_exactly_that_sentence_modified():
    current = list(PRIOR_SENTENCES)
    current[1] = current[1].replace("largest", "second-largest")
    diffs = diff_sentences(current, PRIOR_SENTENCES)
    assert _labels(diffs) == [UNCHANGED, MODIFIED, UNCHANGED, UNCHANGED]
    assert diffs[1].best_prior_ordinal == 1
    assert diffs[1].best_similarity > 0.8


def test_number_change_alone_is_a_modification():
    current = list(PRIOR_SENTENCES)
    current[1] = current[1].replace("18.4%", "23.9%").replace("2021", "2022")
    diffs = diff_sentences(current, PRIOR_SENTENCES)
    assert _labels(diffs) == [UNCHANGED, MODIFIED, UNCHANGED, UNCHANGED]


def test_punctuation_only_change_is_unchanged():
    current = list(PRIOR_SENTENCES)
    current[0] = "The Company operates three semiconductor fabrication facilities in Oregon"
    diffs = diff_sentences(current, PRIOR_SENTENCES)
    assert _labels(diffs) == [UNCHANGED] * 4


def test_insertion_in_the_middle_does_not_cascade_modified_labels():
    inserted = "In March 2022 we acquired a test and assembly operation in Malaysia."
    current = PRIOR_SENTENCES[:2] + [inserted] + PRIOR_SENTENCES[2:]
    diffs = diff_sentences(current, PRIOR_SENTENCES)
    assert _labels(diffs) == [UNCHANGED, UNCHANGED, INSERTED, UNCHANGED, UNCHANGED]
    assert MODIFIED not in _labels(diffs)


def test_insertion_in_the_middle_keeps_the_prior_ordinals_aligned():
    inserted = "In March 2022 we acquired a test and assembly operation in Malaysia."
    current = PRIOR_SENTENCES[:2] + [inserted] + PRIOR_SENTENCES[2:]
    diffs = diff_sentences(current, PRIOR_SENTENCES)
    assert [d.best_prior_ordinal for d in diffs] == [0, 1, None, 2, 3]


def test_deleted_sentence_leaves_the_survivors_unchanged():
    current = PRIOR_SENTENCES[:1] + PRIOR_SENTENCES[2:]
    diffs = diff_sentences(current, PRIOR_SENTENCES)
    assert _labels(diffs) == [UNCHANGED, UNCHANGED, UNCHANGED]
    assert [d.best_prior_ordinal for d in diffs] == [0, 2, 3]


def test_repeated_boilerplate_sentence_still_matches():
    # difflib's autojunk would drop an element filling more than 1% of a
    # 200-element sequence, which on filing text is the boilerplate line.
    boilerplate = "See Note 12 for further discussion."
    prior = [boilerplate] * 300
    current = [boilerplate] * 300
    diffs = diff_sentences(current, prior)
    assert _labels(diffs) == [UNCHANGED] * 300


def test_empty_prior_section_labels_everything_inserted():
    diffs = diff_sentences(PRIOR_SENTENCES, [])
    assert _labels(diffs) == [INSERTED] * len(PRIOR_SENTENCES)
    assert all(d.best_prior_ordinal is None for d in diffs)


def test_empty_current_section_yields_no_diffs():
    assert diff_sentences([], PRIOR_SENTENCES) == []


def test_two_empty_sections_are_identical():
    assert normalized_levenshtein_distance("", "") == 0.0
    assert tfidf_cosine_similarity("", "") == 1.0


def test_empty_against_non_empty_is_maximum_change():
    assert normalized_levenshtein_distance("", "some prose here") == 1.0
    assert tfidf_cosine_similarity("", "some prose here") == 0.0


def test_threshold_is_the_only_thing_separating_modified_from_inserted():
    current = list(PRIOR_SENTENCES)
    current[1] = current[1].replace("largest", "second-largest")
    strict = diff_sentences(current, PRIOR_SENTENCES, modified_threshold=0.99)
    assert _labels(strict) == [UNCHANGED, INSERTED, UNCHANGED, UNCHANGED]
    # the rejected candidate is still reported, so a sweep needs no recompute
    assert strict[1].best_prior_ordinal == 1


TARGET_CIK = 111
PEER_CIK = 222
NEWCOMER_CIK = 333

T = datetime(2023, 6, 1, tzinfo=timezone.utc)
T_MINUS_2 = T - timedelta(days=730)
T_MINUS_1 = T - timedelta(days=365)
T_PLUS_1 = T + timedelta(days=365)


def _add_filing(con, accession, cik, form, filed_at, items):
    db.insert_filing(
        con,
        Filing(
            accession=accession,
            cik=cik,
            ticker=f"T{cik}",
            sector="semiconductors",
            form=form,
            filed_at=filed_at,
            period_end=None,
            url=f"https://example.com/{accession}",
        ),
    )
    for item in items:
        db.insert_section(
            con,
            Section(
                section_id=f"{accession}#{item}",
                accession=accession,
                item=item,
                text=f"body of {accession} item {item}",
                char_start=0,
                char_end=1,
            ),
        )


@pytest.fixture
def con():
    connection = db.connect(":memory:")
    db.create_schema(connection)

    # target firm: 1A in all three years, 7 missing from the middle year
    _add_filing(connection, "tgt-m2", TARGET_CIK, "10-K", T_MINUS_2, ["1A", "7"])
    _add_filing(connection, "tgt-m1", TARGET_CIK, "10-K", T_MINUS_1, ["1A"])
    _add_filing(connection, "tgt-t0", TARGET_CIK, "10-K", T, ["1A", "7"])

    # same firm, different form, item label deliberately collided with the
    # 10-K's and filed later than the 10-K it would outrank on date alone
    _add_filing(
        connection, "tgt-10q", TARGET_CIK, "10-Q", T_MINUS_1 + timedelta(days=10), ["1A"]
    )

    # a peer with the same item filed before the newcomer's first filing
    _add_filing(connection, "peer-m1", PEER_CIK, "10-K", T_MINUS_1, ["1A"])
    _add_filing(connection, "new-t0", NEWCOMER_CIK, "10-K", T, ["1A"])
    return connection


def test_align_returns_the_immediately_preceding_same_item_section(con):
    prior = align_prior_section(con, "tgt-t0#1A", as_of=T)
    assert prior is not None
    assert prior.section_id == "tgt-m1#1A"
    assert prior.item == "1A"
    assert prior.filed_at == T_MINUS_1
    assert prior.text == "body of tgt-m1 item 1A"


def test_align_returns_none_for_a_first_filing(con):
    assert align_prior_section(con, "tgt-m2#1A", as_of=T_MINUS_2) is None


def test_align_never_falls_back_to_another_firm(con):
    # the peer filed the same item at t-1; the newcomer's first filing still
    # has no prior, and a section-level diff against the peer would be
    # confident nonsense
    assert align_prior_section(con, "new-t0#1A", as_of=T) is None


def test_align_never_falls_back_to_another_item(con):
    # tgt-m1 carries no item 7, so item 7 aligns two years back, not to 1A
    prior = align_prior_section(con, "tgt-t0#7", as_of=T)
    assert prior is not None
    assert prior.section_id == "tgt-m2#7"


def test_align_matches_the_same_form(con):
    # the 10-Q's colliding 1A is 10 days more recent than tgt-m1's and would
    # win on date alone
    prior = align_prior_section(con, "tgt-t0#1A", as_of=T)
    assert prior.section_id == "tgt-m1#1A"


def test_align_never_returns_a_section_dated_at_or_after_as_of(con):
    for section_id in ("tgt-t0#1A", "tgt-t0#7", "tgt-m1#1A", "new-t0#1A"):
        for as_of in (T_MINUS_2, T_MINUS_1, T, T_PLUS_1):
            prior = align_prior_section(con, section_id, as_of=as_of)
            if prior is not None:
                assert prior.filed_at < as_of


def test_as_of_exactly_on_a_filing_date_excludes_that_filing(con):
    prior = align_prior_section(con, "tgt-t0#1A", as_of=T_MINUS_1)
    assert prior is not None
    assert prior.section_id == "tgt-m2#1A"


def test_as_of_after_the_section_itself_cannot_align_it_to_itself(con):
    prior = align_prior_section(con, "tgt-m1#1A", as_of=T_PLUS_1)
    assert prior is not None
    assert prior.section_id == "tgt-m2#1A"


def test_as_of_after_the_section_itself_cannot_align_a_later_filing(con):
    prior = align_prior_section(con, "tgt-m1#1A", as_of=T_PLUS_1)
    assert prior.filed_at < T_MINUS_1


def test_align_rejects_a_naive_as_of(con):
    with pytest.raises(ValueError):
        align_prior_section(con, "tgt-t0#1A", as_of=datetime(2023, 6, 1))


def test_align_raises_on_an_unknown_section_id(con):
    with pytest.raises(ValueError, match="broken join"):
        align_prior_section(con, "does-not-exist#1A", as_of=T)


def test_align_has_no_default_as_of(con):
    with pytest.raises(TypeError):
        align_prior_section(con, "tgt-t0#1A")
