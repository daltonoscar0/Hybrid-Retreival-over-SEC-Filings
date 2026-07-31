"""Kneser-Ney contract, written before the implementation.

The failure mode worth this much test code: an implementation that backs off
to raw unigram frequency instead of continuation counts still produces
plausible numbers, still orders sentences, and is not Kneser-Ney. The "San
Francisco" pair below is the test that separates the two, and it is written so
that a raw-frequency backoff returns exactly equal probabilities where the
correct model does not.

The rest is there because novelty is a difference of two of these numbers, and
a difference cancels the errors that are visible in a single model's output
while keeping the ones that are not: an unnormalized context, a discount that
reserves no escape mass, a firm model that quietly leaks its own filing into
its own history.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from ticker.novelty.kneser_ney import (
    BOS,
    DEFAULT_ORDER,
    EOS,
    UNK,
    fit,
    shared_vocabulary,
    tokenize,
)
from ticker.records import Sentence

AS_OF = datetime(2024, 1, 1, tzinfo=timezone.utc)
BEFORE = AS_OF - timedelta(days=1)
WELL_BEFORE = AS_OF - timedelta(days=400)
AFTER = AS_OF + timedelta(days=1)


def _sentences(
    texts: list[str], filed_at: datetime = BEFORE, prefix: str = "s"
) -> list[Sentence]:
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


# "francisco" and "office" occur four times each. "francisco" is preceded only
# ever by "san"; "office" is preceded by four different words. Raw unigram
# frequency cannot tell them apart. Continuation counts differ 1 to 4.
SF_TEXTS = [
    "we opened an office in san francisco",
    "the new office in san francisco hired",
    "a regional office near san francisco closed",
    "every branch office beyond san francisco expanded",
]

MONO_TEXTS = ["the company reported strong results"] * 5

PARENT_TEXTS = [
    "the board approved a dividend",
    "the board approved a dividend",
    "the board approved a dividend",
    "the board approved a merger",
]
CHILD_TEXTS = ["our fabrication yields improved"]


def _training_contexts(order: int, texts: list[str]) -> list[tuple[str, ...]]:
    """Every (order-1)-token context the model was actually asked to learn,
    including the padded sentence-initial ones."""
    seen: set[tuple[str, ...]] = set()
    for text in texts:
        tokens = [BOS] * (order - 1) + tokenize(text) + [EOS]
        for i in range(order - 1, len(tokens)):
            seen.add(tuple(tokens[i - order + 1 : i]))
    return sorted(seen)


# --- normalization ---------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2, 3, 5])
def test_probabilities_sum_to_one_in_every_training_context(order):
    model = fit(_sentences(SF_TEXTS), as_of=AS_OF, order=order)
    for context in _training_contexts(order, SF_TEXTS):
        total = sum(model.prob(word, context) for word in model.vocabulary)
        assert total == pytest.approx(1.0, abs=1e-9), context


def test_probabilities_sum_to_one_in_a_context_never_seen_in_training():
    model = fit(_sentences(SF_TEXTS), as_of=AS_OF)
    context = ("cryogenic", "wafer", "yield", "variance")
    total = sum(model.prob(word, context) for word in model.vocabulary)
    assert total == pytest.approx(1.0, abs=1e-9)


def test_probabilities_sum_to_one_when_a_parent_carries_the_backoff():
    vocab = shared_vocabulary(_sentences(PARENT_TEXTS), _sentences(CHILD_TEXTS))
    parent = fit(_sentences(PARENT_TEXTS), as_of=AS_OF, vocabulary=vocab)
    child = fit(_sentences(CHILD_TEXTS, prefix="c"), as_of=AS_OF, vocabulary=vocab, parent=parent)
    for context in _training_contexts(DEFAULT_ORDER, PARENT_TEXTS + CHILD_TEXTS):
        total = sum(child.prob(word, context) for word in child.vocabulary)
        assert total == pytest.approx(1.0, abs=1e-9), context


def test_sentence_boundary_token_is_predictable_and_start_token_is_not():
    model = fit(_sentences(SF_TEXTS), as_of=AS_OF)
    assert EOS in model.vocabulary
    assert BOS not in model.vocabulary  # a start token is conditioned on, never predicted


# --- continuation counts, which is the whole of Kneser-Ney -----------------


def test_continuation_count_differs_from_raw_frequency():
    model = fit(_sentences(SF_TEXTS), as_of=AS_OF, order=3)
    assert model.raw_unigram_count("francisco") == 4
    assert model.raw_unigram_count("office") == 4
    assert model.continuation_count(("francisco",)) == 1  # only ever after "san"
    assert model.continuation_count(("office",)) == 4


def test_backoff_uses_continuation_counts_not_raw_frequency():
    # Both words have raw frequency 4, so a model that backs off to raw
    # unigram frequency scores them identically in a context it has never
    # seen. Kneser-Ney does not: "francisco" is a word that appears in one
    # context, not a word that appears often.
    model = fit(_sentences(SF_TEXTS), as_of=AS_OF, order=3)
    unseen = ("quarterly", "guidance")
    assert model.prob("francisco", unseen) < model.prob("office", unseen)


def test_continuation_count_is_undefined_at_the_highest_order():
    model = fit(_sentences(SF_TEXTS), as_of=AS_OF, order=3)
    with pytest.raises(ValueError):
        model.continuation_count(("in", "san", "francisco"))


# --- discounting -----------------------------------------------------------


@pytest.mark.parametrize("discount", [0.1, 0.5, 0.9])
def test_fixed_discount_is_used_at_every_order(discount):
    model = fit(_sentences(MONO_TEXTS), as_of=AS_OF, order=3, discount=discount)
    assert [model.discount(k) for k in (1, 2, 3)] == [discount] * 3


def test_larger_discount_moves_mass_off_seen_ngrams_onto_unseen_ones():
    context = ("the", "company")
    seen: list[float] = []
    unseen: list[float] = []
    for discount in (0.1, 0.5, 0.9):
        model = fit(_sentences(MONO_TEXTS), as_of=AS_OF, order=3, discount=discount)
        seen.append(model.prob("reported", context))
        unseen.append(model.prob("results", context))  # in vocabulary, never after this context
    assert seen[0] > seen[1] > seen[2]
    assert unseen[0] < unseen[1] < unseen[2]


def test_estimated_discount_stays_inside_the_open_unit_interval():
    model = fit(_sentences(SF_TEXTS), as_of=AS_OF)
    for order in range(1, DEFAULT_ORDER + 1):
        assert 0.0 < model.discount(order) <= 1.0


# --- backoff to the parent model -------------------------------------------


def _parent_and_children():
    # The discount is fixed rather than estimated: every n-gram in a corpus
    # this small is a singleton, the estimate is then exactly 1.0, and a
    # discount of 1.0 releases the child's entire mass so the child and the
    # parent return identical numbers. That is correct behaviour on four
    # sentences and it would make the assertions below vacuous.
    vocab = shared_vocabulary(_sentences(PARENT_TEXTS), _sentences(CHILD_TEXTS))
    kwargs = {"as_of": AS_OF, "vocabulary": vocab, "discount": 0.5}
    parent = fit(_sentences(PARENT_TEXTS), **kwargs)
    child = fit(_sentences(CHILD_TEXTS, prefix="c"), **kwargs, parent=parent)
    orphan = fit(_sentences(CHILD_TEXTS, prefix="c"), **kwargs)
    return parent, child, orphan


def test_ngram_unseen_by_the_child_falls_through_to_the_parent_distribution():
    parent, child, _ = _parent_and_children()
    context = ("the", "board", "approved", "a")
    assert parent.prob("dividend", context) > parent.prob("merger", context)
    # The child has seen neither word and neither context. If its escape mass
    # reaches the parent, the ratio it assigns is exactly the parent's ratio:
    # the escape factor depends on the context, not on the word.
    assert child.prob("dividend", context) / child.prob("merger", context) == pytest.approx(
        parent.prob("dividend", context) / parent.prob("merger", context)
    )
    assert 0.0 < child.prob("dividend", context) < parent.prob("dividend", context)


def test_without_a_parent_the_same_ngram_falls_through_to_a_flat_floor():
    _, _, orphan = _parent_and_children()
    context = ("the", "board", "approved", "a")
    # Same two words, same context, no parent: the uniform base cannot tell
    # them apart. This is the behaviour the parent model exists to replace.
    assert orphan.prob("dividend", context) == pytest.approx(orphan.prob("merger", context))


def test_child_with_no_training_data_reproduces_the_parent_exactly():
    vocab = shared_vocabulary(_sentences(PARENT_TEXTS))
    parent = fit(_sentences(PARENT_TEXTS), as_of=AS_OF, vocabulary=vocab)
    empty = fit([], as_of=AS_OF, vocabulary=vocab, parent=parent)
    assert empty.token_count == 0
    assert empty.surprisal("the board approved a merger") == pytest.approx(
        parent.surprisal("the board approved a merger")
    )


def test_parent_fit_at_a_later_as_of_is_rejected_as_a_leak():
    parent = fit(_sentences(PARENT_TEXTS), as_of=AFTER)
    with pytest.raises(ValueError, match="as_of"):
        fit(_sentences(PARENT_TEXTS, prefix="c"), as_of=AS_OF, parent=parent)


def test_parent_of_a_different_order_is_rejected():
    parent = fit(_sentences(PARENT_TEXTS), as_of=AS_OF, order=3)
    with pytest.raises(ValueError, match="order"):
        fit(_sentences(PARENT_TEXTS, prefix="c"), as_of=AS_OF, order=5, parent=parent)


def test_child_tokens_outside_the_parent_vocabulary_are_rejected():
    parent = fit(_sentences(PARENT_TEXTS), as_of=AS_OF)
    with pytest.raises(ValueError, match="shared_vocabulary"):
        fit(_sentences(CHILD_TEXTS, prefix="c"), as_of=AS_OF, parent=parent)


# --- out of vocabulary -----------------------------------------------------


def test_unseen_word_gets_probability_from_the_backoff_chain_not_zero():
    model = fit(_sentences(SF_TEXTS), as_of=AS_OF)
    p = model.prob("securitization", ("in", "san", "francisco", "we"))
    assert p > 0.0
    assert p == pytest.approx(model.prob(UNK, ("in", "san", "francisco", "we")))


def test_every_word_in_the_vocabulary_has_strictly_positive_probability():
    model = fit(_sentences(SF_TEXTS), as_of=AS_OF)
    context = ("in", "san")
    assert all(model.prob(word, context) > 0.0 for word in model.vocabulary)


# --- surprisal -------------------------------------------------------------


def test_surprisal_is_mean_negative_log2_probability_over_tokens_and_eos():
    model = fit(_sentences(SF_TEXTS), as_of=AS_OF)
    text = "we opened an office in san francisco"
    tokens = tokenize(text)
    predicted = tokens + [EOS]
    padded = [BOS] * (model.order - 1) + tokens
    expected = -sum(
        math.log2(model.prob(word, padded[i : i + model.order - 1]))
        for i, word in enumerate(predicted)
    ) / len(predicted)
    assert model.surprisal(text) == pytest.approx(expected)


def test_surprisal_is_lower_for_text_the_model_was_fit_on():
    model = fit(_sentences(SF_TEXTS), as_of=AS_OF)
    familiar = model.surprisal("we opened an office in san francisco")
    foreign = model.surprisal("goodwill impairment charges reduced segment operating income")
    assert familiar < foreign


def test_surprisal_of_an_empty_string_is_finite():
    model = fit(_sentences(SF_TEXTS), as_of=AS_OF)
    assert model.surprisal("") > 0.0


# --- time discipline, invariant 1 ------------------------------------------


def test_as_of_has_no_default():
    with pytest.raises(TypeError):
        fit(_sentences(SF_TEXTS))


def test_fit_rejects_a_sentence_filed_after_as_of():
    late = _sentences(SF_TEXTS, filed_at=AFTER, prefix="late")
    with pytest.raises(ValueError):
        fit(late, as_of=AS_OF)


def test_fit_rejects_a_sentence_filed_exactly_at_as_of():
    # The strict `<` boundary. A filing at exactly as_of is the document being
    # scored; letting it in is the model reading its own answer.
    at_boundary = _sentences(SF_TEXTS, filed_at=AS_OF, prefix="boundary")
    with pytest.raises(ValueError):
        fit(at_boundary, as_of=AS_OF)


def test_fit_accepts_a_sentence_filed_one_microsecond_before_as_of():
    just_before = _sentences(
        SF_TEXTS, filed_at=AS_OF - timedelta(microseconds=1), prefix="just"
    )
    assert fit(just_before, as_of=AS_OF).token_count > 0


def test_fit_error_names_the_offending_sentence():
    mixed = _sentences(SF_TEXTS[:2], filed_at=WELL_BEFORE) + _sentences(
        SF_TEXTS[2:], filed_at=AS_OF, prefix="offender"
    )
    with pytest.raises(ValueError, match="offender-0"):
        fit(mixed, as_of=AS_OF)


def test_fit_rejects_a_naive_as_of():
    with pytest.raises(ValueError, match="timezone-aware"):
        fit(_sentences(SF_TEXTS), as_of=datetime(2024, 1, 1))


def test_fit_rejects_a_naive_filed_at():
    naive = _sentences(SF_TEXTS, filed_at=datetime(2023, 1, 1), prefix="naive")
    with pytest.raises(ValueError, match="timezone-aware"):
        fit(naive, as_of=AS_OF)


def test_model_records_the_as_of_it_was_fit_under():
    model = fit(_sentences(SF_TEXTS), as_of=AS_OF)
    assert model.as_of == AS_OF
