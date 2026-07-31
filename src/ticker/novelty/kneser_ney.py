"""Interpolated Kneser-Ney, order 5, per firm, backing off to a sector model.

Written out rather than imported. `nltk` is the usual source of an
`KneserNeyInterpolated` and it is not in PLAN section 5's stack, but the
stronger reason is that this file is the measurement instrument: novelty is a
difference of two numbers this module produces, and a difference hides most of
the ways a language model can be subtly wrong. Kneser-Ney has exactly one
property that matters here and exactly one common way of being implemented
without it, so the property is spelled out below and tested directly.

Continuation counts, and why an implementation without them is not this model
------------------------------------------------------------------------------
The lower orders of a Kneser-Ney model are not smoothed unigram frequencies.
They are counts of *distinct contexts*: `N_1+(. w)`, the number of different
words that have ever preceded `w`. The standard example is "Francisco", which
is frequent in a corpus and yet a terrible guess in a context you have not
seen before, because it only ever follows "San". A model that backs off to raw
frequency ranks "Francisco" above genuinely context-free words and is stupid
backoff with a Kneser-Ney label on it. On a novelty score the damage is not
visible: it produces a number, the number varies across sentences, and nothing
downstream complains. `tests/test_kneser_ney.py` pins the distinction with a
pair of words at equal raw frequency and different continuation counts.

One discount per order, not three
---------------------------------
Chen and Goodman's modified Kneser-Ney estimates three discounts per order
(for counts of one, two, and three-or-more) rather than the single absolute
discount used here. That refinement is not implemented. The measure this
feeds is already a contrast between two models fit the same way, so a
systematic estimation improvement applies to both terms and largely cancels;
PLAN's model ladder says to ship the cheap transparent thing first and to let
Phase 5's validation decide whether anything more is warranted. The single
discount is estimated per order as `n1 / (n1 + 2 * n2)` over adjusted-count
types, which is Ney's estimate and the standard interpolated-KN default.

The sector model sits under the firm model, not beside it
----------------------------------------------------------
A firm model given a `parent` uses the parent's full-context probability as
the base distribution at the bottom of its own backoff chain, in place of the
uniform. Mass that escapes the firm's counts therefore lands on the sector's
best estimate for the same context rather than on a flat floor, and an n-gram
the firm has never written but its peers write constantly is scored as
familiar-for-the-sector instead of as maximally surprising. Where the firm
does have counts they dominate; the parent only ever receives what the firm's
discounts release.

Parent and child must share one vocabulary, and `fit` refuses to build them
otherwise. Two distributions defined over different event spaces cannot be
subtracted: the difference would carry an arbitrary constant from the
mismatch. `shared_vocabulary` builds the union for callers.

Out-of-vocabulary words
-----------------------
`<unk>` is in the vocabulary of every model. An unseen word receives the
escape mass the discounts actually reserved, `D * N_1+(context .) / total`,
carried down the chain to the base distribution, not a hardcoded floor. When
two models share a vocabulary and one is the other's parent, an entirely
unseen word contributes `-log2` of the firm's accumulated escape factor to the
contrast: a bounded quantity that reflects how much of the firm's probability
mass had to escape, which is the honest answer for a word neither model knows.

Digits are kept as tokens rather than normalized to a placeholder. A figure no
one has filed before is unseen by the firm model and by the sector model
alike, so the contrast cancels most of it; collapsing digits would instead
hide a genuinely new disclosed number. Phase 5.1's diff agreement is where
that choice gets checked.

Log base is 2 throughout. `surprisal` returns bits per token.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Iterable, Sequence

from ticker.records import Sentence

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"

DEFAULT_ORDER = 5

# A discount of zero reserves no escape mass, which makes every unseen
# continuation probability zero and every surprisal infinite. The estimator
# returns zero whenever a corpus happens to contain no singleton n-grams at
# some order, which small firms do, so the estimate is floored rather than
# trusted.
MIN_DISCOUNT = 0.1
FALLBACK_DISCOUNT = 0.75

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric runs, punctuation dropped.

    Deliberately not the retrieval tokenizer: this one keeps one- and
    two-character tokens, which BM25's `\\w\\w+` pattern discards, because
    "we", "no" and "10" carry real conditional information for a language
    model even though they carry none for a lexical ranker.
    """
    return _TOKEN_RE.findall(text.lower())


def require_aware(value: datetime, label: str) -> None:
    """Same convention as `ticker.db`, restated rather than imported from
    that module's private helper: a naive datetime here would compare against
    aware `filed_at` values and raise a TypeError from inside the fit loop
    instead of at the boundary. Public because `score.build_models` checks
    `as_of` before it does any database work."""
    if value.tzinfo is None:
        raise ValueError(
            f"{label} must be a timezone-aware datetime, got naive {value!r}."
        )


def shared_vocabulary(*sentence_groups: Sequence[Sentence]) -> frozenset[str]:
    """The vocabulary to fit a parent and its child over.

    A firm model and its sector model must be distributions over one event
    space or their surprisals cannot be subtracted. Pass the result to both
    `fit` calls; the parent fit then covers every token the child will see.
    """
    vocab: set[str] = {EOS, UNK}
    for group in sentence_groups:
        for sentence in group:
            vocab.update(tokenize(sentence.text))
    return frozenset(vocab)


class KneserNeyModel:
    """Fit by `fit`; construct it no other way.

    Immutable after construction. The `as_of` it was fit under travels with
    it so a scoring path can check that a sentence is not being scored by a
    model that was fit on the sentence itself.
    """

    __slots__ = (
        "_adjusted",
        "_as_of",
        "_context_total",
        "_context_types",
        "_discounts",
        "_order",
        "_parent",
        "_raw_unigrams",
        "_token_count",
        "_vocabulary",
    )

    def __init__(
        self,
        *,
        order: int,
        as_of: datetime,
        vocabulary: frozenset[str],
        adjusted: dict[int, dict[tuple[str, ...], int]],
        context_total: dict[int, dict[tuple[str, ...], int]],
        context_types: dict[int, dict[tuple[str, ...], int]],
        discounts: dict[int, float],
        raw_unigrams: Counter[str],
        token_count: int,
        parent: "KneserNeyModel | None",
    ) -> None:
        self._order = order
        self._as_of = as_of
        self._vocabulary = vocabulary
        self._adjusted = adjusted
        self._context_total = context_total
        self._context_types = context_types
        self._discounts = discounts
        self._raw_unigrams = raw_unigrams
        self._token_count = token_count
        self._parent = parent

    @property
    def order(self) -> int:
        return self._order

    @property
    def as_of(self) -> datetime:
        return self._as_of

    @property
    def vocabulary(self) -> frozenset[str]:
        return self._vocabulary

    @property
    def parent(self) -> "KneserNeyModel | None":
        return self._parent

    @property
    def token_count(self) -> int:
        """Training tokens, end-of-sentence markers excluded. Zero means the
        model contributes nothing of its own and every probability it returns
        comes from its parent or from the uniform base."""
        return self._token_count

    def discount(self, order: int) -> float:
        return self._discounts[order]

    def raw_unigram_count(self, word: str) -> int:
        """Plain occurrence count. Exposed for diagnostics and for the test
        that contrasts it with `continuation_count`; the model itself never
        uses raw frequency below the highest order."""
        return self._raw_unigrams.get(word, 0)

    def continuation_count(self, ngram: Sequence[str]) -> int:
        """`N_1+(. g)`: how many distinct words have ever preceded `g`.

        Undefined at the highest order, where the model holds raw counts and
        no (order+1)-gram table exists to derive left extensions from.
        """
        length = len(ngram)
        if not 1 <= length < self._order:
            raise ValueError(
                f"continuation counts exist for orders 1 to {self._order - 1}, "
                f"got a {length}-gram in an order-{self._order} model"
            )
        return self._adjusted[length].get(tuple(ngram), 0)

    def prob(self, word: str, context: Sequence[str] = ()) -> float:
        """P(word | context), strictly positive for every word in the
        vocabulary. A context shorter than order-1 is read as
        sentence-initial and left-padded with the start token."""
        target = word if word in self._vocabulary else UNK
        history = tuple(context)
        if self._order > 1:
            history = history[-(self._order - 1) :]
            history = tuple(
                token if token in self._vocabulary or token == BOS else UNK
                for token in history
            )
            if len(history) < self._order - 1:
                history = (BOS,) * (self._order - 1 - len(history)) + history
        else:
            history = ()
        return self._prob(self._order, history, target)

    def surprisal(self, text: str) -> float:
        """Mean bits per token: `-(1/|s|) * sum_i log2 P(w_i | w_<i)`.

        The end-of-sentence marker is predicted like any other token and
        counted in `|s|`. Excluding it would score a truncated fragment
        exactly like the complete sentence it was cut from.
        """
        tokens = [
            token if token in self._vocabulary else UNK for token in tokenize(text)
        ]
        predicted = tokens + [EOS]
        history = [BOS] * (self._order - 1) + tokens
        total = 0.0
        for i, word in enumerate(predicted):
            total += math.log2(self.prob(word, history[i : i + self._order - 1]))
        return -total / len(predicted)

    def _prob(self, order: int, history: tuple[str, ...], word: str) -> float:
        context = history[len(history) - (order - 1) :] if order > 1 else ()
        total = self._context_total[order].get(context, 0)
        if total == 0:
            # Nothing observed after this context at this order, so there is
            # no discounted term and the whole mass passes down unchanged.
            if order == 1:
                return self._base_prob(word, history)
            return self._prob(order - 1, history, word)

        discount = self._discounts[order]
        count = self._adjusted[order].get(context + (word,), 0)
        types = self._context_types[order].get(context, 0)
        lower = (
            self._base_prob(word, history)
            if order == 1
            else self._prob(order - 1, history, word)
        )
        return max(count - discount, 0.0) / total + (discount * types / total) * lower

    def _base_prob(self, word: str, history: tuple[str, ...]) -> float:
        if self._parent is not None:
            return self._parent.prob(word, history)
        return 1.0 / len(self._vocabulary)


def fit(
    sentences: Iterable[Sentence],
    *,
    as_of: datetime,
    order: int = DEFAULT_ORDER,
    discount: float | None = None,
    vocabulary: Iterable[str] | None = None,
    parent: KneserNeyModel | None = None,
) -> KneserNeyModel:
    """Fit an interpolated Kneser-Ney model on sentences filed before `as_of`.

    `as_of` is required and has no default, per invariant 1. Every sentence
    must be filed strictly before it; a sentence filed at exactly `as_of` is
    the document being scored and is rejected by name.

    `discount` overrides the per-order estimate with one fixed value, which
    exists for tests that need the discount held still.
    """
    require_aware(as_of, "as_of")
    if order < 1:
        raise ValueError(f"order must be at least 1, got {order}")
    if discount is not None and not 0.0 < discount <= 1.0:
        raise ValueError(f"discount must lie in (0, 1], got {discount}")

    if parent is not None:
        if parent.order != order:
            raise ValueError(
                f"parent order {parent.order} does not match child order {order}; "
                "the backoff chain is per order and cannot straddle two"
            )
        if parent.as_of > as_of:
            raise ValueError(
                f"parent was fit at as_of={parent.as_of.isoformat()}, later than the "
                f"child's as_of={as_of.isoformat()}. The child would reach data from "
                "after its own cutoff through the backoff chain."
            )
        if vocabulary is not None and _finalize_vocabulary(vocabulary) != parent.vocabulary:
            raise ValueError(
                "explicit vocabulary differs from the parent's; a child and its "
                "parent must span one event space. Build both from "
                "shared_vocabulary(...) or omit the argument."
            )
        vocabulary = parent.vocabulary

    token_lists: list[list[str]] = []
    for sentence in sentences:
        require_aware(sentence.filed_at, f"filed_at on sentence {sentence.sentence_id}")
        if sentence.filed_at >= as_of:
            raise ValueError(
                f"sentence {sentence.sentence_id} is filed at "
                f"{sentence.filed_at.isoformat()}, which is not strictly before "
                f"as_of={as_of.isoformat()}. Fitting on it would leak the document "
                "being scored into the model that scores it."
            )
        token_lists.append(tokenize(sentence.text))

    if vocabulary is None:
        vocab = _finalize_vocabulary(
            token for tokens in token_lists for token in tokens
        )
    else:
        vocab = _finalize_vocabulary(vocabulary)
        escaped = {token for tokens in token_lists for token in tokens} - vocab
        if escaped:
            raise ValueError(
                f"{len(escaped)} training token(s) fall outside the given vocabulary "
                f"(for example {sorted(escaped)[:5]}). They would be folded into "
                "<unk>, which silently deletes exactly the firm-specific wording the "
                "novelty measure is looking for. Build the vocabulary with "
                "shared_vocabulary(...) over every group being fit."
            )

    counts: Counter[tuple[str, ...]] = Counter()
    raw_unigrams: Counter[str] = Counter()
    padding = (BOS,) * (order - 1)
    token_count = 0
    for tokens in token_lists:
        mapped = tuple(token if token in vocab else UNK for token in tokens) + (EOS,)
        token_count += len(tokens)
        raw_unigrams.update(mapped)
        sequence = padding + mapped
        # Only n-grams whose last token is a real token or the end marker are
        # counted, which is what padding each order separately would give and
        # keeps the start token out of every predicted position.
        for i in range(order - 1, len(sequence)):
            counts[sequence[i - order + 1 : i + 1]] += 1

    adjusted: dict[int, dict[tuple[str, ...], int]] = {order: dict(counts)}
    for lower_order in range(order - 1, 0, -1):
        continuation: dict[tuple[str, ...], int] = defaultdict(int)
        # Each key of the order above is one distinct left extension of its
        # own suffix, so counting keys counts contexts. Padding to order-1
        # start tokens guarantees every countable n-gram below the highest
        # order has a left neighbour, which is why the key sets of the
        # adjusted tables match the key sets of the raw ones and the chain
        # can be derived downward without keeping every raw table.
        for gram in adjusted[lower_order + 1]:
            continuation[gram[1:]] += 1
        adjusted[lower_order] = dict(continuation)

    context_total: dict[int, dict[tuple[str, ...], int]] = {}
    context_types: dict[int, dict[tuple[str, ...], int]] = {}
    discounts: dict[int, float] = {}
    for level in range(1, order + 1):
        totals: dict[tuple[str, ...], int] = defaultdict(int)
        types: dict[tuple[str, ...], int] = defaultdict(int)
        singletons = 0
        doubletons = 0
        for gram, count in adjusted[level].items():
            context = gram[:-1]
            totals[context] += count
            types[context] += 1
            if count == 1:
                singletons += 1
            elif count == 2:
                doubletons += 1
        context_total[level] = dict(totals)
        context_types[level] = dict(types)
        discounts[level] = (
            discount
            if discount is not None
            else _estimate_discount(singletons, doubletons)
        )

    return KneserNeyModel(
        order=order,
        as_of=as_of,
        vocabulary=vocab,
        adjusted=adjusted,
        context_total=context_total,
        context_types=context_types,
        discounts=discounts,
        raw_unigrams=raw_unigrams,
        token_count=token_count,
        parent=parent,
    )


def _finalize_vocabulary(tokens: Iterable[str]) -> frozenset[str]:
    """Every model predicts the end marker and the unknown token and never
    predicts the start token, whatever the caller passed."""
    return (frozenset(tokens) | {EOS, UNK}) - {BOS}


def _estimate_discount(singletons: int, doubletons: int) -> float:
    denominator = singletons + 2 * doubletons
    if denominator == 0:
        return FALLBACK_DISCOUNT
    return min(max(singletons / denominator, MIN_DISCOUNT), 1.0)
