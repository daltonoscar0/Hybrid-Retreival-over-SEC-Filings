"""The Lazy Prices baseline: how much of a section's language changed.

Cohen, Malloy and Nguyen (2020), *Lazy Prices*, Journal of Finance 75(4),
1371-1415, measure period-over-period change in a filing with normalized
Levenshtein distance and TF-IDF cosine similarity against the same firm's
previous filing. This module computes both at the item-section level and
adds the sentence-level diff that labels each current sentence inserted,
modified, or unchanged.

Why this file imports nothing from the rest of ticker.novelty
--------------------------------------------------------------
PLAN section 5.1 validates the surprisal novelty score against these diff
labels: AUC of novelty on the binary changed/unchanged label. A shared
tokenizer, a shared prior-period lookup, or a shared notion of what counts
as the same sentence would make that AUC partly a measurement of the shared
code. A bug in the shared part would move the measure and its own validator
in the same direction and never show up in the number. So this module
reimplements everything it needs, including tokenization and the
prior-period alignment, and imports neither ticker.novelty.kneser_ney nor
ticker.novelty.score, directly or through any other ticker module. The only
ticker import here is the data layer's aware-datetime check, which is a
schema convention rather than part of either measure.

Tokens, not characters
----------------------
All three measures run over one token stream: casefolded runs of letters,
plus numbers with their internal separators kept, so "1,500" and "2.7"
survive as single tokens. Punctuation and whitespace are dropped.

Lazy Prices works at the word level and so does this. The stronger reason is
local. Section text here comes out of an HTML-to-text conversion, and that
conversion churns whitespace and punctuation between filings for reasons
that have nothing to do with what the firm wrote. A character-level distance
over 50,000-character sections would spend most of its budget on that churn
and report a change that no reader would recognize as one. The cost is that
an edit consisting only of punctuation reads as unchanged, which is the
right reading for a measure of new content.

The three measures sharing a token stream is deliberate: when the distance
and the cosine disagree on a pair, the disagreement is about what they
measure, not about how they split the text.

TF-IDF is fit on the pair, not on a corpus
-------------------------------------------
The vectorizer sees the two documents being compared and nothing else. That
makes the IDF nearly degenerate: with sklearn's smoothing every term has
document frequency 1 or 2, so a term unique to one of the two documents
carries weight ln(3/2) + 1 = 1.41 and a shared term carries 1.0. Read the
number as a cosine over term counts with a mild extra penalty for
disagreement, not as a corpus TF-IDF cosine.

Fitting IDF on the corpus instead would buy a real rarity weighting and cost
two things. First, document frequencies taken over the whole corpus are
taken over filings from after t, which is the leak invariant 1 exists to
prevent; an honest corpus fit needs its own `as_of` and one vectorizer per
filing date. Second, and worse for this module's job: the thing 5.1
validates is a leak-sensitive measure, and its validator should be leak-free
by construction rather than by care. A pairwise fit depends on nothing but
the two strings, so there is no `as_of` to get wrong, no corpus version to
record, and the number reproduces from the pair alone years later.

The modified threshold is 0.40
-------------------------------
`difflib.SequenceMatcher` over the two sentence lists returns equal, insert,
delete and replace blocks. Equal and insert decide themselves. Only
sentences inside a replace block need a judgment, and the judgment is
whether the block rewrites the prior sentences it displaced (modified) or is
new text that merely sits where old text was (inserted). Each current
sentence in a replace block takes its highest token similarity to any prior
sentence in the same block and is modified when that similarity is at least
0.40.

0.40 is the middle of a measured trough. Over the first 40 consecutive
same-firm 10-K pairs in the cached corpus, items 1A and 7, 68 section pairs
and 25,136 current sentences:

    block     share of current sentences
    equal                          55.4%
    replace                        41.1%
    insert                          3.5%

    best similarity in a replace block   share of those sentences
    [0.00, 0.20)                                            32.3%
    [0.20, 0.30)                                             6.6%
    [0.30, 0.50)                                            11.6%
    [0.50, 0.80)                                            21.4%
    [0.80, 1.00]                                            28.2%

At 0.05 resolution the flattest stretch is [0.30, 0.50), each bin holding
2.7% to 3.2% of replace-block sentences against 15.9% in [0.90, 1.00]. The
distribution is bimodal and 0.40 sits at the bottom between the modes. From
the other direction: random pairs of sentences drawn from the same two
sections reach 0.167 at the 99th percentile and clear 0.40 in 0.19% of
draws, so unrelated text does not reach the threshold.

Two things make the exact value low-stakes. Sliding it across the whole
trough relabels 4.8% of current sentences. And 5.1's label is binary, with
inserted and modified both counting as changed, so no threshold anywhere in
[0, 1) moves that number at all. `modified_threshold` is a parameter so the
sensitivity sweep is one loop if the AUC comes back near 0.5.

Alignment is the immediately preceding period, not the same quarter last year
------------------------------------------------------------------------------
Lazy Prices compares a 10-Q against the same fiscal quarter of the prior
year, which controls for seasonal language in quarterly reports. This module
takes the most recent prior filing of the same form carrying the same item,
because the labels feed a validation of a measure whose firm language model
is fit on the firm's entire history before t. Against a filing four quarters
back, text that first appeared three quarters ago would be labeled changed
while the language model, having already seen it, scores it as familiar. The
disagreement would be an artifact of two different definitions of "prior",
not a finding about the measure. The seasonal-language concern is real and
shows up as a lower unchanged rate on 10-Q pairs than on 10-K pairs; report
the two form types separately rather than pooling them.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Sequence

import duckdb
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ticker.db import _require_aware

MODIFIED_THRESHOLD = 0.40

UNCHANGED = "unchanged"
MODIFIED = "modified"
INSERTED = "inserted"

SentenceLabel = Literal["unchanged", "modified", "inserted"]

# Letters collapse to bare runs; digits keep internal separators so "1,500"
# and "2.7" stay whole, with the trailing-digit anchor stopping the token
# from swallowing the period that ends a sentence.
_TOKEN_RE = re.compile(r"[a-z]+|[0-9](?:[0-9.,]*[0-9])?")


def tokenize(text: str) -> list[str]:
    """The one token stream all three measures run over."""
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True, slots=True)
class SentenceDiff:
    """One current sentence's verdict.

    `best_prior_ordinal` and `best_similarity` describe the closest prior
    sentence considered, whatever the verdict turned out to be, so a caller
    can re-threshold a stored diff without recomputing it. They are None and
    0.0 only when no prior sentence was a candidate at all. The alignment
    the label actually accepted is `best_prior_ordinal` when `label` is
    MODIFIED or UNCHANGED, and nothing when it is INSERTED.
    """

    ordinal: int
    label: SentenceLabel
    best_prior_ordinal: int | None
    best_similarity: float


@dataclass(frozen=True, slots=True)
class SectionComparison:
    levenshtein_distance: float
    tfidf_cosine_similarity: float
    sentence_diffs: tuple[SentenceDiff, ...]


@dataclass(frozen=True, slots=True)
class PriorSection:
    """A section row plus the filing date that qualified it as prior.

    `filed_at` is on the record rather than left to a join because every
    consumer of this function has to report the gap it aligned across, and a
    prior section carried around without its date is the input to exactly
    the mistake this module is supposed to prevent.
    """

    section_id: str
    accession: str
    item: str
    text: str
    filed_at: datetime


def normalized_levenshtein_distance(current: str, prior: str) -> float:
    """Token-level Levenshtein distance divided by max(len_current, len_prior).

    Normalizing by the longer of the two token counts, which is rapidfuzz's
    convention for unit edit costs, puts the result in [0, 1] with 0 for
    identical token streams and 1 for streams sharing nothing. The
    alternative of dividing by the sum of the two lengths caps a total
    rewrite of two equal-length sections at 0.5 and makes the top of the
    range unreadable; dividing by the prior length alone lets a section that
    doubled in size score above 1.

    Two empty token streams are identical, distance 0. One empty stream
    against a non-empty one is distance 1.
    """
    return float(Levenshtein.normalized_distance(tokenize(current), tokenize(prior)))


def tfidf_cosine_similarity(current: str, prior: str) -> float:
    """Cosine between the two documents' TF-IDF vectors, vectorizer fit on
    the pair alone. See the module docstring for why the fit is pairwise and
    what that does to the IDF.

    Two empty token streams are identical, similarity 1. One empty stream
    against a non-empty one shares no term, similarity 0.
    """
    current_tokens = tokenize(current)
    prior_tokens = tokenize(prior)
    if not current_tokens or not prior_tokens:
        return 1.0 if not current_tokens and not prior_tokens else 0.0

    # analyzer= bypasses sklearn's own lowercasing and token pattern, which
    # drops single-character tokens and would put the cosine on a different
    # token stream from the distance.
    vectorizer = TfidfVectorizer(analyzer=tokenize)
    matrix = vectorizer.fit_transform([current, prior])
    return float(cosine_similarity(matrix[0], matrix[1])[0, 0])


def diff_sentences(
    current: Sequence[str],
    prior: Sequence[str],
    *,
    modified_threshold: float = MODIFIED_THRESHOLD,
) -> list[SentenceDiff]:
    """Label every sentence in `current` against the sentence list `prior`.

    One `SentenceDiff` per current sentence, in order, with `ordinal` the
    index into `current`. Sentences present only in `prior` are deletions
    and have nothing to label; a caller that wants them can count them from
    the prior ordinals no diff refers to.

    Diffing the sentence *list* rather than the raw text is what keeps an
    insertion from cascading: `SequenceMatcher` reports the inserted run as
    its own block and leaves the surrounding equal blocks equal, so the
    sentences after an insertion keep their unchanged labels instead of
    shifting one position and reading as edits.

    A prior sentence may be the best match for several current sentences.
    That is not a bug to break: a sentence split into two is genuinely a
    modification of one prior sentence, and forcing a one-to-one assignment
    would label the second half inserted.
    """
    current_tokens = [tuple(tokenize(text)) for text in current]
    prior_tokens = [tuple(tokenize(text)) for text in prior]

    diffs: list[SentenceDiff] = []
    # autojunk drops elements filling more than 1% of a sequence of 200 or
    # more, which on filing text means the repeated boilerplate sentence
    # that is precisely what has to match. Off, always.
    matcher = difflib.SequenceMatcher(None, prior_tokens, current_tokens, autojunk=False)
    for tag, prior_lo, prior_hi, cur_lo, cur_hi in matcher.get_opcodes():
        if tag == "delete":
            continue
        if tag == "equal":
            diffs.extend(
                SentenceDiff(
                    ordinal=cur,
                    label=UNCHANGED,
                    best_prior_ordinal=prior_lo + (cur - cur_lo),
                    best_similarity=1.0,
                )
                for cur in range(cur_lo, cur_hi)
            )
            continue
        if tag == "insert":
            diffs.extend(
                SentenceDiff(
                    ordinal=cur,
                    label=INSERTED,
                    best_prior_ordinal=None,
                    best_similarity=0.0,
                )
                for cur in range(cur_lo, cur_hi)
            )
            continue

        diffs.extend(
            _label_replace_block(
                current_tokens[cur_lo:cur_hi],
                prior_tokens[prior_lo:prior_hi],
                cur_lo,
                prior_lo,
                modified_threshold,
            )
        )

    return diffs


def _label_replace_block(
    current_tokens: list[tuple[str, ...]],
    prior_tokens: list[tuple[str, ...]],
    cur_offset: int,
    prior_offset: int,
    modified_threshold: float,
) -> list[SentenceDiff]:
    """Modified or inserted for each current sentence in one replace block.

    The similarity matrix over a replace block is quadratic in the block
    size, and a wholesale rewrite makes the whole section one block, so this
    goes through rapidfuzz's threaded C path rather than a Python loop: a
    800 x 800 block is 0.09s that way against 1.2s single-threaded.
    """
    scores = process.cdist(
        current_tokens,
        prior_tokens,
        scorer=Levenshtein.normalized_similarity,
        workers=-1,
    )
    diffs: list[SentenceDiff] = []
    for row, similarities in enumerate(scores):
        best = int(similarities.argmax())
        similarity = float(similarities[best])
        diffs.append(
            SentenceDiff(
                ordinal=cur_offset + row,
                label=MODIFIED if similarity >= modified_threshold else INSERTED,
                best_prior_ordinal=prior_offset + best,
                best_similarity=similarity,
            )
        )
    return diffs


def compare_sections(
    current: Sequence[str],
    prior: Sequence[str],
    *,
    modified_threshold: float = MODIFIED_THRESHOLD,
) -> SectionComparison:
    """All three Lazy Prices numbers over one (current, prior) sentence pair.

    Both arguments are sentence lists rather than section text, so the
    document-level distance and cosine are computed over exactly the
    sentences the diff labeled. Feeding the section text to one measure and
    the stored sentence rows to another would let a splitter change move the
    document-level numbers without moving the labels, and the two would
    stop being comparable. `ticker.sentence_split.split_sentences` already
    collapses runs of whitespace, so joining its output on a single space
    reproduces the section text token for token.
    """
    current_text = " ".join(current)
    prior_text = " ".join(prior)
    return SectionComparison(
        levenshtein_distance=normalized_levenshtein_distance(current_text, prior_text),
        tfidf_cosine_similarity=tfidf_cosine_similarity(current_text, prior_text),
        sentence_diffs=tuple(
            diff_sentences(current, prior, modified_threshold=modified_threshold)
        ),
    )


_CURRENT_SECTION_SQL = """
    SELECT f.cik, f.form, sec.item, f.filed_at
    FROM sections sec
    JOIN filings f ON sec.accession = f.accession
    WHERE sec.section_id = ?
"""

_PRIOR_SECTION_SQL = """
    SELECT sec.section_id, sec.accession, sec.item, sec.text, f.filed_at
    FROM sections sec
    JOIN filings f ON sec.accession = f.accession
    WHERE f.cik = ? AND f.form = ? AND sec.item = ? AND f.filed_at < ?
    ORDER BY f.filed_at DESC, sec.accession DESC
    LIMIT 1
"""


def align_prior_section(
    con: duckdb.DuckDBPyConnection, section_id: str, *, as_of: datetime
) -> PriorSection | None:
    """The same firm's most recent earlier version of this section, or None.

    Matched on (cik, form, item). Item labels already differ across forms in
    this corpus, so form is redundant today; it is in the key anyway because
    "the prior version of a 10-K's MD&A is a 10-K's MD&A" is the actual
    intent, and a later form whose extractor reuses an item label should
    fail to align rather than align wrongly.

    `as_of` is required and filters strictly `<`, per invariant 1. The
    effective cutoff is the earlier of `as_of` and the section's own filing
    date, so a caller who passes a later `as_of` cannot align a section
    against itself or against a filing that came after it. Same-day filings
    are excluded by the strict comparison, which loses the rare
    filed-and-amended-same-day pair and is the safe direction to lose it in.

    Returns None when the firm has no earlier filing carrying this item,
    which is the ordinary case for its first filing in the window. The
    caller records those sections as unaligned and reports the count. There
    is deliberately no fallback to a different item or a different firm: a
    diff against an unrelated section produces a full page of confident,
    meaningless change labels, and 5.1 would read them as ground truth.

    Raises ValueError for an unknown `section_id`, rather than returning
    None, so a broken join never reads as a first filing.
    """
    _require_aware(as_of, "as_of")

    row = con.execute(_CURRENT_SECTION_SQL, [section_id]).fetchone()
    if row is None:
        raise ValueError(
            f"align_prior_section: no section {section_id!r} joined to a filing. "
            "An unknown section_id is a broken join, not a section without a prior."
        )
    cik, form, item, filed_at = row

    cutoff = min(as_of, filed_at)
    prior = con.execute(_PRIOR_SECTION_SQL, [cik, form, item, cutoff]).fetchone()
    if prior is None:
        return None
    return PriorSection(*prior)
