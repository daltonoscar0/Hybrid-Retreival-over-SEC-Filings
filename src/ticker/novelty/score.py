"""Novelty as a contrast, plus the guardrails that keep it one.

novelty(s) = surprisal_firm(s) - surprisal_background(s)

Both terms in bits per token, both models fit strictly before the same
`as_of`, both over the same vocabulary. That is invariant 2 in full: raw
surprisal is a property of a model and a sentence, not of a sentence, and it
is not exported from this module in any form. Nothing here returns a bare
surprisal and nothing here accepts one.

Why the contrast rather than the firm model alone
--------------------------------------------------
A firm model on its own gives a high score to any sentence containing unusual
English, which in filings means jargon, statute names, and long numbers. All
of those are equally unusual to the sector model, so subtracting cancels them.
What survives is the quantity a reader wants: ordinary sector language this
firm has never written. The residual bias runs the other way and is worth
knowing about. A firm model is fit on less text than its sector model, so it
reserves more escape mass, so a token neither model has seen still scores
slightly positive. The offset is bounded and shared across every sentence
scored by the same pair of models, which is why the pair is built once per
(firm, as_of) rather than per sentence.

Why the z-scored value has its own type
----------------------------------------
Invariant 3. Within-document z-scores are for colouring a UI, where only the
ordering inside one document matters. Averaging them across documents gives
every document a mean of zero, so a corpus-level "novelty index" built that
way is a table of noise that looks exactly like a table of results. A comment
saying so is not enough: `novelty_zscored_for_display` returns
`DisplayNovelties`, `document_novelty_index` accepts only `DocumentNovelty`,
and `DisplayNovelty` carries no raw value to reach through. The mistake is a
TypeError at the call site.

Invariant 4 lives elsewhere: nothing in this module is wired into a relevance
score, and the ranking path takes novelty as a separate column.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import duckdb

from ticker import db
from ticker.novelty.kneser_ney import (
    DEFAULT_ORDER,
    KneserNeyModel,
    fit,
    require_aware,
    shared_vocabulary,
)
from ticker.records import Sentence


@dataclass(frozen=True, slots=True)
class RawNovelty:
    """One sentence's firm-minus-background contrast, in bits per token.

    Comparable across documents and across firms, which is the whole reason
    it exists alongside the display type. The two component surprisals are
    deliberately not fields: invariant 2 keeps raw surprisal out of every
    caller's reach, and a field is a reach.
    """

    sentence_id: str
    contrast: float


@dataclass(frozen=True, slots=True)
class DocumentNovelty:
    document_id: str
    as_of: datetime
    scores: tuple[RawNovelty, ...]


@dataclass(frozen=True, slots=True)
class DisplayNovelty:
    sentence_id: str
    z: float


@dataclass(frozen=True, slots=True)
class DisplayNovelties:
    """Within-document z-scores, for highlighting one document's sentences.

    Not accepted by any aggregation function in this module. See the module
    docstring.
    """

    document_id: str
    values: tuple[DisplayNovelty, ...]


def _check_pair(firm_lm: KneserNeyModel, background_lm: KneserNeyModel) -> None:
    if firm_lm.as_of != background_lm.as_of:
        raise ValueError(
            f"firm model as_of={firm_lm.as_of.isoformat()} and background model "
            f"as_of={background_lm.as_of.isoformat()} differ. A contrast between "
            "two different time horizons measures the horizon, not the sentence."
        )
    if firm_lm.vocabulary != background_lm.vocabulary:
        raise ValueError(
            "firm and background models were fit over different vocabularies. "
            "Their surprisals are then defined over different event spaces and "
            "the difference carries an arbitrary constant. Build both from "
            "shared_vocabulary(...), or use build_models()."
        )


def novelty_raw(
    sentence: Sentence, firm_lm: KneserNeyModel, background_lm: KneserNeyModel
) -> float:
    """Bits per token by which the firm model is more surprised than its sector.

    Positive means new for this firm and ordinary for its peers. Negative
    means this firm's own boilerplate. Zero means the two models agree, which
    is what a firm with no prior filings produces.
    """
    _check_pair(firm_lm, background_lm)
    if sentence.filed_at < firm_lm.as_of:
        raise ValueError(
            f"sentence {sentence.sentence_id} is filed at "
            f"{sentence.filed_at.isoformat()}, before the models' "
            f"as_of={firm_lm.as_of.isoformat()}. It may be in their training data, "
            "in which case the score is the model recognising itself."
        )
    return firm_lm.surprisal(sentence.text) - background_lm.surprisal(sentence.text)


def score_document(
    document_id: str,
    sentences: list[Sentence] | tuple[Sentence, ...],
    firm_lm: KneserNeyModel,
    background_lm: KneserNeyModel,
) -> DocumentNovelty:
    """Raw contrasts for one document, in the order given.

    `document_id` is the caller's: an accession for a filing, a section_id
    for a section. It travels with the scores so a later z-scoring cannot be
    handed two documents' sentences without the caller having said so.
    """
    _check_pair(firm_lm, background_lm)
    return DocumentNovelty(
        document_id=document_id,
        as_of=firm_lm.as_of,
        scores=tuple(
            RawNovelty(
                sentence_id=sentence.sentence_id,
                contrast=novelty_raw(sentence, firm_lm, background_lm),
            )
            for sentence in sentences
        ),
    )


def novelty_zscored_for_display(doc: DocumentNovelty) -> DisplayNovelties:
    """Z-score the raw contrasts within one document, for UI highlighting.

    Population standard deviation, not sample: the document is not a sample
    of itself. A document whose sentences all score alike gets zeros rather
    than a division by zero, which is the honest rendering of "nothing here
    stands out from the rest of this document".
    """
    if not isinstance(doc, DocumentNovelty):
        raise TypeError(
            f"expected DocumentNovelty, got {type(doc).__name__}. Z-scoring is "
            "defined over one document's raw contrasts."
        )
    if not doc.scores:
        raise ValueError(f"document {doc.document_id} has no scored sentences")

    contrasts = [score.contrast for score in doc.scores]
    mean = sum(contrasts) / len(contrasts)
    deviation = math.sqrt(sum((c - mean) ** 2 for c in contrasts) / len(contrasts))
    return DisplayNovelties(
        document_id=doc.document_id,
        values=tuple(
            DisplayNovelty(
                sentence_id=score.sentence_id,
                z=0.0 if deviation == 0.0 else (score.contrast - mean) / deviation,
            )
            for score in doc.scores
        ),
    )


def document_novelty_index(doc: DocumentNovelty) -> float:
    """Mean raw contrast over a document, comparable across documents."""
    if isinstance(doc, (DisplayNovelties, DisplayNovelty)):
        raise TypeError(
            "z-scored novelty is within-document only (invariant 3). Every "
            "document's z-scores average to zero, so aggregating them produces a "
            "number that varies only with rounding. Aggregate the DocumentNovelty "
            "these came from."
        )
    if not isinstance(doc, DocumentNovelty):
        raise TypeError(f"expected DocumentNovelty, got {type(doc).__name__}")
    if not doc.scores:
        raise ValueError(f"document {doc.document_id} has no scored sentences")
    return sum(score.contrast for score in doc.scores) / len(doc.scores)


def _sector_section_ids(
    con: duckdb.DuckDBPyConnection, sector: str
) -> frozenset[str]:
    """Sections belonging to filers in `sector`, with no time predicate.

    Sector membership is metadata, not a fitted statistic, so this query has
    nothing to leak. The strict `<` filter stays in `ticker.db` where
    invariant 1 puts it, and this set only narrows what that filter already
    returned.
    """
    rows = con.execute(
        """
        SELECT sec.section_id
        FROM sections sec
        JOIN filings f ON sec.accession = f.accession
        WHERE f.sector = ?
        """,
        [sector],
    ).fetchall()
    return frozenset(row[0] for row in rows)


def _check_sector(con: duckdb.DuckDBPyConnection, cik: int, sector: str) -> None:
    rows = con.execute(
        "SELECT DISTINCT sector FROM filings WHERE cik = ?", [cik]
    ).fetchall()
    if not rows:
        raise ValueError(
            f"cik {cik} has no filings in this database. A firm model fit on "
            "nothing scores every sentence at zero, which reads as a measurement "
            "rather than as a typo."
        )
    sectors = {row[0] for row in rows}
    if sector not in sectors:
        raise ValueError(
            f"cik {cik} files under {sorted(sectors)}, not {sector!r}. The "
            "background would be drawn from the wrong peer group."
        )


def build_models(
    con: duckdb.DuckDBPyConnection,
    *,
    cik: int,
    sector: str,
    as_of: datetime,
    order: int = DEFAULT_ORDER,
) -> tuple[KneserNeyModel, KneserNeyModel]:
    """(firm model, sector-matched background model), both fit before `as_of`.

    The background is the firm's own sector minus the firm itself, which is
    the control PLAN section 1 argues for: a corpus-wide background would
    charge the firm for language that is standard in semiconductors and
    unheard of in regional banks. `db.background_sentences` handles the time
    filter and the firm exclusion; the sector narrowing happens here rather
    than in a new data-layer query, so there is exactly one place in the
    codebase that writes a `filed_at <` predicate.

    Both models are fit over the union vocabulary so their surprisals can be
    subtracted, and the background is wired in as the firm model's parent so
    an n-gram the firm has never written falls through to its peers rather
    than to a uniform floor.

    Cost note: this loads every sentence filed before `as_of` outside the
    firm and drops the other sector's in Python. Callers scoring a whole
    corpus should cache the pair per (sector, as_of) rather than rebuilding
    it per filing.
    """
    require_aware(as_of, "as_of")
    _check_sector(con, cik, sector)
    sector_sections = _sector_section_ids(con, sector)

    firm_sentences = db.prior_sentences(con, cik, as_of)
    background_sentences = [
        sentence
        for sentence in db.background_sentences(con, cik, as_of)
        if sentence.section_id in sector_sections
    ]

    vocabulary = shared_vocabulary(firm_sentences, background_sentences)
    background_lm = fit(
        background_sentences, as_of=as_of, order=order, vocabulary=vocabulary
    )
    firm_lm = fit(
        firm_sentences,
        as_of=as_of,
        order=order,
        vocabulary=vocabulary,
        parent=background_lm,
    )
    return firm_lm, background_lm
