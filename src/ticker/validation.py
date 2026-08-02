"""Phase 5 validation: does the novelty contrast track anything real.

Two independent checks, both of which run without a human label.

5.1 asks whether novelty separates sentences the Lazy Prices diff calls
changed from those it calls unchanged, as an AUC. The diff is computed by
`ticker.novelty.lazy_prices` from the text alone, with no access to the
surprisal code, so agreement between the two is agreement between two
independent constructions rather than a measure confirming itself.

5.2 asks where novelty concentrates by item type. PLAN calls this a sanity
check, and it is the one that can come back as a negative result: novelty
concentrating in Item 1 Business, the most boilerplate-heavy item in the
corpus, would say the measure detects template text rather than news.

Neither number means anything without the join diagnostics beside it. An
AUC of 0.5 is produced just as readily by a broken sentence-ID join as by a
measure that does not work, and the two are told apart only by the coverage
counts this module reports alongside every estimate.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime

import duckdb

from ticker.novelty.lazy_prices import INSERTED, MODIFIED, align_prior_section, diff_sentences

CHANGED_LABELS = frozenset({INSERTED, MODIFIED})

# Mirrors scripts/score_novelty.py's RAW_CONTRAST. Duplicated rather than
# imported because scripts/ is not a package and src/ must not depend on it.
RAW_CONTRAST = "raw_contrast"


@dataclass(frozen=True, slots=True)
class SectionObservations:
    """One aligned section's paired (novelty, changed) rows.

    Kept grouped by section rather than flattened because the bootstrap
    resamples sections, not sentences. See `bootstrap_ci`.
    """

    section_id: str
    item: str
    prior_accession: str
    gap_days: int
    novelty: tuple[float, ...]
    changed: tuple[bool, ...]


@dataclass
class JoinDiagnostics:
    """Why rows were lost, counted at every stage they can be lost at.

    RUN.md C1 says that when the AUC comes back near 0.5 the sentence-ID
    join is the first thing to check, ahead of the measure itself. That
    check is only possible if the losses were counted while they happened,
    so they are counted here rather than reconstructed afterwards.
    """

    sections_total: int = 0
    sections_unaligned: int = 0
    sections_empty_prior: int = 0
    sections_used: int = 0
    sentences_diffed: int = 0
    sentences_scored: int = 0
    sentences_joined: int = 0
    unjoined_examples: list[str] = field(default_factory=list)

    @property
    def join_rate(self) -> float:
        return self.sentences_joined / self.sentences_diffed if self.sentences_diffed else 0.0


_SECTION_SENTENCES_SQL = """
    SELECT sentence_id, ordinal, text
    FROM sentences
    WHERE section_id = ?
    ORDER BY ordinal
"""

_SECTION_FILED_SQL = """
    SELECT sec.item, f.filed_at
    FROM sections sec
    JOIN filings f ON sec.accession = f.accession
    WHERE sec.section_id = ?
"""


def _section_sentences(
    con: duckdb.DuckDBPyConnection, section_id: str
) -> tuple[list[str], list[str]]:
    """(sentence_ids, texts) in ordinal order, positions checked.

    `diff_sentences` numbers its output by position in the list it was
    handed, not by the database ordinal. Those two agree only while every
    section's ordinals are contiguous from zero, which is true of this
    corpus today and is not a property the schema enforces. Asserting it
    here turns a future gap into a crash instead of into a novelty score
    silently attached to the wrong sentence, which is a failure that would
    surface only as an AUC near 0.5 with no other symptom.
    """
    rows = con.execute(_SECTION_SENTENCES_SQL, [section_id]).fetchall()
    for position, (_, ordinal, _) in enumerate(rows):
        if ordinal != position:
            raise RuntimeError(
                f"section {section_id!r} has a gap in its sentence ordinals at "
                f"position {position} (ordinal {ordinal}). diff_sentences numbers by "
                "position, so the novelty join would attach scores to the wrong "
                "sentences. Reingest the section rather than relaxing this check."
            )
    return [row[0] for row in rows], [row[2] for row in rows]


def collect_diff_agreement(
    con: duckdb.DuckDBPyConnection,
    novelty_by_sentence: dict[str, float],
    section_ids: list[str],
    *,
    diagnostics: JoinDiagnostics | None = None,
) -> tuple[list[SectionObservations], JoinDiagnostics]:
    """Pair every scored sentence with the diff's verdict on it.

    `as_of` for the alignment is the section's own filing date, so the prior
    is strictly earlier per invariant 1. A section whose firm has no earlier
    filing carrying the item has no prior to diff against and is excluded
    rather than counted as all-inserted: labelling a first filing's every
    sentence "new" would be true and would also inflate the AUC for a reason
    that has nothing to do with novelty.
    """
    diag = diagnostics or JoinDiagnostics()
    observations: list[SectionObservations] = []

    for section_id in section_ids:
        diag.sections_total += 1
        row = con.execute(_SECTION_FILED_SQL, [section_id]).fetchone()
        if row is None:
            raise ValueError(f"unknown section_id {section_id!r}")
        item, filed_at = row

        prior = align_prior_section(con, section_id, as_of=filed_at)
        if prior is None:
            diag.sections_unaligned += 1
            continue

        sentence_ids, current_texts = _section_sentences(con, section_id)
        _, prior_texts = _section_sentences(con, prior.section_id)
        if not current_texts or not prior_texts:
            diag.sections_empty_prior += 1
            continue

        diffs = diff_sentences(current_texts, prior_texts)
        diag.sentences_diffed += len(diffs)

        novelty: list[float] = []
        changed: list[bool] = []
        for diff in diffs:
            sentence_id = sentence_ids[diff.ordinal]
            score = novelty_by_sentence.get(sentence_id)
            if score is None:
                if len(diag.unjoined_examples) < 10:
                    diag.unjoined_examples.append(sentence_id)
                continue
            novelty.append(score)
            changed.append(diff.label in CHANGED_LABELS)

        diag.sentences_joined += len(novelty)
        if not novelty:
            continue

        diag.sections_used += 1
        observations.append(
            SectionObservations(
                section_id=section_id,
                item=item,
                prior_accession=prior.accession,
                gap_days=(filed_at - prior.filed_at).days,
                novelty=tuple(novelty),
                changed=tuple(changed),
            )
        )

    diag.sentences_scored = len(novelty_by_sentence)
    return observations, diag


@dataclass(frozen=True, slots=True)
class Coverage:
    """What the score file holds against what the corpus holds, by form.

    `JoinDiagnostics` counts rows lost inside the sections the file contains,
    so a firm the scoring pass never reached is invisible to it: its sections
    are simply not in the file to be counted as missing. A truncated pass
    therefore produces a clean-looking join rate over a corpus subset, which
    is the failure this reconciles against the `filings` table instead.
    """

    forms: tuple[str, ...]
    accessions_in_file: int
    accessions_in_corpus: int
    ciks_in_file: int
    ciks_in_corpus: int
    missing_accessions: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_accessions


_CORPUS_ACCESSIONS_SQL = """
    SELECT DISTINCT f.accession, f.cik
    FROM filings f
    JOIN sections sec ON sec.accession = f.accession
    WHERE f.form IN ({placeholders})
"""


def coverage(con: duckdb.DuckDBPyConnection, path) -> Coverage:
    """Reconcile a score file against the filings it claims to cover."""
    import json

    forms: set[str] = set()
    accessions: set[str] = set()
    ciks: set[int] = set()
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            forms.add(record["form"])
            accessions.add(record["accession"])
            ciks.add(record["cik"])

    ordered_forms = tuple(sorted(forms))
    if not ordered_forms:
        return Coverage((), 0, 0, 0, 0, ())

    placeholders = ", ".join("?" for _ in ordered_forms)
    rows = con.execute(
        _CORPUS_ACCESSIONS_SQL.format(placeholders=placeholders), list(ordered_forms)
    ).fetchall()
    corpus_accessions = {row[0] for row in rows}
    corpus_ciks = {row[1] for row in rows}

    return Coverage(
        forms=ordered_forms,
        accessions_in_file=len(accessions),
        accessions_in_corpus=len(corpus_accessions),
        ciks_in_file=len(ciks),
        ciks_in_corpus=len(corpus_ciks),
        missing_accessions=tuple(sorted(corpus_accessions - accessions)),
    )


def roc_auc(scores: list[float], labels: list[bool]) -> float:
    """AUC by the rank form of the Mann-Whitney statistic, midranks for ties.

    Written out rather than taken from sklearn because the bootstrap calls it
    thousands of times and the rank form is the cheap one. `tests/
    test_validation.py` asserts it equals `sklearn.metrics.roc_auc_score` to
    twelve decimals over random inputs including heavy ties, which is the
    same cross-implementation check `ticker.evaluation` runs against
    pytrec_eval.

    Returns 0.5 when either class is absent: with nothing to separate, no
    ranking is better than any other, and raising would abort a bootstrap
    resample that happened to draw one class.
    """
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        midrank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = midrank
        i = j + 1

    positive_rank_sum = sum(ranks[i] for i in range(len(labels)) if labels[i])
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def bootstrap_ci(
    observations: list[SectionObservations],
    *,
    resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(point estimate, lower, upper) for the diff-agreement AUC.

    Resamples sections with replacement, never sentences. Sentences inside
    one section share a firm, a filing date, and one prior alignment, so
    they are not independent draws; resampling them individually would treat
    a section's worth of correlated rows as a section's worth of evidence
    and return an interval too narrow to be honest. The cluster bootstrap
    is the standard correction and it is the difference between a CI that
    excludes 0.5 and one that does not.
    """
    if not observations:
        return 0.5, 0.5, 0.5

    flat_scores = [s for obs in observations for s in obs.novelty]
    flat_labels = [c for obs in observations for c in obs.changed]
    point = roc_auc(flat_scores, flat_labels)

    rng = random.Random(seed)
    estimates: list[float] = []
    n = len(observations)
    for _ in range(resamples):
        drawn = [observations[rng.randrange(n)] for _ in range(n)]
        scores = [s for obs in drawn for s in obs.novelty]
        labels = [c for obs in drawn for c in obs.changed]
        estimates.append(roc_auc(scores, labels))

    estimates.sort()
    lower = estimates[int((alpha / 2) * resamples)]
    upper = estimates[min(int((1 - alpha / 2) * resamples), resamples - 1)]
    return point, lower, upper


def mean_novelty_by_item(
    rows: list[tuple[str, float]],
    *,
    resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, tuple[int, float, float, float]]:
    """item -> (n, mean, lower, upper), percentile bootstrap over sentences.

    Sentence-level resampling is right here and wrong in `bootstrap_ci`. The
    quantity is a per-item mean over the corpus rather than a ranking
    statistic, and the question it answers is where novelty sits across all
    sentences of an item, so the sentence is the unit.
    """
    by_item: dict[str, list[float]] = {}
    for item, score in rows:
        by_item.setdefault(item, []).append(score)

    # Seeded once for the whole table, not once per item. Reseeding inside the
    # loop gives every item an identical stream of resample indices, so two
    # items with the same n draw the same positions and their intervals move
    # together. Each CI stays individually valid and the table stops supporting
    # the cross-item comparison it exists to invite.
    rng = random.Random(seed)
    out: dict[str, tuple[int, float, float, float]] = {}
    for item, values in sorted(by_item.items()):
        n = len(values)
        mean = sum(values) / n
        means = sorted(
            sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
        )
        lower = means[int((alpha / 2) * resamples)]
        upper = means[min(int((1 - alpha / 2) * resamples), resamples - 1)]
        out[item] = (n, mean, lower, upper)
    return out


def load_scores(path) -> tuple[dict[str, float], list[tuple[str, float]], list[str]]:
    """(novelty by sentence_id, (item, novelty) rows, distinct section_ids).

    Reads the raw contrast column and nothing else. Invariant 3 keeps the
    z-score to within-document display, and every number in this module is
    cross-document.
    """
    import json

    by_sentence: dict[str, float] = {}
    item_rows: list[tuple[str, float]] = []
    sections: dict[str, None] = {}
    with open(path) as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            kind = record.get("score_kind")
            if kind != RAW_CONTRAST:
                raise ValueError(
                    f"{path} line {number} has score_kind {kind!r}, expected "
                    f"{RAW_CONTRAST!r}. Every number this module computes is a "
                    "cross-document aggregation, and invariant 3 confines the "
                    "z-score to within-document display. A file written before "
                    "the column existed carries no such assurance and has to be "
                    "rescored with scripts/score_novelty.py rather than assumed."
                )
            by_sentence[record["sentence_id"]] = record["novelty"]
            item_rows.append((record["item"], record["novelty"]))
            sections[record["section_id"]] = None
    return by_sentence, item_rows, list(sections)
