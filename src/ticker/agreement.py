"""Cohen's kappa between two judging passes of the same pairs.

Why a second pass at all
-------------------------
A judge's grades are an instrument, not ground truth, and the instrument
drifts over a long session: sharp at judgment 4, tired at judgment 400.
Re-judging a random sample blind to the first grade and comparing the two
passes with Cohen's kappa (Cohen, 1960) is the standard way to report how
much of a judging pass's signal survives its own noise. This is
intra-annotator agreement -- one person judging their own earlier work a
second time -- not inter-annotator agreement between two different judges.
It is a ceiling: no retrieval-quality claim built on these qrels can be more
reliable than the labels it is measured against, and this number says how
reliable that is.

A skip on either pass is kept as its own category rather than dropped. A
judge who was confident enough to grade something on pass 1 but skips it on
pass 2 (or the reverse) is disagreeing with themselves, and silently
excluding that pair from the kappa computation would make the tool's
reliability look better than it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ticker.qrels import iter_judgments, latest_by_pair

Label = int | str  # a grade 0-3, or the string "skip"


def _label(grade: int | None) -> Label:
    return "skip" if grade is None else grade


def cohens_kappa(labels_a: Sequence[Label], labels_b: Sequence[Label]) -> float:
    """Unweighted Cohen's kappa over paired categorical labels.

    Returns nan for zero pairs, or when both passes are perfectly uniform
    and identical (expected agreement is 1.0, so the ratio is 0/0 -- total
    agreement with no variance to measure, not a meaningful kappa value; the
    caller should report the raw agreement rate for that case instead).
    """
    n = len(labels_a)
    if n != len(labels_b):
        raise ValueError("labels_a and labels_b must be the same length")
    if n == 0:
        return float("nan")

    categories = sorted(set(labels_a) | set(labels_b), key=str)
    index = {category: i for i, category in enumerate(categories)}
    matrix = [[0] * len(categories) for _ in categories]
    for a, b in zip(labels_a, labels_b):
        matrix[index[a]][index[b]] += 1

    po = sum(matrix[i][i] for i in range(len(categories))) / n
    row_totals = [sum(row) for row in matrix]
    col_totals = [sum(matrix[i][j] for i in range(len(categories))) for j in range(len(categories))]
    pe = sum(row_totals[i] * col_totals[i] for i in range(len(categories))) / (n * n)

    if pe == 1.0:
        return 1.0 if po == 1.0 else float("nan")
    return (po - pe) / (1 - pe)


@dataclass(frozen=True)
class AgreementReport:
    n_pairs: int
    observed_agreement: float
    expected_agreement: float
    kappa: float
    confusion: dict[tuple[Label, Label], int]  # (pass1_label, pass2_label) -> count
    pass2_missing_from_pass1: int  # pairs in pass2 with no matching pass1 judgment


def compare_passes(pass1_path: Path, pass2_path: Path) -> AgreementReport:
    pass1 = latest_by_pair(iter_judgments(pass1_path))
    pass2 = latest_by_pair(iter_judgments(pass2_path))

    common = [pair for pair in pass2 if pair in pass1]
    missing = len(pass2) - len(common)

    labels_a = [_label(pass1[pair].grade) for pair in common]
    labels_b = [_label(pass2[pair].grade) for pair in common]

    n = len(common)
    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n if n else float("nan")

    confusion: dict[tuple[Label, Label], int] = {}
    for a, b in zip(labels_a, labels_b):
        confusion[(a, b)] = confusion.get((a, b), 0) + 1

    kappa = cohens_kappa(labels_a, labels_b)

    if n:
        categories = sorted(set(labels_a) | set(labels_b), key=str)
        row_totals = {c: sum(1 for x in labels_a if x == c) for c in categories}
        col_totals = {c: sum(1 for x in labels_b if x == c) for c in categories}
        pe = sum(row_totals[c] * col_totals[c] for c in categories) / (n * n)
    else:
        pe = float("nan")

    return AgreementReport(
        n_pairs=n,
        observed_agreement=po,
        expected_agreement=pe,
        kappa=kappa,
        confusion=confusion,
        pass2_missing_from_pass1=missing,
    )
