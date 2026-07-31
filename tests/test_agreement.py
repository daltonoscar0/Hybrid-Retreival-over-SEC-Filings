"""Cohen's kappa arithmetic checked against hand-computed values, plus the
pass1/pass2 join logic. Runs against tmp_path only.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from ticker.agreement import cohens_kappa, compare_passes
from ticker.qrels import Judgment, append_judgment


def test_perfect_agreement_gives_kappa_one():
    labels = [0, 1, 2, 3, 0, 1, 2, 3]
    assert cohens_kappa(labels, labels) == pytest.approx(1.0)


def test_uniform_identical_labels_give_kappa_one():
    # po = pe = 1: the 0/0 case, defined as perfect agreement rather than nan
    labels = ["A"] * 5
    assert cohens_kappa(labels, labels) == 1.0


def test_classic_two_category_example_matches_hand_calculation():
    # Confusion matrix:
    #            b=A   b=B
    #   a=A       10     5
    #   a=B        5    10
    # po = 20/30 = 0.6667 ; pe = (15*15 + 15*15) / 900 = 0.5
    # kappa = (0.6667 - 0.5) / (1 - 0.5) = 0.3333
    labels_a = ["A"] * 10 + ["A"] * 5 + ["B"] * 5 + ["B"] * 10
    labels_b = ["A"] * 10 + ["B"] * 5 + ["A"] * 5 + ["B"] * 10
    kappa = cohens_kappa(labels_a, labels_b)
    assert kappa == pytest.approx(1 / 3)


def test_empty_input_is_nan():
    assert math.isnan(cohens_kappa([], []))


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        cohens_kappa([1, 2], [1])


def test_compare_passes_joins_on_common_pairs(tmp_path: Path):
    pass1 = tmp_path / "standard.jsonl"
    pass2 = tmp_path / "standard.rejudge.jsonl"

    append_judgment(pass1, Judgment("q1", "c1", 2, "t", "s1"))
    append_judgment(pass1, Judgment("q1", "c2", 0, "t", "s1"))
    append_judgment(pass1, Judgment("q1", "c3", None, "t", "s1"))  # not rejudged
    append_judgment(pass1, Judgment("q2", "c4", 3, "t", "s1"))

    append_judgment(pass2, Judgment("q1", "c1", 2, "t2", "s2"))  # agree
    append_judgment(pass2, Judgment("q1", "c2", 1, "t2", "s2"))  # disagree
    append_judgment(pass2, Judgment("q2", "c4", 3, "t2", "s2"))  # agree

    report = compare_passes(pass1, pass2)

    assert report.n_pairs == 3
    assert report.pass2_missing_from_pass1 == 0
    assert report.observed_agreement == pytest.approx(2 / 3)
    assert report.confusion == {(2, 2): 1, (0, 1): 1, (3, 3): 1}


def test_compare_passes_reports_pairs_missing_from_pass1(tmp_path: Path):
    pass1 = tmp_path / "standard.jsonl"
    pass2 = tmp_path / "standard.rejudge.jsonl"

    append_judgment(pass1, Judgment("q1", "c1", 2, "t", "s1"))
    append_judgment(pass2, Judgment("q1", "c1", 2, "t2", "s2"))
    append_judgment(pass2, Judgment("q1", "c-not-in-pass1", 1, "t2", "s2"))

    report = compare_passes(pass1, pass2)
    assert report.n_pairs == 1
    assert report.pass2_missing_from_pass1 == 1


def test_compare_passes_uses_latest_grade_per_pair(tmp_path: Path):
    pass1 = tmp_path / "standard.jsonl"
    pass2 = tmp_path / "standard.rejudge.jsonl"

    append_judgment(pass1, Judgment("q1", "c1", 0, "t", "s1"))
    append_judgment(pass1, Judgment("q1", "c1", 3, "t", "s1"))  # redo, latest wins
    append_judgment(pass2, Judgment("q1", "c1", 3, "t2", "s2"))

    report = compare_passes(pass1, pass2)
    assert report.confusion == {(3, 3): 1}


def test_skip_is_its_own_category_not_dropped(tmp_path: Path):
    pass1 = tmp_path / "standard.jsonl"
    pass2 = tmp_path / "standard.rejudge.jsonl"

    append_judgment(pass1, Judgment("q1", "c1", None, "t", "s1"))  # skipped pass 1
    append_judgment(pass2, Judgment("q1", "c1", 2, "t2", "s2"))  # graded pass 2

    report = compare_passes(pass1, pass2)
    assert report.n_pairs == 1
    assert report.confusion == {("skip", 2): 1}
