"""Tests for the Phase 5 validation numbers.

The AUC implementation is checked against sklearn rather than against
hand-computed values, for the same reason `ticker.evaluation` is checked
against pytrec_eval: an independent implementation catches a wrong tie rule
or a wrong direction, and a hand-computed expectation written by whoever
wrote the code catches neither.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest
from sklearn.metrics import roc_auc_score

from ticker import db
from ticker.records import Filing, Section, Sentence
from ticker.validation import (
    JoinDiagnostics,
    bootstrap_ci,
    collect_diff_agreement,
    mean_novelty_by_item,
    roc_auc,
)

CIK = 1234
T0 = datetime(2022, 2, 1, tzinfo=timezone.utc)
T1 = datetime(2023, 2, 1, tzinfo=timezone.utc)


def _add(con, accession, filed_at, item, sentences):
    db.insert_filing(
        con,
        Filing(
            accession=accession,
            cik=CIK,
            ticker="TST",
            sector="semiconductors",
            form="10-K",
            filed_at=filed_at,
            period_end=None,
            url=f"https://example.com/{accession}",
        ),
    )
    section_id = f"{accession}#{item}"
    db.insert_section(
        con,
        Section(
            section_id=section_id,
            accession=accession,
            item=item,
            text=" ".join(sentences),
            char_start=0,
            char_end=1,
        ),
    )
    for ordinal, text in enumerate(sentences):
        db.insert_sentence(
            con,
            Sentence(
                sentence_id=f"{section_id}:{ordinal}",
                section_id=section_id,
                ordinal=ordinal,
                text=text,
                filed_at=filed_at,
            ),
        )
    return section_id


@pytest.fixture
def connection():
    c = db.connect(":memory:")
    db.create_schema(c)
    return c


# --- roc_auc against sklearn ---------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_roc_auc_agrees_with_sklearn_on_continuous_scores(seed):
    rng = random.Random(seed)
    scores = [rng.gauss(0, 1) for _ in range(200)]
    labels = [rng.random() < 0.4 for _ in range(200)]
    if len(set(labels)) < 2:
        pytest.skip("degenerate draw")
    assert roc_auc(scores, labels) == pytest.approx(
        roc_auc_score(labels, scores), abs=1e-12
    )


@pytest.mark.parametrize("seed", range(8))
def test_roc_auc_agrees_with_sklearn_under_heavy_ties(seed):
    """Ties are the case a naive AUC gets wrong, so they get their own test.

    Three distinct score values over 200 rows means most pairs are tied, and
    a rank rule that breaks ties by position instead of averaging them
    disagrees with sklearn immediately.
    """
    rng = random.Random(seed)
    scores = [rng.choice([0.0, 1.0, 2.0]) for _ in range(200)]
    labels = [rng.random() < 0.5 for _ in range(200)]
    if len(set(labels)) < 2:
        pytest.skip("degenerate draw")
    assert roc_auc(scores, labels) == pytest.approx(
        roc_auc_score(labels, scores), abs=1e-12
    )


def test_roc_auc_is_one_when_novelty_ranks_every_changed_sentence_first():
    assert roc_auc([3.0, 2.0, 1.0, 0.0], [True, True, False, False]) == 1.0


def test_roc_auc_is_half_when_a_class_is_absent():
    assert roc_auc([1.0, 2.0, 3.0], [True, True, True]) == 0.5
    assert roc_auc([1.0, 2.0, 3.0], [False, False, False]) == 0.5


def test_roc_auc_is_half_for_all_tied_scores():
    assert roc_auc([1.0] * 6, [True, False, True, False, True, False]) == 0.5


# --- the join ------------------------------------------------------------


def test_the_join_maps_diff_ordinals_onto_the_right_sentence_ids(connection):
    """The end-to-end check that the join is not off by anything.

    The prior section keeps sentences a and b and drops nothing; the current
    section keeps a and b and inserts x and y. Scoring only the inserted
    sentences high must produce a perfect AUC. An off-by-one in the ordinal
    to sentence_id map moves the high scores onto unchanged sentences and
    drives this to 0 rather than to 0.5, which is why the assertion is on
    1.0 exactly and not on "better than chance".
    """
    _add(connection, "old", T0, "7", ["alpha alpha alpha", "beta beta beta"])
    current = _add(
        connection,
        "new",
        T1,
        "7",
        ["alpha alpha alpha", "beta beta beta", "xray xray xray", "yankee yankee yankee"],
    )
    novelty = {
        f"{current}:0": 0.0,
        f"{current}:1": 0.0,
        f"{current}:2": 9.0,
        f"{current}:3": 9.0,
    }
    observations, diag = collect_diff_agreement(connection, novelty, [current])

    assert diag.sections_used == 1
    assert diag.sentences_joined == 4
    assert diag.join_rate == 1.0
    scores = [s for o in observations for s in o.novelty]
    labels = [c for o in observations for c in o.changed]
    assert labels == [False, False, True, True]
    assert roc_auc(scores, labels) == 1.0


def test_a_section_with_no_prior_is_excluded_and_counted(connection):
    first = _add(connection, "only", T0, "7", ["alpha", "beta"])
    novelty = {f"{first}:0": 1.0, f"{first}:1": 2.0}
    observations, diag = collect_diff_agreement(connection, novelty, [first])
    assert observations == []
    assert diag.sections_unaligned == 1
    assert diag.sections_used == 0


def test_unscored_sentences_are_dropped_and_counted_not_defaulted(connection):
    """A missing novelty score must not become a zero.

    Defaulting an unscored sentence to 0.0 would quietly enter it as the
    least novel sentence in the corpus, which is a fabricated observation
    with a real effect on the AUC.
    """
    _add(connection, "old", T0, "7", ["alpha alpha", "beta beta"])
    current = _add(connection, "new", T1, "7", ["alpha alpha", "gamma gamma"])
    observations, diag = collect_diff_agreement(
        connection, {f"{current}:0": 1.0}, [current]
    )
    assert diag.sentences_diffed == 2
    assert diag.sentences_joined == 1
    assert diag.join_rate == pytest.approx(0.5)
    assert f"{current}:1" in diag.unjoined_examples
    assert [len(o.novelty) for o in observations] == [1]


def test_a_gap_in_sentence_ordinals_raises_rather_than_misjoining(connection):
    _add(connection, "old", T0, "7", ["alpha", "beta"])
    current = _add(connection, "new", T1, "7", ["alpha", "beta"])
    connection.execute(
        "UPDATE sentences SET ordinal = 5 WHERE sentence_id = ?", [f"{current}:1"]
    )
    with pytest.raises(RuntimeError, match="gap in its sentence ordinals"):
        collect_diff_agreement(connection, {f"{current}:0": 1.0}, [current])


def test_the_prior_is_strictly_earlier(connection):
    """align_prior_section enforces this; the check is that validation uses it.

    A same-day amendment must not become the prior, because diffing a
    filing against its own same-day twin produces an all-unchanged label set
    and pulls the AUC toward 0.5 for a reason unrelated to novelty.
    """
    _add(connection, "same-day", T1, "7", ["alpha", "beta"])
    current = _add(connection, "current", T1, "7", ["alpha", "gamma"])
    observations, diag = collect_diff_agreement(
        connection, {f"{current}:0": 1.0, f"{current}:1": 2.0}, [current]
    )
    assert diag.sections_unaligned == 1
    assert observations == []


# --- bootstrap -----------------------------------------------------------


def _observation(section_id, novelty, changed):
    from ticker.validation import SectionObservations

    return SectionObservations(
        section_id=section_id,
        item="7",
        prior_accession="prior",
        gap_days=365,
        novelty=tuple(novelty),
        changed=tuple(changed),
    )


def test_bootstrap_ci_brackets_its_point_estimate():
    rng = random.Random(0)
    observations = [
        _observation(
            f"s{i}",
            [rng.gauss(1, 1) for _ in range(20)] + [rng.gauss(0, 1) for _ in range(20)],
            [True] * 20 + [False] * 20,
        )
        for i in range(30)
    ]
    point, lower, upper = bootstrap_ci(observations, resamples=300, seed=1)
    assert lower <= point <= upper
    assert point > 0.5


def test_bootstrap_ci_is_deterministic_under_a_seed():
    observations = [_observation(f"s{i}", [1.0, 0.0], [True, False]) for i in range(10)]
    assert bootstrap_ci(observations, resamples=200, seed=7) == bootstrap_ci(
        observations, resamples=200, seed=7
    )


def test_bootstrap_ci_resamples_sections_not_sentences():
    """The cluster bootstrap must widen the interval, not just shift it.

    Every section here is internally perfectly separating but the sections
    disagree with each other, so sentence-level resampling would see 200
    independent rows and return a tight interval, while section-level
    resampling sees 10 clusters and cannot. Asserting the interval is wide
    is asserting the unit of resampling is the section.
    """
    good = [_observation(f"g{i}", [1.0] * 10 + [0.0] * 10, [True] * 10 + [False] * 10) for i in range(5)]
    bad = [_observation(f"b{i}", [0.0] * 10 + [1.0] * 10, [True] * 10 + [False] * 10) for i in range(5)]
    _, lower, upper = bootstrap_ci(good + bad, resamples=400, seed=3)
    assert upper - lower > 0.25


def test_bootstrap_ci_on_no_observations_is_half():
    assert bootstrap_ci([], resamples=10) == (0.5, 0.5, 0.5)


# --- 5.2 -----------------------------------------------------------------


def test_mean_novelty_by_item_reports_n_mean_and_a_bracketing_interval():
    rows = [("1", 0.0), ("1", 1.0), ("7", 5.0), ("7", 7.0), ("7", 6.0)]
    out = mean_novelty_by_item(rows, resamples=200, seed=0)
    assert set(out) == {"1", "7"}
    n, mean, lower, upper = out["7"]
    assert n == 3
    assert mean == pytest.approx(6.0)
    assert lower <= mean <= upper


def test_mean_novelty_by_item_separates_items():
    rows = [("1", 0.0)] * 50 + [("7", 10.0)] * 50
    out = mean_novelty_by_item(rows, resamples=200, seed=0)
    assert out["1"][1] < out["7"][1]


def test_two_items_with_equal_n_get_independent_resamples():
    """Guards against reseeding the RNG inside the per-item loop.

    Two items with identical n and identical value multisets must still draw
    different resample positions. Under a per-item reseed they draw the same
    ones, the two intervals become byte-identical, and a table built to be
    read across items reports intervals that move together.
    """
    values = [float(i) for i in range(40)]
    rows = [("1", v) for v in values] + [("7", v) for v in values]
    out = mean_novelty_by_item(rows, resamples=300, seed=0)
    assert out["1"][0] == out["7"][0]
    assert (out["1"][2], out["1"][3]) != (out["7"][2], out["7"][3])


def test_load_scores_refuses_a_file_without_the_raw_contrast_marker(tmp_path):
    """The type firewall around invariant 3 does not survive serialization.

    In process, the display z-score is its own class and the aggregation path
    rejects it by isinstance. On disk both are a float under `novelty`. The
    marker is what carries the distinction across, so reading a file that
    lacks it has to fail rather than proceed.
    """
    import json

    from ticker.validation import load_scores

    path = tmp_path / "scores.jsonl"
    path.write_text(
        json.dumps(
            {"sentence_id": "s1", "novelty": 1.0, "section_id": "sec", "item": "7"}
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="score_kind"):
        load_scores(path)


def test_load_scores_refuses_a_file_marked_as_z_scores(tmp_path):
    import json

    from ticker.validation import load_scores

    path = tmp_path / "scores.jsonl"
    path.write_text(
        json.dumps(
            {
                "sentence_id": "s1",
                "score_kind": "zscore_display",
                "novelty": 1.0,
                "section_id": "sec",
                "item": "7",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="zscore_display"):
        load_scores(path)


def test_load_scores_accepts_a_raw_contrast_file(tmp_path):
    import json

    from ticker.validation import RAW_CONTRAST, load_scores

    path = tmp_path / "scores.jsonl"
    with path.open("w") as f:
        for i in range(3):
            f.write(
                json.dumps(
                    {
                        "sentence_id": f"s{i}",
                        "score_kind": RAW_CONTRAST,
                        "novelty": float(i),
                        "section_id": "sec",
                        "item": "7",
                    }
                )
                + "\n"
            )
    by_sentence, item_rows, sections = load_scores(path)
    assert len(by_sentence) == 3
    assert len(item_rows) == 3
    assert sections == ["sec"]


def _write_scores(path, records):
    import json

    from ticker.validation import RAW_CONTRAST

    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps({"score_kind": RAW_CONTRAST, **record}) + "\n")


def test_coverage_reports_a_complete_score_file_as_complete(connection, tmp_path):
    from ticker.validation import coverage

    section = _add(connection, "acc1", T0, "7", ["alpha", "beta"])
    path = tmp_path / "s.jsonl"
    _write_scores(
        path,
        [
            {
                "sentence_id": f"{section}:0",
                "novelty": 1.0,
                "section_id": section,
                "item": "7",
                "accession": "acc1",
                "cik": CIK,
                "form": "10-K",
            }
        ],
    )
    cov = coverage(connection, path)
    assert cov.forms == ("10-K",)
    assert cov.complete
    assert cov.accessions_in_file == cov.accessions_in_corpus == 1


def test_coverage_names_the_filings_a_truncated_pass_never_reached(
    connection, tmp_path
):
    """The failure JoinDiagnostics structurally cannot see.

    A firm the pass never reached contributes no sections to the file, so
    nothing inside the file registers as missing and the join rate reads
    100%. Only a reconciliation against the corpus catches it.
    """
    from ticker.validation import coverage

    scored = _add(connection, "acc1", T0, "7", ["alpha", "beta"])
    _add(connection, "acc2", T1, "7", ["gamma", "delta"])
    path = tmp_path / "s.jsonl"
    _write_scores(
        path,
        [
            {
                "sentence_id": f"{scored}:0",
                "novelty": 1.0,
                "section_id": scored,
                "item": "7",
                "accession": "acc1",
                "cik": CIK,
                "form": "10-K",
            }
        ],
    )
    cov = coverage(connection, path)
    assert not cov.complete
    assert cov.missing_accessions == ("acc2",)
    assert cov.accessions_in_file == 1
    assert cov.accessions_in_corpus == 2


def test_coverage_only_reconciles_against_the_forms_the_file_contains(
    connection, tmp_path
):
    """A 10-K-only file is not incomplete for having no 10-Q rows.

    Scoping novelty to one form is a scope decision. Counting every 10-Q as a
    missing filing would turn that decision into a permanent false alarm and
    train the reader to ignore the warning.
    """
    from ticker.validation import coverage

    section = _add(connection, "acc1", T0, "7", ["alpha", "beta"])
    db.insert_filing(
        connection,
        Filing(
            accession="q1",
            cik=CIK,
            ticker="TST",
            sector="semiconductors",
            form="10-Q",
            filed_at=T1,
            period_end=None,
            url="https://example.com/q1",
        ),
    )
    db.insert_section(
        connection,
        Section(
            section_id="q1#2",
            accession="q1",
            item="2",
            text="quarterly",
            char_start=0,
            char_end=1,
        ),
    )
    path = tmp_path / "s.jsonl"
    _write_scores(
        path,
        [
            {
                "sentence_id": f"{section}:0",
                "novelty": 1.0,
                "section_id": section,
                "item": "7",
                "accession": "acc1",
                "cik": CIK,
                "form": "10-K",
            }
        ],
    )
    cov = coverage(connection, path)
    assert cov.forms == ("10-K",)
    assert cov.complete


def test_join_diagnostics_join_rate_is_zero_when_nothing_was_diffed():
    assert JoinDiagnostics().join_rate == 0.0
