"""Fusion arithmetic, normalization, the split, and the train/test firewall.

The RRF case is computed by hand below rather than compared against another
implementation, because the point of the test is that this module implements
the formula in Cormack et al. and not some neighboring one. A second library
agreeing would only mean both made the same choice about, say, whether rank
is 0-based.

`test_tune_weights_returns_no_score` and `test_held_out_score_refuses_overlap`
are the pair that matters. Together they say: the tuning call cannot hand you
a number, and the scoring call will not accept the tuning queries. There is no
single call that tunes and reports.
"""

from __future__ import annotations

import importlib.util
import sys

from pathlib import Path

import pytest

from ticker.evaluation import discover_runs, load_run_jsonl
from ticker.fusion import (
    DEFAULT_RRF_K,
    held_out_score,
    normalize_run,
    rrf,
    score_run,
    split_queries,
    tune_weights,
    usable_query_ids,
    weighted,
    write_run_jsonl,
)

# scripts/ is not a package; import scripts/fuse.py by file path, the same way
# tests/test_bm25.py imports scripts/index.py.
_SPEC = importlib.util.spec_from_file_location(
    "ticker_scripts_fuse",
    Path(__file__).resolve().parents[1] / "scripts" / "fuse.py",
)
fuse_script = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = fuse_script
_SPEC.loader.exec_module(fuse_script)




def _assert_runs_close(actual: dict, expected: dict) -> None:
    """pytest.approx refuses nested dicts, so compare one query at a time."""
    assert sorted(actual) == sorted(expected)
    for query_id in expected:
        assert actual[query_id] == pytest.approx(expected[query_id])


def _two_systems() -> dict[str, dict[str, dict[str, float]]]:
    """bm25 ranks c1 > c2 > c3, dense ranks c3 > c1 > c4, on one query."""
    return {
        "bm25": {"q1": {"c1": 12.0, "c2": 8.0, "c3": 2.0}},
        "dense": {"q1": {"c3": 0.91, "c1": 0.80, "c4": 0.62}},
    }


def test_rrf_matches_hand_computed_scores():
    fused = rrf(_two_systems(), k=60)

    # bm25 ranks: c1=1, c2=2, c3=3. dense ranks: c3=1, c1=2, c4=3.
    expected = {
        "c1": 1 / 61 + 1 / 62,
        "c2": 1 / 62,
        "c3": 1 / 63 + 1 / 61,
        "c4": 1 / 63,
    }
    assert fused["q1"] == pytest.approx(expected)


def test_rrf_ranks_the_document_both_systems_like_first():
    fused = rrf(_two_systems())
    order = list(fused["q1"])
    # c1 is rank 1 and 2; c3 is rank 3 and 1. 1/61 + 1/62 beats 1/63 + 1/61.
    assert order[0] == "c1"
    assert order[1] == "c3"
    assert order[-1] in {"c2", "c4"}


def test_rrf_default_k_is_the_paper_value():
    assert DEFAULT_RRF_K == 60
    assert rrf(_two_systems()) == rrf(_two_systems(), k=60)


def test_rrf_ignores_score_scale_and_reads_only_ranks():
    """Scaling one system's scores by 1000 changes no rank, so RRF must not
    move. This is the property that lets it run without normalization."""
    runs = _two_systems()
    scaled = {
        "bm25": {"q1": {c: s * 1000 for c, s in runs["bm25"]["q1"].items()}},
        "dense": runs["dense"],
    }
    _assert_runs_close(rrf(scaled), rrf(runs))


def test_rrf_breaks_score_ties_by_chunk_id_not_insertion_order():
    a = {"sys": {"q1": {"c1": 1.0, "c2": 1.0}}}
    b = {"sys": {"q1": {"c2": 1.0, "c1": 1.0}}}
    assert rrf(a) == rrf(b)
    assert list(rrf(a)["q1"]) == ["c1", "c2"]


def test_rrf_rejects_nonpositive_k():
    with pytest.raises(ValueError, match="positive"):
        rrf(_two_systems(), k=0)


def test_min_max_normalization_maps_each_query_to_zero_one():
    run = {"q1": {"c1": 12.0, "c2": 8.0, "c3": 2.0}, "q2": {"c9": 0.5}}
    normalized = normalize_run(run)

    assert normalized["q1"] == pytest.approx({"c1": 1.0, "c2": 0.6, "c3": 0.0})
    # A single-document query has zero spread; the epsilon floor keeps it
    # finite rather than dividing by zero.
    assert normalized["q2"]["c9"] == pytest.approx(0.0)


def test_min_max_normalization_is_per_query_not_pooled():
    """A query whose raw scores are all small must still reach 1.0. Pooling
    across queries would leave it near zero and let the other query's
    documents own the fused ranking."""
    run = {"big": {"c1": 100.0, "c2": 50.0}, "small": {"c3": 0.2, "c4": 0.1}}
    normalized = normalize_run(run)
    assert normalized["small"]["c3"] == pytest.approx(1.0)
    assert normalized["big"]["c1"] == pytest.approx(1.0)


def test_normalize_run_none_passes_scores_through():
    run = {"q1": {"c1": 12.0, "c2": 8.0}}
    assert normalize_run(run, norm="none") == run


def test_normalize_run_rejects_unknown_norm():
    with pytest.raises(ValueError, match="unknown norm"):
        normalize_run({"q1": {"c1": 1.0}}, norm="softmax")


def test_weighted_sums_normalized_scores_not_raw_ones():
    runs = _two_systems()
    fused = weighted(runs, {"bm25": 0.5, "dense": 0.5})

    # normalized bm25: c1=1.0, c2=0.6, c3=0.0
    # normalized dense: c3=1.0, c1=(0.80-0.62)/0.29, c4=0.0
    dense_c1 = (0.80 - 0.62) / (0.91 - 0.62)
    expected = {
        "c1": 0.5 * 1.0 + 0.5 * dense_c1,
        "c2": 0.5 * 0.6,
        "c3": 0.5 * 0.0 + 0.5 * 1.0,
        "c4": 0.5 * 0.0,
    }
    assert fused["q1"] == pytest.approx(expected)


def test_weighted_without_normalization_lets_bm25_own_the_ranking():
    """The failure mode the normalization step exists for: raw BM25 scores in
    the tens against cosines under one, equally weighted, reproduce BM25's
    ranking exactly."""
    runs = _two_systems()
    raw = weighted(runs, {"bm25": 0.5, "dense": 0.5}, norm="none")
    bm25_order = ["c1", "c2", "c3"]
    assert [c for c in raw["q1"] if c in bm25_order] == bm25_order


def test_weighted_weight_of_one_reproduces_that_systems_ranking():
    runs = _two_systems()
    fused = weighted(runs, {"bm25": 1.0, "dense": 0.0})
    top = [c for c in fused["q1"]][:2]
    assert top == ["c1", "c2"]


def test_weighted_requires_a_weight_for_every_run():
    with pytest.raises(ValueError, match="exactly the fused systems"):
        weighted(_two_systems(), {"bm25": 1.0})


def test_weighted_rejects_a_weight_for_an_unknown_system():
    with pytest.raises(ValueError, match="exactly the fused systems"):
        weighted(_two_systems(), {"bm25": 0.5, "dense": 0.3, "colbert": 0.2})


def test_split_queries_is_deterministic_for_a_seed():
    ids = [f"q{i:02d}" for i in range(1, 41)]
    first = split_queries(ids, frac=0.5, seed=42)
    second = split_queries(ids, frac=0.5, seed=42)
    assert first == second


def test_split_queries_ignores_input_order_and_duplicates():
    ids = [f"q{i:02d}" for i in range(1, 41)]
    shuffled = list(reversed(ids)) + ids[:5]
    assert split_queries(shuffled, frac=0.5, seed=42) == split_queries(
        ids, frac=0.5, seed=42
    )


def test_split_queries_partitions_without_overlap():
    ids = [f"q{i:02d}" for i in range(1, 41)]
    train, test = split_queries(ids, frac=0.6, seed=7)
    assert set(train) & set(test) == set()
    assert sorted(train + test) == ids
    assert len(train) == 24


def test_split_queries_different_seed_gives_a_different_partition():
    ids = [f"q{i:02d}" for i in range(1, 41)]
    assert split_queries(ids, frac=0.5, seed=1) != split_queries(ids, frac=0.5, seed=2)


def test_split_queries_returns_sorted_halves():
    ids = [f"q{i:02d}" for i in range(1, 41)]
    train, test = split_queries(ids, frac=0.5, seed=3)
    assert train == sorted(train)
    assert test == sorted(test)


def test_split_queries_never_returns_an_empty_side():
    train, test = split_queries(["q1", "q2", "q3"], frac=0.01, seed=0)
    assert len(train) == 1
    assert len(test) == 2


def test_split_queries_rejects_degenerate_fracs():
    ids = ["q1", "q2"]
    with pytest.raises(ValueError, match="frac"):
        split_queries(ids, frac=0.0, seed=0)
    with pytest.raises(ValueError, match="frac"):
        split_queries(ids, frac=1.0, seed=0)


def test_split_queries_rejects_a_single_query():
    with pytest.raises(ValueError, match="at least 2"):
        split_queries(["q1"], frac=0.5, seed=0)


def _tuning_fixture():
    """Four queries where dense is right and bm25 is wrong, so the grid search
    has an unambiguous optimum to find."""
    qrels = {f"q{i}": {f"good{i}": 3, f"bad{i}": 0} for i in range(1, 5)}
    runs = {
        "bm25": {f"q{i}": {f"bad{i}": 9.0, f"good{i}": 1.0} for i in range(1, 5)},
        "dense": {f"q{i}": {f"good{i}": 0.9, f"bad{i}": 0.1} for i in range(1, 5)},
    }
    return qrels, runs


def test_tune_weights_returns_no_score():
    """The structural half of the train/test firewall: tuning hands back
    weights and nothing that could be printed as a result."""
    qrels, runs = _tuning_fixture()
    result = tune_weights(qrels, runs, ["q1", "q2"], step=0.5)

    assert set(result) == {"bm25", "dense"}
    assert all(isinstance(v, float) for v in result.values())
    assert result["dense"] > result["bm25"]


def test_held_out_score_refuses_overlap():
    """The other half: the scoring call will not accept the tuning queries,
    so tune-then-score cannot collapse into one number on one split."""
    qrels, runs = _tuning_fixture()
    train, test = ["q1", "q2"], ["q3", "q4"]
    fused = weighted(runs, {"bm25": 0.0, "dense": 1.0})

    with pytest.raises(ValueError, match="both the tuning and the scoring"):
        held_out_score(qrels, fused, train, train)
    with pytest.raises(ValueError, match="both the tuning and the scoring"):
        held_out_score(qrels, fused, train, train + test)

    assert held_out_score(qrels, fused, train, test) == pytest.approx(1.0)


def test_held_out_score_and_train_score_are_separate_numbers():
    qrels, runs = _tuning_fixture()
    train, test = ["q1", "q2"], ["q3", "q4"]
    weights = tune_weights(qrels, runs, train, step=0.5)
    fused = weighted(runs, weights)

    train_score = score_run(qrels, fused, train)
    test_score = held_out_score(qrels, fused, train, test)
    assert train_score == pytest.approx(1.0)
    assert test_score == pytest.approx(1.0)


def test_tune_weights_needs_two_runs():
    qrels, runs = _tuning_fixture()
    with pytest.raises(ValueError, match="2\\+ runs"):
        tune_weights(qrels, {"bm25": runs["bm25"]}, ["q1"])


def test_tune_weights_rejects_a_train_split_with_no_usable_query():
    qrels, runs = _tuning_fixture()
    with pytest.raises(ValueError, match="nothing to tune on"):
        tune_weights(qrels, runs, ["q99"], step=0.5)


def test_usable_query_ids_drops_unjudged_and_unretrieved_queries():
    qrels = {"q1": {"c1": 1}, "q2": {"c2": 1}, "q3": {}}
    runs = {
        "bm25": {"q1": {"c1": 1.0}, "q2": {"c2": 1.0}, "q3": {"c3": 1.0}},
        "dense": {"q1": {"c1": 0.5}},
    }
    assert usable_query_ids(qrels, runs, ["q1", "q2", "q3", "q4"]) == ["q1"]


def test_score_run_requires_at_least_one_judged_query():
    qrels, runs = _tuning_fixture()
    with pytest.raises(ValueError, match="any judgment"):
        score_run(qrels, runs["bm25"], ["q99"])


def test_write_run_jsonl_round_trips_through_the_evaluation_loader(tmp_path: Path):
    fused = rrf(_two_systems())
    path = tmp_path / "rrf.jsonl"
    rows = write_run_jsonl(path, fused)

    assert rows == 4
    _assert_runs_close(load_run_jsonl(path), fused)


def test_write_run_jsonl_is_byte_stable(tmp_path: Path):
    fused = rrf(_two_systems())
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_run_jsonl(a, fused)
    write_run_jsonl(b, rrf(dict(reversed(list(_two_systems().items())))))
    assert a.read_bytes() == b.read_bytes()


def test_the_tuned_run_is_not_discoverable_by_the_ablation_ladder(tmp_path: Path):
    """The guard that matters is where the file lands, not what tuned it.

    `ticker.evaluation.discover_runs` globs the runs directory and
    `scripts/evaluate.py` scores everything it returns over every judged query.
    A tuned run sitting in that directory would be reported on the queries that
    chose its weights, with every in-process guard in this module having done
    its job. `scripts/fuse.py` writes it to a subdirectory for that reason, and
    this pins the subdirectory to being out of the glob.
    """
    runs_dir = tmp_path / "runs"
    tuned_dir = runs_dir / fuse_script.TUNED_SUBDIR
    write_run_jsonl(runs_dir / "bm25.jsonl", _two_systems()["bm25"])
    write_run_jsonl(runs_dir / "rrf.jsonl", rrf(_two_systems()))
    write_run_jsonl(tuned_dir / f"{fuse_script.WSUM_NAME}.jsonl", _two_systems()["dense"])

    discovered = set(discover_runs(runs_dir))

    assert discovered == {"bm25", "rrf"}
    assert fuse_script.WSUM_NAME not in discovered
    assert (tuned_dir / f"{fuse_script.WSUM_NAME}.jsonl").exists()
