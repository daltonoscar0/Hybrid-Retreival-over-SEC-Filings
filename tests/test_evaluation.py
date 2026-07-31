"""ranx-backed metrics, bootstrap CIs, and the pytrec_eval cross-check.

The pytrec_eval wiring (deliverable 7) lives here as a repeatable assertion
rather than a one-off script run: `test_ranx_agrees_with_pytrec_eval_on_ndcg10`
is the "once, wired, and it foreclosed a category of doubt" check, and it
stays true on every future run rather than being true once and undocumented.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from ticker.evaluation import (
    bootstrap_ci,
    build_eval_report,
    discover_runs,
    evaluate_one_metric,
    load_run_jsonl,
    ndcg10_cross_check,
    render_latex,
    render_markdown,
)
from ticker.qrels import Judgment, append_judgment, load_qrels_dict


def _qrels_dict() -> dict[str, dict[str, int]]:
    return {
        "q1": {"d1": 3, "d2": 1, "d3": 0},
        "q2": {"d1": 0, "d2": 2, "d4": 3},
        "q3": {"d5": 1, "d6": 2},
    }


def _run_dict(good: bool) -> dict[str, dict[str, float]]:
    if good:
        return {
            "q1": {"d1": 3.0, "d2": 2.0, "d3": 1.0},
            "q2": {"d4": 3.0, "d2": 2.0, "d1": 1.0},
            "q3": {"d6": 2.0, "d5": 1.0},
        }
    return {
        "q1": {"d3": 3.0, "d2": 2.0, "d1": 1.0},
        "q2": {"d1": 3.0, "d4": 2.0, "d2": 1.0},
        "q3": {"d5": 2.0, "d6": 1.0},
    }


def test_load_run_jsonl_groups_by_query(tmp_path: Path):
    path = tmp_path / "bm25.jsonl"
    path.write_text(
        '{"query_id": "q1", "chunk_id": "c1", "score": 2.5}\n'
        '{"query_id": "q1", "chunk_id": "c2", "score": 1.0}\n'
        '{"query_id": "q2", "chunk_id": "c3", "score": 9.0}\n'
    )
    run = load_run_jsonl(path)
    assert run == {"q1": {"c1": 2.5, "c2": 1.0}, "q2": {"c3": 9.0}}


def test_discover_runs_on_missing_dir_returns_empty(tmp_path: Path):
    assert discover_runs(tmp_path / "nope") == {}


def test_discover_runs_maps_stem_to_path_sorted(tmp_path: Path):
    (tmp_path / "dense.jsonl").write_text("")
    (tmp_path / "bm25.jsonl").write_text("")
    runs = discover_runs(tmp_path)
    assert list(runs.keys()) == ["bm25", "dense"]


def test_bootstrap_ci_contains_the_mean():
    scores = np.array([0.9, 0.8, 0.85, 0.95, 0.7, 0.88])
    lo, hi = bootstrap_ci(scores, n_boot=500, seed=1)
    assert lo <= scores.mean() <= hi


def test_bootstrap_ci_deterministic_given_seed():
    scores = np.array([0.1, 0.9, 0.5, 0.3])
    a = bootstrap_ci(scores, n_boot=200, seed=7)
    b = bootstrap_ci(scores, n_boot=200, seed=7)
    assert a == b


def test_bootstrap_ci_empty_scores_is_nan():
    lo, hi = bootstrap_ci(np.array([]), n_boot=100)
    assert math.isnan(lo) and math.isnan(hi)


def test_build_eval_report_single_system_has_no_ranx_report():
    report = build_eval_report(
        Path("standard.jsonl"), _qrels_dict(), {"bm25": _run_dict(good=True)}, n_boot=100,
    )
    assert len(report.systems) == 1
    assert report.ranx_report is None
    assert report.systems[0].metrics["ndcg@10"] == pytest.approx(1.0)


def test_build_eval_report_two_systems_produces_ranx_report():
    runs = {"good": _run_dict(good=True), "bad": _run_dict(good=False)}
    report = build_eval_report(Path("standard.jsonl"), _qrels_dict(), runs, n_boot=100)
    assert report.ranx_report is not None
    good = next(s for s in report.systems if s.name == "good")
    bad = next(s for s in report.systems if s.name == "bad")
    assert good.metrics["ndcg@10"] > bad.metrics["ndcg@10"]


def test_render_markdown_single_system_has_no_significance_claim():
    report = build_eval_report(
        Path("standard.jsonl"), _qrels_dict(), {"bm25": _run_dict(good=True)}, n_boot=50,
    )
    markdown = render_markdown(report)
    assert "bm25" in markdown
    assert "no pairwise significance test" in markdown


def test_render_markdown_two_systems_includes_table_rows():
    runs = {"good": _run_dict(good=True), "bad": _run_dict(good=False)}
    report = build_eval_report(Path("standard.jsonl"), _qrels_dict(), runs, n_boot=50)
    markdown = render_markdown(report)
    assert "| a | good |" in markdown or "| b | good |" in markdown
    assert "good" in markdown and "bad" in markdown


def test_render_latex_single_system_is_a_plain_table():
    report = build_eval_report(
        Path("standard.jsonl"), _qrels_dict(), {"bm25": _run_dict(good=True)}, n_boot=50,
    )
    latex = render_latex(report)
    assert "\\begin{tabular}" in latex
    assert "bm25" in latex


def test_render_latex_two_systems_uses_ranx_report():
    runs = {"good": _run_dict(good=True), "bad": _run_dict(good=False)}
    report = build_eval_report(Path("standard.jsonl"), _qrels_dict(), runs, n_boot=50)
    latex = render_latex(report)
    assert "\\caption" in latex


def test_evaluate_one_metric_zero_fills_queries_missing_from_run():
    from ranx import Qrels, Run

    qrels = Qrels.from_dict({"q1": {"d1": 1}, "q2": {"d2": 1}})
    run = Run.from_dict({"q1": {"d1": 1.0}})  # q2 never retrieved anything
    scores = evaluate_one_metric(qrels, run, "ndcg@10")
    assert len(scores) == 2
    assert min(scores) == 0.0


def test_load_qrels_dict_feeds_evaluation_directly(tmp_path: Path):
    path = tmp_path / "standard.jsonl"
    append_judgment(path, Judgment("q1", "d1", 3, "t", "s"))
    append_judgment(path, Judgment("q1", "d2", 1, "t", "s"))
    qrels_dict = load_qrels_dict(path)
    report = build_eval_report(path, qrels_dict, {"bm25": {"q1": {"d1": 2.0, "d2": 1.0}}}, n_boot=20)
    assert report.systems[0].metrics["ndcg@10"] == pytest.approx(1.0)


def test_ranx_agrees_with_pytrec_eval_on_ndcg10():
    """Deliverable 7: wire pytrec_eval once, assert ranx agrees on nDCG@10
    to 4 decimals for one system."""
    ranx_mean, pytrec_mean = ndcg10_cross_check(_qrels_dict(), _run_dict(good=True))
    assert round(ranx_mean, 4) == round(pytrec_mean, 4)


def test_ranx_agrees_with_pytrec_eval_on_ndcg10_for_an_imperfect_run():
    ranx_mean, pytrec_mean = ndcg10_cross_check(_qrels_dict(), _run_dict(good=False))
    assert round(ranx_mean, 4) == round(pytrec_mean, 4)
