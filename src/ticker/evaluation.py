"""Metrics, bootstrap CIs, and paired significance tests for the ablation
ladder, plus the run-file contract retrieval systems write against.

The run-file contract
-----------------------
Nothing in this repo builds a retriever; that is a different agent's job,
gated on this module existing and running. The contract a retriever needs to
satisfy is exactly this: for every system, write one JSONL file to
`data/runs/<system_name>.jsonl`, one line per (query, retrieved chunk):

    {"query_id": "q01", "chunk_id": "<chunk_id>", "score": 12.34}

Higher score ranks first; ties break arbitrarily. The file stem becomes the
system's name in every table this module produces (`bm25.jsonl` -> `bm25`).
`scripts/evaluate.py` discovers every `*.jsonl` under `--runs-dir` and scores
it against whichever qrels file it is pointed at. No index format, no
retriever base class, no shared query-execution code -- a run file is the
entire interface, so BM25, dense, RRF, and weighted fusion can each be
produced by whatever code is easiest to write and still land in the same
table.

Metrics and how they are computed
-----------------------------------
nDCG@10, MRR@10, Recall@100 via `ranx.evaluate` (Jarvelin-Kekalainen linear
gain, matching `pytrec_eval`'s `ndcg_cut`; see `ndcg10_cross_check` and the
test that pins the two libraries to agreement within 1e-4). Confidence
intervals are the percentile bootstrap over queries (resample query IDs with
replacement, recompute the mean, repeat `n_boot` times), because ranx does
not ship one and the analytic nDCG variance is not worth deriving when
resampling is a dozen lines. Paired significance between every system pair
is `ranx.compare`'s two-sided paired Student's t-test over per-query scores,
which is what PLAN.md and REFERENCES.md both specify.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pytrec_eval
from ranx import Qrels, Run, compare
from ranx.data_structures.report import Report

from ticker.qrels import load_qrels_dict

METRICS = ["ndcg@10", "mrr@10", "recall@100"]
DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 42
DEFAULT_MAX_P = 0.05


def load_run_jsonl(path: Path) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            run.setdefault(row["query_id"], {})[row["chunk_id"]] = float(row["score"])
    return run


def discover_runs(runs_dir: Path) -> dict[str, Path]:
    """system name (file stem) -> path, sorted by name for a stable table
    row order run to run."""
    if not runs_dir.exists():
        return {}
    return {path.stem: path for path in sorted(runs_dir.glob("*.jsonl"))}


def bootstrap_ci(
    per_query_scores: np.ndarray,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `per_query_scores`, resampling
    queries (not judgments) with replacement -- the query is the unit of
    analysis here, matching how the paired t-test in `ranx.compare` treats
    it."""
    n = len(per_query_scores)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    resampled_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.integers(0, n, size=n)
        resampled_means[i] = per_query_scores[sample].mean()
    lo, hi = np.percentile(resampled_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


@dataclass(frozen=True)
class SystemResult:
    name: str
    n_queries: int
    metrics: dict[str, float]  # metric -> mean
    cis: dict[str, tuple[float, float]]  # metric -> (lo, hi)


@dataclass(frozen=True)
class EvalReport:
    qrels_path: Path
    n_queries_judged: int
    systems: list[SystemResult]
    ranx_report: Report | None  # None when fewer than 2 systems (no pairwise test to run)
    metrics: list[str]
    max_p: float


def evaluate_systems(
    qrels_dict: dict[str, dict[str, int]],
    runs: dict[str, dict[str, dict[str, float]]],
    *,
    metrics: Sequence[str] = METRICS,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    max_p: float = DEFAULT_MAX_P,
) -> tuple[list[SystemResult], Report | None]:
    """`runs` is {system_name: {query_id: {chunk_id: score}}}."""
    qrels = Qrels.from_dict(qrels_dict)
    ranx_runs = []
    results: list[SystemResult] = []

    for name, run_dict in runs.items():
        run = Run.from_dict(run_dict, name=name)
        per_query = {}
        cis = {}
        for metric in metrics:
            scores = evaluate_one_metric(qrels, run, metric)
            per_query[metric] = float(np.mean(scores)) if len(scores) else float("nan")
            cis[metric] = bootstrap_ci(scores, n_boot=n_boot, seed=seed)
        results.append(
            SystemResult(
                name=name,
                n_queries=len(qrels_dict),
                metrics=per_query,
                cis=cis,
            )
        )
        ranx_runs.append(run)

    ranx_report = None
    if len(ranx_runs) >= 2:
        ranx_report = compare(
            qrels, ranx_runs, list(metrics), stat_test="student", max_p=max_p,
            make_comparable=True,
        )

    return results, ranx_report


def build_eval_report(
    qrels_path: Path,
    qrels_dict: dict[str, dict[str, int]],
    runs: dict[str, dict[str, dict[str, float]]],
    *,
    metrics: Sequence[str] = METRICS,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    max_p: float = DEFAULT_MAX_P,
) -> EvalReport:
    systems, ranx_report = evaluate_systems(
        qrels_dict, runs, metrics=metrics, n_boot=n_boot, seed=seed, max_p=max_p
    )
    return EvalReport(
        qrels_path=qrels_path,
        n_queries_judged=len(qrels_dict),
        systems=systems,
        ranx_report=ranx_report,
        metrics=list(metrics),
        max_p=max_p,
    )


_TABLE_LABELS = "abcdefghijklmnopqrstuvwxyz"


def render_markdown(report: EvalReport) -> str:
    lines = [
        f"# Retrieval evaluation -- `{report.qrels_path}`",
        "",
        f"{report.n_queries_judged} queries judged, {len(report.systems)} system(s) scored.",
        "",
    ]
    header = "| # | system | " + " | ".join(m.upper() for m in report.metrics) + " |"
    sep = "|---|---|" + "---|" * len(report.metrics)
    lines += [header, sep]

    for i, system in enumerate(report.systems):
        label = _TABLE_LABELS[i] if i < len(_TABLE_LABELS) else str(i)
        cells = []
        for metric in report.metrics:
            mean = system.metrics[metric]
            lo, hi = system.cis[metric]
            superscript = ""
            if report.ranx_report is not None:
                superscript = report.ranx_report.get_superscript_for_table(system.name, metric)
            cells.append(f"{mean:.4f}{superscript} (95% CI {lo:.4f}-{hi:.4f})")
        lines.append(f"| {label} | {system.name} | " + " | ".join(cells) + " |")

    lines.append("")
    if report.ranx_report is not None:
        lines.append(
            f"Superscript letters mark a statistically significant paired "
            f"t-test win over that lettered system (p < {report.max_p})."
        )
    else:
        lines.append(
            "Fewer than two systems scored -- no pairwise significance test to run yet."
        )
    lines.append(
        f"95% CIs are a {DEFAULT_N_BOOT}-resample percentile bootstrap over queries."
    )
    return "\n".join(lines) + "\n"


def render_latex(report: EvalReport) -> str:
    if report.ranx_report is not None:
        return report.ranx_report.to_latex()

    lines = [
        "% fewer than two systems scored -- ranx.compare needs 2+ for a "
        "significance table; this is a plain table of point estimates only.",
        "\\begin{tabular}{l" + "c" * len(report.metrics) + "}",
        "\\toprule",
        "System & " + " & ".join(m.upper() for m in report.metrics) + " \\\\",
        "\\midrule",
    ]
    for system in report.systems:
        cells = [f"{system.metrics[m]:.4f}" for m in report.metrics]
        lines.append(f"{system.name} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def evaluate_one_metric(qrels: Qrels, run: Run, metric: str) -> np.ndarray:
    from ranx import evaluate as ranx_evaluate

    scores = ranx_evaluate(qrels, run, metric, return_mean=False, make_comparable=True)
    return np.asarray(scores)


def ndcg10_cross_check(
    qrels_dict: dict[str, dict[str, int]], run_dict: dict[str, dict[str, float]]
) -> tuple[float, float]:
    """Mean nDCG@10 for one system computed two ways: `ranx` and
    `pytrec_eval`. Returns (ranx_mean, pytrec_eval_mean); the caller asserts
    they agree to 4 decimals. Wired once, per PLAN.md, to foreclose the
    "does ranx actually implement nDCG correctly" doubt rather than assert
    it."""
    qrels = Qrels.from_dict(qrels_dict)
    run = Run.from_dict(run_dict)
    ranx_scores = evaluate_one_metric(qrels, run, "ndcg@10")
    ranx_mean = float(np.mean(ranx_scores)) if len(ranx_scores) else float("nan")

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, {"ndcg_cut.10"})
    # pytrec_eval expects every query in the run to have at least an empty
    # dict; queries with no retrieved chunks would otherwise raise.
    pytrec_run = {qid: run_dict.get(qid, {}) for qid in qrels_dict}
    pytrec_scores = evaluator.evaluate(pytrec_run)
    pytrec_mean = float(np.mean([v["ndcg_cut_10"] for v in pytrec_scores.values()]))

    return ranx_mean, pytrec_mean
