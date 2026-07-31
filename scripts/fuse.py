#!/usr/bin/env python3
"""Fuse the single-system run files into `rrf.jsonl` and `wsum.jsonl`.

Reads the run files `scripts/search.py` wrote, writes fused run files next to
them in the same `{query_id, chunk_id, score}` contract, so
`scripts/evaluate.py` picks the fused systems up as two more rows of the
ablation ladder with no extra wiring. The fusion arithmetic and the
train/test discipline live in `ticker.fusion`; this script is the CLI shell
plus the tuning report.

The two arms have different prerequisites
------------------------------------------
RRF reads ranks only. It needs no relevance judgments and no tuning, so it
runs the day the run files exist and its output is final.

Weighted fusion has to learn its weights from judgments. Until
`data/qrels/standard.jsonl` has some, this script prints what is missing,
skips the weighted arm, and exits 0. It does not write a placeholder qrels
file and it does not invent judgments. A fabricated qrels file would turn
every number downstream of it into a measurement of the fabrication, and the
whole point of the two-qrel design is that a human made those calls.

What the report records, and why the train number is in it
-----------------------------------------------------------
`reports/fusion_tuning.md` records the split seed, the exact train and test
query ids, the tuned weights, the tuning-set metric, and the held-out metric.
The tuning-set number is printed next to the held-out one and labelled as not
a result, because a reader who sees only the held-out number cannot tell
whether the weights overfit, and the gap between the two is the answer.

Usage:
  uv run python scripts/fuse.py
  uv run python scripts/fuse.py --runs-dir /tmp/runs --qrels /tmp/synthetic.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ticker.evaluation import load_run_jsonl
from ticker.fusion import (
    DEFAULT_METRIC,
    DEFAULT_RRF_K,
    DEFAULT_WEIGHT_STEP,
    held_out_score,
    rrf,
    score_run,
    split_queries,
    tune_weights,
    usable_query_ids,
    weighted,
    write_run_jsonl,
)
from ticker.qrels import load_qrels_dict

DEFAULT_RUNS_DIR = Path("data/runs")
DEFAULT_QRELS = Path("data/qrels/standard.jsonl")
DEFAULT_REPORT = Path("reports/fusion_tuning.md")
DEFAULT_SYSTEMS = ("bm25", "dense")
DEFAULT_TRAIN_FRAC = 0.5
DEFAULT_SPLIT_SEED = 42

RRF_NAME = "rrf"
WSUM_NAME = "wsum"


def _load_systems(runs_dir: Path, systems: list[str]):
    """Returns ({name: run}, [missing names]). A missing run file is not an
    error here: the point of reporting it is that the caller runs
    `scripts/search.py` for that arm, not that this script fails."""
    runs = {}
    missing = []
    for name in systems:
        path = runs_dir / f"{name}.jsonl"
        if not path.exists():
            missing.append(name)
            continue
        runs[name] = load_run_jsonl(path)
    return runs, missing


def _report_header(runs, runs_dir: Path, rrf_k: int, rrf_rows: int) -> list[str]:
    lines = [
        "# Fusion",
        "",
        "| system | queries | rows |",
        "|---|---|---|",
    ]
    for name, run in sorted(runs.items()):
        lines.append(f"| {name} | {len(run)} | {sum(len(v) for v in run.values())} |")
    lines += [
        "",
        "## RRF",
        "",
        f"`k = {rrf_k}`, the value reported in Cormack, Clarke, Buettcher (SIGIR "
        "2009). Chosen there on TREC data and not tuned on this corpus: with no "
        "third split reserved for it, fitting `k` would fit a free parameter on "
        "the same queries the result is reported over.",
        "",
        f"Fused {len(runs)} systems into `{runs_dir / (RRF_NAME + '.jsonl')}` "
        f"({rrf_rows} rows). RRF reads ranks only, so it needs no judgments and "
        "this arm is final.",
        "",
    ]
    return lines


def _skipped_weighted_section(qrels_path: Path) -> list[str]:
    return [
        "## Weighted fusion",
        "",
        f"Skipped. `{qrels_path}` has no judgments yet, and weighted fusion "
        "learns its weights from them. No weights, no tuning-set number, and no "
        "held-out number are reported, because there is nothing to compute them "
        "from. Judge with `scripts/judge.py` and re-run this command.",
        "",
    ]


def _weighted_section(
    *,
    qrels_path: Path,
    n_judged: int,
    frac: float,
    seed: int,
    metric: str,
    step: float,
    train: list[str],
    test: list[str],
    weights: dict[str, float],
    train_metric: float,
    test_metric: float,
    single_system_test: dict[str, float],
    out_path: Path,
    rows: int,
) -> list[str]:
    weight_str = ", ".join(f"`{name}` {w:.2f}" for name, w in sorted(weights.items()))
    lines = [
        "## Weighted fusion",
        "",
        f"Qrels: `{qrels_path}`, {n_judged} judged queries, of which "
        f"{len(train) + len(test)} are also retrieved by every arm and so are "
        "usable for tuning.",
        "",
        f"Split: train fraction {frac}, seed {seed}, grid step {step}, objective "
        f"`{metric}`. {len(train)} train / {len(test)} test.",
        "",
        f"Train ids: {' '.join(train)}",
        "",
        f"Test ids: {' '.join(test)}",
        "",
        f"Tuned weights: {weight_str}",
        "",
        "| split | " + metric + " |",
        "|---|---|",
        f"| held-out (test) | **{test_metric:.4f}** |",
        f"| tuning set (train), not a result | {train_metric:.4f} |",
        "",
        "The held-out number is the one that goes in the results table. The "
        "tuning-set number is here so the gap between them is visible; a large "
        "gap means the weights fit the train queries rather than the task.",
        "",
        "Single systems on the same held-out split, for scale:",
        "",
        "| system | " + metric + " |",
        "|---|---|",
    ]
    for name, value in sorted(single_system_test.items()):
        lines.append(f"| {name} | {value:.4f} |")
    lines += [
        "",
        f"Fused run written to `{out_path}` ({rows} rows).",
        "",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--systems", nargs="+", default=list(DEFAULT_SYSTEMS))
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--step", type=float, default=DEFAULT_WEIGHT_STEP)
    args = parser.parse_args()

    runs, missing = _load_systems(args.runs_dir, args.systems)
    if missing:
        print(
            f"missing run file(s) for {missing} under {args.runs_dir}. Produce "
            "them with scripts/search.py --retriever module.path:callable, then "
            "re-run this command."
        )
    if len(runs) < 2:
        print(
            f"fusion needs 2+ run files, found {sorted(runs)}. Nothing fused, "
            "nothing written."
        )
        return

    rrf_run = rrf(runs, k=args.rrf_k)
    rrf_path = args.runs_dir / f"{RRF_NAME}.jsonl"
    rrf_rows = write_run_jsonl(rrf_path, rrf_run)
    print(f"rrf k={args.rrf_k} over {sorted(runs)} -> {rrf_path} ({rrf_rows} rows)")

    lines = _report_header(runs, args.runs_dir, args.rrf_k, rrf_rows)

    qrels = load_qrels_dict(args.qrels)
    wsum_path = args.runs_dir / f"{WSUM_NAME}.jsonl"
    if not qrels:
        print(
            f"no judgments in {args.qrels}, so the weighted arm is skipped. RRF "
            "is done and needs nothing further. Weighted fusion tunes its "
            "weights on judged queries; there is no honest way to produce it "
            "before a human has judged. Judge with scripts/judge.py, then "
            "re-run."
        )
        if wsum_path.exists():
            print(
                f"warning: {wsum_path} exists from an earlier run and was not "
                "refreshed. Its weights predate the current judgments. Delete "
                "it before scoring if that matters."
            )
        lines += _skipped_weighted_section(args.qrels)
        _write_report(args.report, lines)
        return

    all_ids = sorted({query_id for run in runs.values() for query_id in run})
    usable = usable_query_ids(qrels, runs, all_ids)
    if len(usable) < 2:
        print(
            f"only {len(usable)} query is judged and retrieved by every arm; "
            "weighted fusion needs at least 2 so it can hold one out. Weighted "
            "arm skipped."
        )
        lines += _skipped_weighted_section(args.qrels)
        _write_report(args.report, lines)
        return

    train, test = split_queries(usable, args.train_frac, args.split_seed)
    weights = tune_weights(qrels, runs, train, args.metric, step=args.step)
    wsum_run = weighted(runs, weights)
    rows = write_run_jsonl(wsum_path, wsum_run)

    train_metric = score_run(qrels, wsum_run, train, args.metric)
    test_metric = held_out_score(qrels, wsum_run, train, test, args.metric)
    single_system_test = {
        name: held_out_score(qrels, run, train, test, args.metric)
        for name, run in runs.items()
    }
    single_system_test[RRF_NAME] = held_out_score(
        qrels, rrf_run, train, test, args.metric
    )

    lines += _weighted_section(
        qrels_path=args.qrels,
        n_judged=len(qrels),
        frac=args.train_frac,
        seed=args.split_seed,
        metric=args.metric,
        step=args.step,
        train=train,
        test=test,
        weights=weights,
        train_metric=train_metric,
        test_metric=test_metric,
        single_system_test=single_system_test,
        out_path=wsum_path,
        rows=rows,
    )
    _write_report(args.report, lines)

    weight_str = ", ".join(f"{name}={w:.2f}" for name, w in sorted(weights.items()))
    print(f"tuned weights on {len(train)} train queries: {weight_str}")
    print(f"wsum -> {wsum_path} ({rows} rows)")
    print(
        f"{args.metric}: held-out (test, n={len(test)}) {test_metric:.4f}  |  "
        f"tuning set (train, n={len(train)}, not a result) {train_metric:.4f}"
    )


def _write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
