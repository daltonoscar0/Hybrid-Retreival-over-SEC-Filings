#!/usr/bin/env python3
"""Score every retrieval run against a qrels file: nDCG@10, MRR@10,
Recall@100, bootstrap CIs, paired significance markers. One command
reproduces `reports/results.md` and its LaTeX twin.

Metric logic and the run-file contract retrieval systems must satisfy live
in `ticker.evaluation`; read that module's docstring first, especially if
you are the retrieval agent looking for what to write.

Runs before qrels or run files exist. With no judgments in --qrels or no
`*.jsonl` files under --runs-dir, this still writes a well-formed markdown
and LaTeX file explaining what is missing, so nothing downstream is blocked
on human judging finishing first.

Usage:
  uv run python scripts/evaluate.py
  uv run python scripts/evaluate.py --qrels data/qrels/novelty.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ticker.evaluation import (
    DEFAULT_MAX_P,
    DEFAULT_N_BOOT,
    DEFAULT_SEED,
    METRICS,
    build_eval_report,
    discover_runs,
    load_run_jsonl,
    render_latex,
    render_markdown,
)
from ticker.qrels import load_qrels_dict

DEFAULT_QRELS = Path("data/qrels/standard.jsonl")
DEFAULT_RUNS_DIR = Path("data/runs")


def _default_out_paths(qrels_path: Path) -> tuple[Path, Path]:
    stem = qrels_path.stem
    md = Path("reports/results.md") if stem == "standard" else Path(f"reports/results_{stem}.md")
    return md, md.with_suffix(".tex")


def _write_stub(out_md: Path, out_tex: Path, qrels_path: Path, reason: str) -> None:
    header = "| # | system | " + " | ".join(m.upper() for m in METRICS) + " |"
    sep = "|---|---|" + "---|" * len(METRICS)
    body = (
        f"# Retrieval evaluation -- `{qrels_path}`\n\n"
        f"*{reason}*\n\n"
        f"{header}\n{sep}\n"
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(body)
    out_tex.write_text(f"% {reason}\n")
    print(reason)
    print(f"wrote stub tables -> {out_md}, {out_tex}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--tex", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-p", type=float, default=DEFAULT_MAX_P)
    args = parser.parse_args()

    default_md, default_tex = _default_out_paths(args.qrels)
    out_md = args.out or default_md
    out_tex = args.tex or default_tex

    qrels_dict = load_qrels_dict(args.qrels)
    if not qrels_dict:
        _write_stub(
            out_md, out_tex, args.qrels,
            f"no judgments in {args.qrels} yet. Pool with scripts/pool.py, "
            "judge with scripts/judge.py, then re-run this command.",
        )
        return

    run_paths = discover_runs(args.runs_dir)
    if not run_paths:
        _write_stub(
            out_md, out_tex, args.qrels,
            f"no run files under {args.runs_dir}/*.jsonl yet. See "
            "ticker.evaluation's module docstring for the "
            "{query_id, chunk_id, score} JSONL contract a retriever writes "
            "to be scored here.",
        )
        return

    runs = {name: load_run_jsonl(path) for name, path in run_paths.items()}
    report = build_eval_report(
        args.qrels, qrels_dict, runs,
        n_boot=args.n_boot, seed=args.seed, max_p=args.max_p,
    )

    markdown = render_markdown(report)
    latex = render_latex(report)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown)
    out_tex.write_text(latex)

    print(markdown)
    print(f"wrote {out_md}, {out_tex}")


if __name__ == "__main__":
    main()
