#!/usr/bin/env python3
"""Report intra-annotator agreement between a judging pass and its rejudge.

Cohen's kappa logic lives in `ticker.agreement`; this script loads the two
files and prints the report. See that module's docstring for why a second
pass and why skips count as their own category.

Usage:
  uv run python scripts/judge.py --rejudge 30       # produce the second pass
  uv run python scripts/agreement.py                 # report kappa on it
  uv run python scripts/agreement.py --qrels data/qrels/novelty.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ticker.agreement import compare_passes
from ticker.judging import rejudge_path

DEFAULT_QRELS = Path("data/qrels/standard.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS, help="the first-pass qrels file")
    parser.add_argument(
        "--rejudge", type=Path, default=None,
        help="the second-pass file; defaults to the qrels file's *.rejudge.jsonl sibling",
    )
    args = parser.parse_args()

    pass2_path = args.rejudge or rejudge_path(args.qrels)

    if not args.qrels.exists() or args.qrels.stat().st_size == 0:
        print(f"{args.qrels} has no judgments yet. Judge first, then rejudge, then run this.")
        return
    if not pass2_path.exists() or pass2_path.stat().st_size == 0:
        print(
            f"{pass2_path} does not exist yet. Produce a second pass with:\n"
            f"  uv run python scripts/judge.py --qrels {args.qrels} --rejudge 30"
        )
        return

    report = compare_passes(args.qrels, pass2_path)

    print(f"pass 1: {args.qrels}")
    print(f"pass 2: {pass2_path}")
    print(f"paired judgments: {report.n_pairs}")
    if report.pass2_missing_from_pass1:
        print(
            f"warning: {report.pass2_missing_from_pass1} pass-2 pairs have no "
            "matching pass-1 judgment (excluded)"
        )
    print(f"observed agreement (po): {report.observed_agreement:.3f}")
    print(f"expected agreement (pe): {report.expected_agreement:.3f}")
    print(f"Cohen's kappa: {report.kappa:.3f}")

    print("\nconfusion (pass1 -> pass2 : count)")
    for (a, b), count in sorted(report.confusion.items(), key=lambda kv: str(kv[0])):
        marker = "" if a == b else "  <- disagreement"
        print(f"  {a} -> {b} : {count}{marker}")


if __name__ == "__main__":
    main()
