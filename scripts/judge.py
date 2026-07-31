#!/usr/bin/env python3
"""Judge pooled chunks one at a time. Every session logic decision lives in
`ticker.judging`; this script owns the terminal: single-keystroke input, no
Enter required, because a slow tool turns two days of judging into two
weeks.

Grading a chunk is one keypress: 0/1/2/3 grades and advances, s skips and
advances, q quits. Every keypress writes its line to the output file before
the next chunk is shown, so killing the process loses at most the item on
screen, and re-running picks up exactly where you left off.

Standard pass:
  uv run python scripts/judge.py
  -> reads data/queries.jsonl + data/pool/pool.jsonl
  -> appends to data/qrels/standard.jsonl

Novelty-conditioned pass, same tool, different instruction, separate file,
never merged with the standard file:
  uv run python scripts/judge.py --instruction novelty
  -> appends to data/qrels/novelty.jsonl

Second pass for intra-annotator agreement (see scripts/agreement.py):
  uv run python scripts/judge.py --rejudge 30
  -> samples 30 already-graded pairs from data/qrels/standard.jsonl and
     appends the second-pass grades to data/qrels/standard.rejudge.jsonl,
     blind to the first-pass grade (this script never shows it to you)

Restrict to one query while judging (useful for finishing a query you
started, or for the 25-query novelty subset per PLAN Phase 6):
  uv run python scripts/judge.py --query q07 --query q12
"""

from __future__ import annotations

import argparse
import json
import sys
import termios
import tty
import uuid
from pathlib import Path

from ticker import db
from ticker.judging import (
    INSTRUCTIONS,
    rejudge_path,
    run_judging_session,
    sample_for_rejudge,
)

DEFAULT_QUERIES = Path("data/queries.jsonl")
DEFAULT_DB_PATH = Path("data/ticker.duckdb")
DEFAULT_POOL = Path("data/pool/pool.jsonl")
OUT_BY_INSTRUCTION = {
    "standard": Path("data/qrels/standard.jsonl"),
    "novelty": Path("data/qrels/novelty.jsonl"),
}


def _read_key() -> str:
    """One character, no Enter, restoring terminal settings on exit."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if ch in ("\x03", "\x04"):  # ctrl-c / ctrl-d
        return "q"
    return ch.lower()


def _load_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            queries[row["query_id"]] = row["text"]
    return queries


def _load_pool(path: Path) -> dict[str, list[str]]:
    pool: dict[str, list[str]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pool[row["query_id"]] = row["chunk_ids"]
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--instruction", choices=sorted(INSTRUCTIONS), default="standard")
    parser.add_argument("--out", type=Path, default=None, help="override the qrels output path")
    parser.add_argument(
        "--query", action="append", default=None, dest="query_ids",
        help="restrict this session to one query_id (repeatable)",
    )
    parser.add_argument(
        "--rejudge", type=int, default=None, metavar="N",
        help="second-pass mode: sample N graded pairs from the target qrels file "
        "and re-judge them into a sibling *.rejudge.jsonl file",
    )
    parser.add_argument("--rejudge-seed", type=int, default=None)
    parser.add_argument(
        "--redo", action="store_true",
        help="re-present pairs already judged in the output file instead of skipping them",
    )
    parser.add_argument("--session-id", default=None)
    args = parser.parse_args()

    out_path = args.out or OUT_BY_INSTRUCTION[args.instruction]
    session_id = args.session_id or uuid.uuid4().hex[:12]
    query_texts = _load_queries(args.queries)

    con = db.connect(args.db)
    try:
        if args.rejudge is not None:
            sample = sample_for_rejudge(out_path, args.rejudge, seed=args.rejudge_seed)
            if not sample:
                print(f"nothing graded yet in {out_path}, nothing to rejudge")
                return
            pool: dict[str, list[str]] = {}
            for query_id, chunk_id in sample:
                pool.setdefault(query_id, []).append(chunk_id)
            queries = [(qid, query_texts[qid]) for qid in pool if qid in query_texts]
            rejudge_out = rejudge_path(out_path)
            print(
                f"rejudge pass: {len(sample)} pairs across {len(queries)} queries "
                f"-> {rejudge_out}"
            )
            out_path = rejudge_out
        else:
            pool = _load_pool(args.pool)
            wanted = set(args.query_ids) if args.query_ids else None
            queries = [
                (qid, text)
                for qid, text in query_texts.items()
                if qid in pool and (wanted is None or qid in wanted)
            ]
            if not queries:
                print("no queries to judge (check --queries, --pool, --query filters)")
                return

        print(f"instruction: {INSTRUCTIONS[args.instruction]}")
        print(f"writing to {out_path}   session {session_id}")
        print("keys: 0-3 grade, s skip, q quit -- no Enter needed\n")

        summary = run_judging_session(
            con,
            queries,
            pool,
            instruction=args.instruction,
            out_path=out_path,
            session_id=session_id,
            next_key=_read_key,
            redo=args.redo,
        )
    finally:
        con.close()

    print("\n" + "-" * 40)
    print(f"judged this session: {summary.judged_this_session}")
    if summary.grade_counts:
        counts = ", ".join(f"{k}:{v}" for k, v in sorted(summary.grade_counts.items()))
        print(f"grade counts this session: {counts}")
    done = summary.already_judged_pairs + summary.judged_this_session
    print(f"total judged in pool: {done}/{summary.total_pool_pairs}")
    if summary.quit_early:
        print("stopped early -- rerun to resume where you left off")


if __name__ == "__main__":
    main()
