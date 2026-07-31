#!/usr/bin/env python3
"""Run every wired retriever over the query set and write one run file each.

This is the producer for the contract `ticker.evaluation` documents: one
`data/runs/<system>.jsonl` per system, one line of
`{"query_id", "chunk_id", "score"}` per retrieved chunk. Read that module's
docstring before changing anything here. `scripts/evaluate.py` discovers
whatever this leaves in `--runs-dir` and scores it; `scripts/fuse.py` reads
the same files and writes fused ones alongside them.

Retriever wiring
-----------------
`--retriever` takes a `module.path:callable` spec and repeats, the same
convention `scripts/pool.py` uses for `--bm25` and `--dense`. The callable is
imported and invoked with no arguments and must return an object exposing
`.name` (str) and `.search(query: str, k: int) -> list[RankedChunk]`, i.e.
`ticker.retrieval_types.Retriever`. The run file is named from `.name`, not
from the spec, so the system's name in the results table is chosen by the
retriever module rather than by whatever the caller typed on the command
line.

One flag rather than pool.py's two named flags because the set of systems on
the ablation ladder grows and each new one should not need an argparse edit
here.

Why `--k` defaults to 100
--------------------------
Recall@100 is one of the three reported metrics. A run truncated at 10 scores
its Recall@100 against only 10 candidates and reports a number that looks
like a poor retriever rather than a truncated file. 100 is the smallest depth
at which every reported metric is computable.

Usage:
  uv run python scripts/search.py --retriever ticker.retrieval.bm25:load_retriever
  uv run python scripts/search.py \
      --retriever ticker.retrieval.bm25:load_retriever \
      --retriever ticker.retrieval.dense:load_retriever
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from pathlib import Path

from ticker.fusion import write_run_jsonl
from ticker.retrieval_types import Retriever

DEFAULT_QUERIES = Path("data/queries.jsonl")
DEFAULT_RUNS_DIR = Path("data/runs")
DEFAULT_K = 100


def _load_queries(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_retriever(spec: str) -> Retriever:
    module_name, sep, attr = spec.partition(":")
    if not sep:
        raise ValueError(f"retriever spec must be 'module.path:callable', got {spec!r}")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    retriever = factory()
    if not hasattr(retriever, "search"):
        raise TypeError(f"{spec!r} did not return an object with a .search method")
    return retriever


def run_retriever(retriever: Retriever, queries: list[dict], k: int):
    """Returns (run dict, per-query latencies in seconds).

    The run dict is `{query_id: {chunk_id: score}}`. Rank order is not stored
    and does not need to be: score order is the ranking, and
    `ticker.fusion.write_run_jsonl` sorts by score with chunk_id breaking
    ties, so a retriever whose ties come back in arbitrary order still
    produces a byte-identical file run to run.
    """
    run: dict[str, dict[str, float]] = {}
    latencies_s: list[float] = []
    for query in queries:
        start = time.perf_counter()
        results = retriever.search(query["text"], k)
        latencies_s.append(time.perf_counter() - start)
        run[query["query_id"]] = {rc.chunk_id: float(rc.score) for rc in results}
    return run, latencies_s


def _latency_line(latencies_s: list[float]) -> str:
    ms = sorted(x * 1000 for x in latencies_s)
    p95 = ms[max(0, int(0.95 * len(ms)) - 1)]
    return (
        f"latency ms/query: mean {statistics.mean(ms):.1f}  "
        f"median {statistics.median(ms):.1f}  p95 {p95:.1f}  max {max(ms):.1f}  "
        f"| total {sum(latencies_s):.1f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--retriever",
        action="append",
        default=None,
        metavar="module.path:callable",
        help="repeatable; one run file is written per retriever",
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args()

    specs = args.retriever or []
    if not specs:
        parser.error(
            "no --retriever given. Pass at least one module.path:callable, e.g. "
            "--retriever ticker.retrieval.bm25:load_retriever"
        )

    queries = _load_queries(args.queries)
    if not queries:
        print(f"no queries in {args.queries}, nothing to run")
        return

    if args.k < 100:
        print(
            f"warning: --k {args.k} is below the Recall@100 cutoff, so any "
            "Recall@100 this run reports will be an artifact of the truncation."
        )

    written: list[str] = []
    for spec in specs:
        retriever = _load_retriever(spec)
        run, latencies_s = run_retriever(retriever, queries, args.k)
        out_path = args.runs_dir / f"{retriever.name}.jsonl"
        rows = write_run_jsonl(out_path, run)
        empty = [qid for qid, scores in run.items() if not scores]
        print(f"{retriever.name}  <- {spec}")
        print(f"  top-{args.k} for {len(queries)} queries -> {out_path} ({rows} rows)")
        print(f"  {_latency_line(latencies_s)}")
        if empty:
            print(f"  {len(empty)} queries returned nothing: {empty[:5]}")
        written.append(str(out_path))

    print(f"\nwrote {len(written)} run file(s): {', '.join(written)}")
    print(
        "no metric is computed here. Score these with scripts/evaluate.py once "
        "there are judgments to score against."
    )


if __name__ == "__main__":
    main()
