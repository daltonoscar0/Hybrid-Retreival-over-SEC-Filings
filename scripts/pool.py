#!/usr/bin/env python3
"""Build judging pools for every query in a query set.

Pooling logic and the TREC-pooling justification live in `ticker.pooling`;
this script is the CLI shell: load queries, optionally wire retrievers,
write `data/pool/pool.jsonl`.

Retriever wiring
-----------------
`--bm25` and `--dense` each take a `module.path:callable` spec. The callable
is imported and invoked with no arguments; it must return an object exposing
`.name` (str) and `.search(query: str, k: int) -> list[RankedChunk]`, i.e.
`ticker.retrieval_types.Retriever`. Neither module exists yet -- BM25 and
dense retrieval are a different agent's deliverable -- so both flags are
optional. Pooling with neither wired still runs, using the keyword seed
source alone, and says so loudly: a pool missing two of its three sources is
not a pool a judge should trust yet, and re-running this script once real
retrievers exist is expected and cheap (it fully overwrites the pool file).

Usage:
  uv run python scripts/pool.py
  uv run python scripts/pool.py \
      --bm25 ticker.retrieval.bm25:load_retriever \
      --dense ticker.retrieval.dense:load_retriever
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from ticker import db
from ticker.pooling import DEFAULT_K, DEFAULT_KEYWORD_K, pool_queries
from ticker.retrieval_types import Retriever

DEFAULT_QUERIES = Path("data/queries.jsonl")
DEFAULT_DB_PATH = Path("data/ticker.duckdb")
DEFAULT_OUT = Path("data/pool/pool.jsonl")


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--keyword-k", type=int, default=DEFAULT_KEYWORD_K)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bm25", default=None, help="module.path:callable returning a Retriever")
    parser.add_argument("--dense", default=None, help="module.path:callable returning a Retriever")
    args = parser.parse_args()

    queries = _load_queries(args.queries)
    if not queries:
        print(f"no queries in {args.queries}, nothing to pool")
        return

    retrievers = []
    for flag_name, spec in (("--bm25", args.bm25), ("--dense", args.dense)):
        if spec is None:
            continue
        retrievers.append(_load_retriever(spec))
        print(f"wired {flag_name} -> {spec}")

    if not retrievers:
        print(
            "no --bm25 or --dense retriever wired; pooling from the keyword seed "
            "only. This pool is incomplete -- re-run once BM25 and dense "
            "retrieval exist, before anyone judges it."
        )

    con = db.connect(args.db)
    try:
        pooled = pool_queries(
            con,
            [(q["query_id"], q["text"]) for q in queries],
            retrievers,
            k=args.k,
            keyword_k=args.keyword_k,
            seed=args.seed,
        )
    finally:
        con.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for pooled_query in pooled:
            f.write(
                json.dumps(
                    {
                        "query_id": pooled_query.query_id,
                        "chunk_ids": list(pooled_query.chunk_ids),
                        "sources": pooled_query.sources,
                        "pool_size": len(pooled_query.chunk_ids),
                    }
                )
                + "\n"
            )

    sizes = [len(pq.chunk_ids) for pq in pooled]
    print(f"pooled {len(pooled)} queries -> {args.out}")
    if sizes:
        print(
            f"pool size: min {min(sizes)}  max {max(sizes)}  "
            f"mean {sum(sizes) / len(sizes):.1f}  total judgments if all pools "
            f"are graded: {sum(sizes)}"
        )


if __name__ == "__main__":
    main()
