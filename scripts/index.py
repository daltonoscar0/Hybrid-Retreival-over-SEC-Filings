#!/usr/bin/env python3
"""Build the BM25 index over `chunks`.

Reads every chunk from the DuckDB corpus, ordered by `chunk_id` so build
order -- and therefore the on-disk index -- does not depend on DuckDB's
physical row order or any other incidental state. Tokenizes with
`ticker.retrieval.bm25.tokenize_texts` (the same function `BM25Retriever`
uses at query time), builds a `bm25s.BM25` index using the `lucene` scoring
variant (see `ticker.retrieval.bm25`'s module docstring for which of the five
Kamphuis et al. variants and why), and writes:

  data/index/bm25/
    data.csc.index.npy, indices.csc.index.npy, indptr.csc.index.npy   -- scores
    vocab.index.json, params.index.json                                -- bm25s state
    chunk_ids.json           -- position -> chunk_id, this repo's own sidecar
    manifest.json             -- build parameters, library versions, corpus
                                  fingerprint, timings (see below)

Byte-identical rebuilds. BM25 indexing here has no stochastic step: token
order is insertion order over a fixed-order corpus, IDF/TF are closed-form
arithmetic, and there is no threading in `bm25s`'s index-build path (only
`retrieve` takes an `n_threads` argument). `--verify-repeat` rebuilds a
second time into a sibling directory and byte-diffs every index artifact
(everything except `manifest.json`, whose `timing_s` field is wall-clock and
expected to differ run to run; every other manifest field is still checked).

Usage:
  uv run python scripts/index.py
  uv run python scripts/index.py --verify-repeat
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import platform
import shutil
import tempfile
import time
from pathlib import Path

import bm25s
import duckdb
import numpy as np

from ticker.retrieval.bm25 import (
    BM25_VARIANT,
    CHUNK_IDS_FILENAME,
    DEFAULT_B,
    DEFAULT_DELTA,
    DEFAULT_INDEX_DIR,
    DEFAULT_K1,
    MANIFEST_FILENAME,
    STOPWORDS,
    TOKEN_PATTERN,
    tokenize_texts,
)

DEFAULT_DB = Path("data/ticker.duckdb")

# BM25 indexing is deterministic end to end (see module docstring); nothing
# here actually consumes randomness. Seeded anyway, defensively, so this
# script follows the repo-wide "seed everything" convention and does not
# silently stop being reproducible if a future bm25s version adds one.
SEED = 0

INDEX_ARTIFACT_FILES = (
    "data.csc.index.npy",
    "indices.csc.index.npy",
    "indptr.csc.index.npy",
    "vocab.index.json",
    "params.index.json",
    CHUNK_IDS_FILENAME,
)

# manifest.json fields that must match across a rebuild; timing_s is
# excluded on purpose -- wall-clock time is not a reproducibility property.
MANIFEST_STABLE_KEYS = (
    "method", "k1", "b", "delta", "tokenizer", "bm25s_version",
    "n_chunks", "vocab_size", "corpus_sha256", "seed",
)


def _fetch_chunks(db_path: Path) -> tuple[list[str], list[str]]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            "SELECT chunk_id, text FROM chunks ORDER BY chunk_id"
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows], [r[1] for r in rows]


def _corpus_fingerprint(chunk_ids: list[str], texts: list[str]) -> str:
    """SHA256 over (chunk_id, text) pairs in build order. Stands in for a
    model revision hash -- BM25 has no pretrained weights to pin, so the
    corpus content plus library version plus hyperparameters is the entire
    input surface, and this hash lets a later run confirm the index still
    matches the corpus it claims to be built from."""
    digest = hashlib.sha256()
    for chunk_id, text in zip(chunk_ids, texts):
        digest.update(chunk_id.encode())
        digest.update(b"\0")
        digest.update(text.encode())
        digest.update(b"\x1e")
    return digest.hexdigest()


def build_index(
    db_path: Path,
    out_dir: Path,
    *,
    method: str = BM25_VARIANT,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
    delta: float = DEFAULT_DELTA,
    show_progress: bool = True,
) -> dict:
    np.random.seed(SEED)

    t0 = time.monotonic()
    chunk_ids, texts = _fetch_chunks(db_path)
    if not chunk_ids:
        raise RuntimeError(f"no chunks found in {db_path}; nothing to index")
    fetch_s = time.monotonic() - t0

    t0 = time.monotonic()
    corpus_tokens = tokenize_texts(texts, show_progress=show_progress)
    tokenize_s = time.monotonic() - t0

    t0 = time.monotonic()
    index = bm25s.BM25(k1=k1, b=b, delta=delta, method=method)
    index.index(corpus_tokens, show_progress=show_progress)
    index_s = time.monotonic() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    index.save(str(out_dir), corpus=None)
    (out_dir / CHUNK_IDS_FILENAME).write_text(json.dumps(chunk_ids))
    save_s = time.monotonic() - t0

    manifest = {
        "method": method,
        "k1": k1,
        "b": b,
        "delta": delta,
        "tokenizer": {
            "token_pattern": TOKEN_PATTERN,
            "stopwords": STOPWORDS,
            "stemmer": None,
            "lower": True,
        },
        "bm25s_version": bm25s.__version__,
        "python_version": platform.python_version(),
        "n_chunks": len(chunk_ids),
        "vocab_size": len(index.vocab_dict),
        "corpus_sha256": _corpus_fingerprint(chunk_ids, texts),
        "seed": SEED,
        "db_path": str(db_path),
        "timing_s": {
            "fetch": fetch_s,
            "tokenize": tokenize_s,
            "index": index_s,
            "save": save_s,
            "total": fetch_s + tokenize_s + index_s + save_s,
        },
    }
    (out_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.glob("*") if f.is_file())


def _verify_repeat(db_path: Path, out_dir: Path, **build_kwargs) -> bool:
    """Rebuild into a scratch directory and byte-diff against `out_dir`."""
    with tempfile.TemporaryDirectory() as scratch:
        scratch_dir = Path(scratch) / "bm25_repeat"
        build_index(db_path, scratch_dir, show_progress=False, **build_kwargs)

        ok = True
        for name in INDEX_ARTIFACT_FILES:
            a, b_path = out_dir / name, scratch_dir / name
            same = a.exists() and b_path.exists() and filecmp.cmp(a, b_path, shallow=False)
            print(f"  {name}: {'byte-identical' if same else 'DIFFERS'}")
            ok = ok and same

        manifest_a = json.loads((out_dir / MANIFEST_FILENAME).read_text())
        manifest_b = json.loads((scratch_dir / MANIFEST_FILENAME).read_text())
        for key in MANIFEST_STABLE_KEYS:
            same = manifest_a.get(key) == manifest_b.get(key)
            print(f"  manifest.{key}: {'matches' if same else 'DIFFERS'}")
            ok = ok and same
        return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--method", default=BM25_VARIANT)
    parser.add_argument("--k1", type=float, default=DEFAULT_K1)
    parser.add_argument("--b", type=float, default=DEFAULT_B)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument(
        "--verify-repeat",
        action="store_true",
        help="rebuild a second time into a scratch dir and byte-diff every index artifact",
    )
    args = parser.parse_args()

    manifest = build_index(
        args.db, args.out, method=args.method, k1=args.k1, b=args.b, delta=args.delta
    )
    size_bytes = _dir_size_bytes(args.out)
    print(
        f"indexed {manifest['n_chunks']} chunks, vocab {manifest['vocab_size']} "
        f"tokens, variant={manifest['method']} -> {args.out}"
    )
    print(
        f"timing: fetch {manifest['timing_s']['fetch']:.2f}s  "
        f"tokenize {manifest['timing_s']['tokenize']:.2f}s  "
        f"index {manifest['timing_s']['index']:.2f}s  "
        f"save {manifest['timing_s']['save']:.2f}s  "
        f"total {manifest['timing_s']['total']:.2f}s"
    )
    print(f"index size on disk: {size_bytes / 1e6:.2f} MB")

    if args.verify_repeat:
        print("rebuilding into a scratch directory to verify byte-identical output...")
        ok = _verify_repeat(
            args.db, args.out, method=args.method, k1=args.k1, b=args.b, delta=args.delta
        )
        print("VERIFIED byte-identical rebuild" if ok else "REBUILD MISMATCH -- see above")


if __name__ == "__main__":
    main()
