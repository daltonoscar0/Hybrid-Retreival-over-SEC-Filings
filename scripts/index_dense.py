#!/usr/bin/env python3
"""Build the dense index over `chunks` with `bge-base-en-v1.5` and FAISS flat.

The dense twin of `scripts/index.py`. Same corpus read (`ORDER BY chunk_id`, so
build order does not depend on DuckDB's physical row order), same
`chunk_ids.json` sidecar mapping index position back to a database `chunk_id`,
same `manifest.json` carrying the full input surface, same `--verify-repeat`
flag. Writes:

  data/index/dense/
    index.faiss      -- IndexFlatIP over L2-normalized vectors, so cosine
    chunk_ids.json   -- position -> chunk_id
    manifest.json    -- model name and resolved revision sha, prefix policy,
                        dim, pooling, dtype, device, library versions, corpus
                        fingerprint, seed, timings

Why the manifest carries the revision sha. BM25 has no pretrained weights, so
`scripts/index.py`'s manifest pins only the corpus and the library version.
Dense retrieval has a second input the corpus fingerprint cannot see: the model
checkpoint. `resolve_revision` reads the commit sha off the resolved snapshot
directory rather than writing "main", and `load_retriever` pins the query-time
model to that sha, so a moved upstream branch cannot silently change what the
saved index means.

Truncation is reported, not hidden. `bge-base-en-v1.5` has a 512-token limit and
some chunks run past it. The tail of a truncated chunk contributes nothing to
its embedding, and a dense arm that quietly drops text would make the BM25
comparison unfair in a direction nobody could see from the results table. The
build tokenizes the corpus once up front and prints how many chunks exceed the
limit; the count goes in the manifest.

`--verify-repeat` re-encodes into a scratch directory and both byte-diffs the
index file and reports the max absolute difference between the two embedding
matrices. The byte comparison is the real check. The max-abs-diff is there
because a float pipeline that stops being bit-identical after a library upgrade
should show the size of the disagreement rather than only the word DIFFERS.

Usage:
  uv run python scripts/index_dense.py --limit 2000
  uv run python scripts/index_dense.py --limit 2000 --verify-repeat
  uv run python scripts/index_dense.py
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import platform
import tempfile
import time
from pathlib import Path

import duckdb
import numpy as np

from ticker.retrieval.dense import (
    CHUNK_IDS_FILENAME,
    DEFAULT_BATCH_SIZE,
    DEFAULT_INDEX_DIR,
    DTYPE,
    EMBEDDING_DIM,
    FAISS_FILENAME,
    MANIFEST_FILENAME,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
    NORMALIZE,
    PASSAGE_INSTRUCTION,
    POOLING_MODE,
    QUERY_INSTRUCTION,
    SEED,
    DenseEncoder,
    load_model,
    pick_device,
    resolve_revision,
    seed_everything,
)

DEFAULT_DB = Path("data/ticker.duckdb")

INDEX_ARTIFACT_FILES = (FAISS_FILENAME, CHUNK_IDS_FILENAME)

# manifest.json fields that must match across a rebuild. timing_s is excluded
# on purpose: wall-clock time is not a reproducibility property.
MANIFEST_STABLE_KEYS = (
    "model_name", "model_revision", "embedding_dim", "normalize", "pooling_mode",
    "max_seq_length", "query_instruction", "passage_instruction", "batch_size",
    "dtype", "device", "index_type", "metric", "n_chunks", "n_chunks_truncated",
    "corpus_sha256", "seed", "limit",
)


def _fetch_chunks(db_path: Path, limit: int | None) -> tuple[list[str], list[str]]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        sql = "SELECT chunk_id, text FROM chunks ORDER BY chunk_id"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = con.execute(sql).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows], [r[1] for r in rows]


def _corpus_fingerprint(chunk_ids: list[str], texts: list[str]) -> str:
    """SHA256 over (chunk_id, text) pairs in build order, byte-for-byte the
    same construction `scripts/index.py` uses, so a BM25 index and a dense
    index built from the same corpus slice carry the same fingerprint and can
    be checked against each other before their runs are fused."""
    digest = hashlib.sha256()
    for chunk_id, text in zip(chunk_ids, texts):
        digest.update(chunk_id.encode())
        digest.update(b"\0")
        digest.update(text.encode())
        digest.update(b"\x1e")
    return digest.hexdigest()


def _count_truncated(model, texts: list[str]) -> int:
    """How many chunks tokenize past the model's input limit.

    Uses the fast tokenizer with truncation off so the true length is visible.
    One extra pass over the corpus, a few seconds on 146k chunks, and it buys
    an honest number for the report instead of a silent quality loss.
    """
    tokenizer = model.tokenizer
    limit = model.max_seq_length
    truncated = 0
    for start in range(0, len(texts), 512):
        batch = texts[start : start + 512]
        encoded = tokenizer(batch, add_special_tokens=True, truncation=False)["input_ids"]
        truncated += sum(1 for ids in encoded if len(ids) > limit)
    return truncated


def build_index(
    db_path: Path,
    out_dir: Path,
    *,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    model=None,
    show_progress: bool = True,
) -> dict:
    """Encode every chunk and write the FAISS index, sidecar and manifest.

    `model` is injectable so a caller that already paid the load cost (the
    `--verify-repeat` second pass) does not pay it twice. Left `None` it loads
    `MODEL_NAME` pinned to the resolved snapshot sha.
    """
    # faiss is imported after the encode, not here. faiss-cpu and torch each
    # ship their own OpenMP runtime, and on macOS having both live in one
    # process segfaults inside `__kmp_fork_barrier` partway through a long
    # encode: reproducibly at batch 0 of 2,315 on this corpus, on both mps and
    # cpu, with KMP_DUPLICATE_LIB_OK=TRUE set. It does not reproduce on the
    # few-hundred-vector builds in tests/test_dense.py, which is why the crash
    # only showed up on the full corpus. Loading faiss after the encode leaves
    # torch's runtime as the only one live during the thousands of OpenMP
    # fork/join cycles that trip it. faiss then does a flat copy and a file
    # write, neither of which needs threads at all.
    import sentence_transformers
    import torch

    seed_everything(SEED)
    device = device or pick_device()
    revision = resolve_revision(MODEL_NAME)

    t0 = time.monotonic()
    chunk_ids, texts = _fetch_chunks(db_path, limit)
    if not chunk_ids:
        raise RuntimeError(f"no chunks found in {db_path}; nothing to index")
    fetch_s = time.monotonic() - t0

    t0 = time.monotonic()
    if model is None:
        model = load_model(MODEL_NAME, revision=revision, device=device)
    load_s = time.monotonic() - t0

    t0 = time.monotonic()
    n_truncated = _count_truncated(model, texts)
    truncation_scan_s = time.monotonic() - t0

    encoder = DenseEncoder(model, batch_size=batch_size)
    t0 = time.monotonic()
    with torch.no_grad():
        vectors = encoder.encode_passages(texts, show_progress=show_progress)
    encode_s = time.monotonic() - t0

    if vectors.shape != (len(chunk_ids), EMBEDDING_DIM):
        raise RuntimeError(
            f"encoder returned {vectors.shape}, expected "
            f"({len(chunk_ids)}, {EMBEDDING_DIM}); model or dim constant is wrong"
        )

    import faiss

    faiss.omp_set_num_threads(1)

    t0 = time.monotonic()
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)
    index_s = time.monotonic() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    faiss.write_index(index, str(out_dir / FAISS_FILENAME))
    (out_dir / CHUNK_IDS_FILENAME).write_text(json.dumps(chunk_ids))
    save_s = time.monotonic() - t0

    manifest = {
        "model_name": MODEL_NAME,
        "model_revision": revision,
        "embedding_dim": EMBEDDING_DIM,
        "normalize": NORMALIZE,
        "pooling_mode": POOLING_MODE,
        "max_seq_length": MAX_SEQ_LENGTH,
        "query_instruction": QUERY_INSTRUCTION,
        "passage_instruction": PASSAGE_INSTRUCTION,
        "uses_query_instruction": bool(QUERY_INSTRUCTION),
        "batch_size": batch_size,
        "dtype": DTYPE,
        "device": device,
        "index_type": "IndexFlatIP",
        "metric": "cosine (inner product over L2-normalized vectors)",
        "sentence_transformers_version": sentence_transformers.__version__,
        "torch_version": torch.__version__,
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "n_chunks": len(chunk_ids),
        "n_chunks_truncated": n_truncated,
        "corpus_sha256": _corpus_fingerprint(chunk_ids, texts),
        "seed": SEED,
        "limit": limit,
        "db_path": str(db_path),
        "timing_s": {
            "fetch": fetch_s,
            "load_model": load_s,
            "truncation_scan": truncation_scan_s,
            "encode": encode_s,
            "index": index_s,
            "save": save_s,
            "total": fetch_s + load_s + truncation_scan_s + encode_s + index_s + save_s,
        },
        "chunks_per_second": len(chunk_ids) / encode_s if encode_s > 0 else None,
    }
    (out_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.glob("*") if f.is_file())


def _read_vectors(index_path: Path) -> np.ndarray:
    import faiss

    index = faiss.read_index(str(index_path))
    return index.reconstruct_n(0, index.ntotal)


def _verify_repeat(db_path: Path, out_dir: Path, **build_kwargs) -> bool:
    """Re-encode into a scratch directory, byte-diff, and report drift size."""
    with tempfile.TemporaryDirectory() as scratch:
        scratch_dir = Path(scratch) / "dense_repeat"
        build_index(db_path, scratch_dir, show_progress=False, **build_kwargs)

        ok = True
        for name in INDEX_ARTIFACT_FILES:
            a, b = out_dir / name, scratch_dir / name
            same = a.exists() and b.exists() and filecmp.cmp(a, b, shallow=False)
            print(f"  {name}: {'byte-identical' if same else 'DIFFERS'}")
            ok = ok and same

        va = _read_vectors(out_dir / FAISS_FILENAME)
        vb = _read_vectors(scratch_dir / FAISS_FILENAME)
        if va.shape == vb.shape:
            print(f"  max abs embedding difference: {np.abs(va - vb).max():.3e}")
        else:
            print(f"  embedding shape mismatch: {va.shape} vs {vb.shape}")
            ok = False

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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="index only the first N chunks by chunk_id; recorded in the manifest "
        "so a subset index is never mistaken for a full one",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default=None, help="mps, cpu; default is mps when available")
    parser.add_argument(
        "--verify-repeat",
        action="store_true",
        help="re-encode into a scratch dir, byte-diff the index and report embedding drift",
    )
    args = parser.parse_args()

    manifest = build_index(
        args.db,
        args.out,
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device,
    )
    size_bytes = _dir_size_bytes(args.out)
    print(
        f"indexed {manifest['n_chunks']} chunks, dim {manifest['embedding_dim']}, "
        f"{manifest['index_type']} on {manifest['device']} -> {args.out}"
    )
    print(
        f"model {manifest['model_name']} @ {manifest['model_revision']}  "
        f"query instruction: {'on' if manifest['uses_query_instruction'] else 'off'}"
    )
    print(
        f"chunks over the {manifest['max_seq_length']}-token limit: "
        f"{manifest['n_chunks_truncated']} "
        f"({100 * manifest['n_chunks_truncated'] / manifest['n_chunks']:.1f}%)"
    )
    t = manifest["timing_s"]
    print(
        f"timing: fetch {t['fetch']:.2f}s  load {t['load_model']:.2f}s  "
        f"truncation scan {t['truncation_scan']:.2f}s  encode {t['encode']:.2f}s  "
        f"index {t['index']:.2f}s  save {t['save']:.2f}s  total {t['total']:.2f}s"
    )
    print(f"encode throughput: {manifest['chunks_per_second']:.1f} chunks/sec")
    print(f"index size on disk: {size_bytes / 1e6:.2f} MB")

    if args.verify_repeat:
        print("re-encoding into a scratch directory to verify a repeatable build...")
        ok = _verify_repeat(
            args.db,
            args.out,
            limit=args.limit,
            batch_size=args.batch_size,
            device=args.device,
        )
        print("VERIFIED repeatable build" if ok else "REBUILD MISMATCH -- see above")


if __name__ == "__main__":
    main()
