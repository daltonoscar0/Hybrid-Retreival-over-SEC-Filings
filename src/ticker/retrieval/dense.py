"""Dense retriever over `chunks`: `BAAI/bge-base-en-v1.5` plus a FAISS flat index.

Model. RUN.md locks the embedding arm to `bge-base-en-v1.5` and cuts the
finance-adapted second arm, so there is no model selection to make here and
`MODEL_NAME` is a constant rather than a parameter. What is still a choice is
which revision: a bare model name resolves to whatever `main` points at on the
day the index was built, and a later rebuild against a moved `main` would
produce embeddings that silently disagree with a saved index. `scripts/index_dense.py`
resolves the actual commit sha of the downloaded snapshot and writes it to the
manifest; `load_retriever` pins the query-time model to that same sha. The index
and the retriever therefore cannot drift apart even if the upstream repo moves.

Index. FAISS `IndexFlatIP` over L2-normalized vectors. Inner product between
unit vectors is cosine similarity, so this is exact cosine search with no
approximation and no training step. PLAN section 3 says flat is fine at this
scale and invariant 6 forbids a vector database; at 146k chunks x 768 dims a
flat index is 450 MB in RAM and searches in single-digit milliseconds, which is
faster than the encoder call that produced the query vector. IVF or HNSW would
add a recall/latency knob that nothing in this project needs and would make the
dense arm's numbers a function of index hyperparameters rather than of the
embedding model, which is the thing being measured.

The query instruction prefix, and why it is on
----------------------------------------------
The bge-v1.5 model card prescribes prepending "Represent this sentence for
searching relevant passages: " to queries only, never to passages, and notes
that for v1.5 the instruction helps retrieval but is not mandatory. This module
turns it on (`QUERY_INSTRUCTION` below). The case here is the asymmetric one the
instruction was tuned for: queries are short topic phrases ("customer
concentration risk", six words at most across the 40-query set) and passages are
four-sentence windows of filing prose averaging 1.1k characters. That is exactly
the short-query-to-long-passage setting the model card recommends it for. The
symmetric case where the card says to drop it, sentence-to-sentence similarity,
does not occur anywhere in this project.

The failure mode this creates is a policy split between build time and query
time. Prefixing at both ends, or at neither, is coherent; prefixing at only one
end by accident degrades retrieval quietly and produces no error. Two things
prevent that. First, `DenseEncoder` is the single place a string is formatted
before it reaches the model, and it exposes `encode_passages` and
`encode_queries` as separate methods, so a caller cannot pick the wrong policy
without picking the wrong method name. Second, the instruction string is written
into the index manifest at build time and read back out of the manifest by
`load_retriever`, so the retriever applies the policy the index was built under
rather than whatever this module's constant happens to say today.

Determinism
-----------
`scripts/index_dense.py` seeds python, numpy and torch, calls `model.eval()` so
dropout is off, and encodes under `torch.no_grad`. A BERT forward pass has no
stochastic step once dropout is disabled, so embeddings repeat bit-for-bit on a
fixed device. They are not bit-identical *across* devices: MPS and CPU differ in
the last few ulps of the accumulated matmuls, which is why the manifest records
`device` and why `--verify-repeat` reports a max absolute embedding difference
alongside the byte comparison rather than only a pass/fail.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from ticker.retrieval_types import RankedChunk

MODEL_NAME = "BAAI/bge-base-en-v1.5"

# See "The query instruction prefix" above. Applied to queries, never to
# passages. The trailing space is part of the model card's string.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
PASSAGE_INSTRUCTION = ""

EMBEDDING_DIM = 768
MAX_SEQ_LENGTH = 512
POOLING_MODE = "cls"
NORMALIZE = True
DTYPE = "float32"

DEFAULT_BATCH_SIZE = 64
SEED = 0

DEFAULT_INDEX_DIR = Path("data/index/dense")
FAISS_FILENAME = "index.faiss"
CHUNK_IDS_FILENAME = "chunk_ids.json"
MANIFEST_FILENAME = "manifest.json"


class EncoderBackend(Protocol):
    """The slice of `sentence_transformers.SentenceTransformer` this module
    uses. Narrow on purpose: tests inject a recording stub in its place and
    exercise the prefix policy and the FAISS round trip with no model
    download."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> np.ndarray: ...


class DenseEncoder:
    """The only place text is formatted before it reaches the model.

    Passages and queries get different treatment (see the module docstring),
    so they get different methods. A caller that wants the passage policy has
    to name `encode_passages`; there is no shared entry point taking a flag
    that could be defaulted wrongly.
    """

    def __init__(
        self,
        backend: EncoderBackend,
        *,
        query_instruction: str = QUERY_INSTRUCTION,
        passage_instruction: str = PASSAGE_INSTRUCTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.backend = backend
        self.query_instruction = query_instruction
        self.passage_instruction = passage_instruction
        self.batch_size = batch_size

    def _encode(self, texts: list[str], prefix: str, show_progress: bool) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        prepared = [prefix + text for text in texts] if prefix else list(texts)
        vectors = self.backend.encode(
            prepared,
            batch_size=self.batch_size,
            normalize_embeddings=NORMALIZE,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        # faiss rejects anything but contiguous float32, and silently so in
        # some builds; the cast is cheap and the copy is a no-op when the
        # backend already returned the right layout.
        return np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))

    def encode_passages(self, texts: list[str], *, show_progress: bool = False) -> np.ndarray:
        return self._encode(texts, self.passage_instruction, show_progress)

    def encode_queries(self, texts: list[str], *, show_progress: bool = False) -> np.ndarray:
        return self._encode(texts, self.query_instruction, show_progress)


class DenseRetriever:
    """Wraps a FAISS index plus the `chunk_ids` array mapping a result's
    positional index back to a database `chunk_id`. FAISS stores vectors and
    integer positions only, so the sidecar is the sole link back to the
    corpus, exactly as in `ticker.retrieval.bm25`."""

    name = "dense"

    def __init__(self, index, chunk_ids: list[str], encoder: DenseEncoder) -> None:
        self._index = index
        self._chunk_ids = chunk_ids
        self._encoder = encoder

    def __len__(self) -> int:
        return len(self._chunk_ids)

    def search(self, query: str, k: int) -> list[RankedChunk]:
        n = len(self._chunk_ids)
        if n == 0 or k <= 0:
            return []
        k = min(k, n)

        query_vector = self._encoder.encode_queries([query])
        scores, positions = self._index.search(query_vector, k)
        return [
            RankedChunk(chunk_id=self._chunk_ids[int(pos)], score=float(score))
            for pos, score in zip(positions[0], scores[0])
            # faiss pads with -1 when fewer than k vectors match. k is clamped
            # to n above so this should not fire, but a -1 would index the
            # sidecar from the end and return a plausible-looking wrong chunk.
            if int(pos) >= 0
        ]


def read_manifest(index_dir: Path | str) -> dict:
    return json.loads((Path(index_dir) / MANIFEST_FILENAME).read_text())


def load_retriever(
    index_dir: Path | str = DEFAULT_INDEX_DIR,
    encoder: DenseEncoder | None = None,
) -> DenseRetriever:
    """Factory matching the `module.path:callable` contract `scripts/pool.py`
    dynamic-imports: callable with no arguments, returns a `Retriever`.

    Every encoder-shaping parameter comes from the manifest the build wrote,
    not from this module's constants: model name, revision sha, batch size and
    the query instruction. Editing a constant here therefore cannot change how
    an already-built index is queried, it can only change the next build. Pass
    `encoder` to skip loading the model, which is what the unit tests do.
    """
    index_dir = Path(index_dir)
    chunk_ids_path = index_dir / CHUNK_IDS_FILENAME
    faiss_path = index_dir / FAISS_FILENAME
    if not chunk_ids_path.exists() or not faiss_path.exists():
        raise FileNotFoundError(
            f"no dense index at {index_dir} (need {FAISS_FILENAME} and "
            f"{CHUNK_IDS_FILENAME}). Run scripts/index_dense.py first."
        )

    import faiss

    manifest = read_manifest(index_dir)
    index = faiss.read_index(str(faiss_path))
    chunk_ids = json.loads(chunk_ids_path.read_text())

    if index.ntotal != len(chunk_ids):
        raise RuntimeError(
            f"dense index at {index_dir} holds {index.ntotal} vectors but the "
            f"chunk_ids sidecar names {len(chunk_ids)}. The index and the sidecar "
            "were written by different builds; rebuild with scripts/index_dense.py."
        )

    if encoder is None:
        encoder = DenseEncoder(
            load_model(manifest["model_name"], revision=manifest["model_revision"]),
            query_instruction=manifest["query_instruction"],
            passage_instruction=manifest["passage_instruction"],
            batch_size=manifest["batch_size"],
        )
    return DenseRetriever(index=index, chunk_ids=chunk_ids, encoder=encoder)


def pick_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def seed_everything(seed: int = SEED) -> None:
    """Seed python, numpy and torch, and ask torch for deterministic kernels.

    `warn_only=True` on `use_deterministic_algorithms`: the strict form raises
    at op dispatch for any kernel without a deterministic implementation, and
    on MPS that would take down a forward pass that is in fact reproducible on
    this device. Warning instead of raising keeps the check visible without
    making the build fail on an op that is not actually a source of run-to-run
    variance here.
    """
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_model(model_name: str = MODEL_NAME, *, revision: str | None = None, device: str | None = None):
    """A `SentenceTransformer` in eval mode on the chosen device.

    `revision` is threaded through rather than defaulted so `load_retriever`
    can pin the query-time model to the exact sha recorded in the manifest.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, revision=revision, device=device or pick_device())
    model.eval()
    return model


def cached_snapshot_revision(model_name: str = MODEL_NAME) -> str | None:
    """The commit sha of the locally cached snapshot, or None if nothing is cached.

    Reads the sha off the cache layout rather than calling `snapshot_download`
    with `local_files_only=True`. That call insists the snapshot hold every
    file in the upstream repo and raises `IncompleteSnapshotError` otherwise,
    which is the normal state here: the build fetches `model.safetensors` and
    the configs and skips `pytorch_model.bin` and `onnx/model.onnx`, which are
    the same weights in two formats this project never loads. Resolving off
    `config.json`'s parent directory gives the same sha and tolerates that.
    """
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(model_name, "config.json")
    if isinstance(cached, str):
        # huggingface_hub lays the cache out as snapshots/<commit_sha>/<file>
        return Path(cached).parent.name
    return None


def resolve_revision(model_name: str = MODEL_NAME) -> str:
    """The commit sha the build actually used, never `"main"`.

    A branch name in the manifest would let a moved upstream branch silently
    change what a saved index means, so this resolves to a sha or raises.
    Prefers the local cache, since that is the checkpoint the encode will
    load; falls back to the hub only when nothing is cached yet.
    """
    revision = cached_snapshot_revision(model_name)
    if revision:
        return revision

    from huggingface_hub import HfApi

    sha = HfApi().model_info(model_name).sha
    if not sha:
        raise RuntimeError(
            f"could not resolve a commit sha for {model_name}. Refusing to write "
            "a branch name into the manifest: the index would no longer identify "
            "the weights that built it."
        )
    return sha
