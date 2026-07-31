"""The retriever interface the evaluation harness consumes.

Neither BM25 nor dense retrieval exists yet -- that is a different agent's
deliverable, gated on this harness existing and running. `scripts/pool.py`
and `scripts/evaluate.py` need something to type against today, so the
interface is defined here rather than assumed: implement `Retriever` and
your ranker plugs into pooling and (once it writes run files, see
`ticker.evaluation`) evaluation with no changes to either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RankedChunk:
    chunk_id: str
    score: float


class Retriever(Protocol):
    """Anything with a name and a `search` method qualifies -- no base class
    to inherit from, so a bare BM25 or FAISS wrapper can conform without
    importing this module."""

    name: str

    def search(self, query: str, k: int) -> list[RankedChunk]:
        """Return up to `k` results, best match first."""
        ...
