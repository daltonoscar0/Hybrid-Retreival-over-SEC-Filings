"""BM25 lexical retriever over `chunks`, built with `bm25s`.

Variant. Kamphuis, de Vries, Boytsov, Lin (2020), "Which BM25 Do You Mean? A
Large-Scale Reproducibility Study of Scoring Variants" (ECIR 2020), catalogue
five scoring variants that production systems disagree on: Robertson
(original/Okapi), ATIRE, BM25L, BM25+, and Lucene. `bm25s` reproduces all
five under its `method=` argument (see `bm25s.scoring._select_tfc_scorer` /
`_select_idf_scorer`). This module fixes `lucene` (`BM25_VARIANT` below):
its IDF term, `log(1 + (N - df + 0.5) / (df + 0.5))`, cannot go negative for
a high-document-frequency term the way Robertson's original can -- Robertson
needs a floor special case for terms in more than half the corpus, which
"customer," "risk," and "quarter" all are here. Lucene is also what
Elasticsearch and Solr ship, so it is a well-exercised default rather than a
research-only variant.

Tokenization is intentionally plain: lowercase, the scikit-learn
CountVectorizer word pattern, bm25s's bundled English stopword list, no
stemmer. No stemmer is a reproducibility choice as much as a quality one --
`bm25s.tokenize`'s stemmed-vocabulary path builds a Python `set` of token
strings and iterates it to assign ids, and set iteration order for strings is
sensitive to `PYTHONHASHSEED`, which would make the on-disk index
non-byte-identical across runs on this codebase's convention (see
`scripts/index.py` for the byte-identical rebuild check). Skipping the
stemmer sidesteps that hazard entirely rather than pinning the hash seed.

Query-time tokenization must use `tokenize_texts`, the exact function
`scripts/index.py` uses at build time. bm25s resolves query tokens to index
vocabulary by string lookup (`BM25.get_tokens_ids`), not by shared numeric
ids, so any drift between the two tokenizer calls silently drops query terms
rather than raising.
"""

from __future__ import annotations

import json
from pathlib import Path

import bm25s

from ticker.retrieval_types import RankedChunk

BM25_VARIANT = "lucene"

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
DEFAULT_DELTA = 0.5  # only affects bm25l / bm25+; unused for "lucene"

TOKEN_PATTERN = r"(?u)\b\w\w+\b"  # bm25s default: scikit-learn's CountVectorizer pattern
STOPWORDS = "english"  # bm25s's bundled list
STEMMER = None  # see module docstring: no stemmer, for reproducibility

DEFAULT_INDEX_DIR = Path("data/index/bm25")
CHUNK_IDS_FILENAME = "chunk_ids.json"
MANIFEST_FILENAME = "manifest.json"


def tokenize_texts(texts: list[str], *, show_progress: bool = False):
    """The one tokenizer both `scripts/index.py` and `BM25Retriever.search`
    call, so index-time and query-time tokenization cannot drift apart."""
    return bm25s.tokenize(
        texts,
        lower=True,
        token_pattern=TOKEN_PATTERN,
        stopwords=STOPWORDS,
        stemmer=STEMMER,
        return_ids=True,
        show_progress=show_progress,
    )


class BM25Retriever:
    """Wraps a `bm25s.BM25` index plus the `chunk_ids` array that maps a
    result's positional index back to a database `chunk_id` -- `bm25s`
    itself only knows integer positions, since the corpus text is not stored
    in the saved index (`corpus=None` at both index and load time)."""

    name = "bm25"

    def __init__(self, index: bm25s.BM25, chunk_ids: list[str]) -> None:
        self._index = index
        self._chunk_ids = chunk_ids

    def __len__(self) -> int:
        return len(self._chunk_ids)

    def search(self, query: str, k: int) -> list[RankedChunk]:
        n = len(self._chunk_ids)
        if n == 0 or k <= 0:
            return []
        k = min(k, n)

        query_tokens = tokenize_texts([query], show_progress=False)
        # An empty or all-stopword query tokenizes to zero surviving tokens.
        # bm25s scores every document 0.0 in that case rather than raising
        # (verified in tests/test_bm25.py), so this still returns k chunk_ids
        # with a flat zero score instead of crashing the caller.
        results = self._index.retrieve(
            query_tokens, k=k, corpus=None, show_progress=False, n_threads=0
        )
        doc_indices = results.documents[0]
        scores = results.scores[0]
        return [
            RankedChunk(chunk_id=self._chunk_ids[int(idx)], score=float(score))
            for idx, score in zip(doc_indices, scores)
        ]


def load_retriever(index_dir: Path | str = DEFAULT_INDEX_DIR) -> BM25Retriever:
    """Factory matching the `module.path:callable` contract `scripts/pool.py`
    dynamic-imports: no arguments, returns a `Retriever`. Loads the index
    `scripts/index.py` built and saved to `index_dir`; raises `FileNotFoundError`
    with a clear message if that has not happened yet."""
    index_dir = Path(index_dir)
    chunk_ids_path = index_dir / CHUNK_IDS_FILENAME
    if not chunk_ids_path.exists():
        raise FileNotFoundError(
            f"no BM25 index at {index_dir} (missing {CHUNK_IDS_FILENAME}). "
            "Run scripts/index.py first."
        )
    index = bm25s.BM25.load(str(index_dir), load_corpus=False, load_vocab=True)
    chunk_ids = json.loads(chunk_ids_path.read_text())
    return BM25Retriever(index=index, chunk_ids=chunk_ids)
