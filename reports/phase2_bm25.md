# Phase 2: BM25 arm

## What was built

`scripts/index.py` builds a BM25 index over `chunks` with `bm25s`, saved to
`data/index/bm25/`. `src/ticker/retrieval/bm25.py` wraps the saved index
behind `ticker.retrieval_types.Retriever` (`name` + `.search(query, k)`), and
exposes `load_retriever()`, a zero-argument factory matching the
`module.path:callable` contract `scripts/pool.py --bm25` dynamic-imports.
`scripts/search.py` is the run-file producer: retrieves top-100 for every
query in `data/queries.jsonl`, writes `data/runs/bm25.jsonl` in the
`{query_id, chunk_id, score}` contract `ticker.evaluation` documents, reports
latency, and prints spot-check results for a human to read. Tests are in
`tests/test_bm25.py`.

`data/queries.jsonl` held 60 candidate queries when this work started and 40
by the time the smoke run below was captured -- the human hand-review pass
`reports/phase2_eval_harness.md` calls for (drop weak queries, fix wording)
ran during this session. All numbers below are against the current 40-query
file; re-running `scripts/search.py` reproduces them against whatever
`data/queries.jsonl` holds at the time.

No relevance judgments exist (`data/qrels/standard.jsonl` is empty), so
nothing here computes or reports nDCG, MRR, or Recall. `scripts/evaluate.py`
still writes only the stub table -- `data/runs/bm25.jsonl` is on disk and
ready for it once judging happens.

## BM25 variant

Kamphuis, de Vries, Boytsov, Lin (2020), *Which BM25 Do You Mean? A
Large-Scale Reproducibility Study of Scoring Variants* (ECIR 2020), catalogue
five scoring variants: Robertson (original/Okapi), ATIRE, BM25L, BM25+, and
Lucene. `bm25s` reproduces all five under `method=`. This index uses
**`lucene`**: its IDF term, `log(1 + (N - df + 0.5) / (df + 0.5))`, cannot go
negative for a high-document-frequency term the way Robertson's original can,
and terms like "customer," "risk," and "quarter" sit in more than half this
corpus's 146,449 chunks -- exactly the regime where Robertson needs a floor
special case and Lucene does not. `k1=1.5, b=0.75` (bm25s defaults);
`delta=0.5` is recorded but unused by `lucene`.

Tokenization: lowercase, the scikit-learn `CountVectorizer` word pattern
(`\b\w\w+\b`), bm25s's bundled English stopword list, no stemmer. No stemmer
is a reproducibility choice, not just a quality one -- see
`ticker/retrieval/bm25.py`'s module docstring for why `bm25s`'s stemmed
vocabulary path is not byte-identical across runs (`PYTHONHASHSEED`-sensitive
set iteration).

## Index build

Full corpus, single build, rebuilt against the corpus as it stands after the
Phase 1 short-Item-7 repairs:

| | |
|---|---|
| chunks indexed | 148,097 |
| vocabulary | 22,349 tokens |
| fetch from DuckDB | 0.5s |
| tokenize | 7.3s |
| build index | 6.6s |
| save to disk | 0.1s |
| **total** | **14.5s** |
| index size on disk | 99.3 MB |

`scripts/index.py --verify-repeat` rebuilds into a scratch directory and
byte-diffs every artifact (`data.csc.index.npy`, `indices.csc.index.npy`,
`indptr.csc.index.npy`, `vocab.index.json`, `params.index.json`,
`chunk_ids.json`) plus every non-timing manifest field. Ran against the full
corpus: all six files byte-identical, all ten manifest fields matched.
`manifest.json` also records
`bm25s==0.3.10`, Python 3.12.13, and a SHA256 over the ordered
`(chunk_id, text)` pairs the index was built from -- BM25 has no pretrained
weights to pin a revision hash to, so the corpus fingerprint plus library
version plus hyperparameters is the full input surface, and this is its
equivalent of a model revision hash.

## Retrieval latency

Top-100 for every query in `data/queries.jsonl` (40 at the time of this run),
single-threaded, on the loaded index (index load itself is not included, and
is sub-second):

| | |
|---|---|
| mean | 2.0 ms/query |
| median | 1.3 ms/query |
| p95 | 2.2 ms/query |
| max | 25.6 ms/query |
| total, 40 queries | 0.1 s |

The max is the first query of the run and is warm-up, not a tail: the second
slowest is under 3 ms.

## Pooling sanity check

`scripts/pool.py --bm25 ticker.retrieval.bm25:load_retriever` (output
directed away from `data/pool/pool.jsonl`, since dense retrieval does not
exist yet and the real pool per `reports/phase2_eval_harness.md` waits on
both sources): pool size went from min 20 / max 20 / mean 20.0
(keyword-seed-only, capped at `keyword-k=20` on every query) to min 25 / max
40 / mean 37.5 with BM25 wired -- BM25 is surfacing chunks the substring
keyword seed misses on every query, which is the whole point of pooling more
than one source. (Per-query pool composition is unaffected by the 60-to-40
query trim above; only the total judgment count, 1,502 now, scales with it.)

## Spot check (not an evaluation)

Three queries, top 3 of 100, chosen for spread: an exact-phrase lexical
query, a semantic/paraphrase query with no single fixed term, and a
natural-language question.

**q01 "customer concentration risk"** -- all three hits are Western Alliance
Bancorp 10-K "Concentrations of Credit Risk" language (2023, 2024, 2021
filings), on the bank's own lending-concentration limits. Plausible: close
lexical overlap on "customer," "concentration," and the risk-factor framing,
though it is credit-concentration risk specifically, not a generic reading of
the query.

**q03 "management commentary on gross margin compression"** -- top two are
Intel's FY2019 10-K MD&A gross-margin discussion, on-topic. Third is a Zions
Bancorporation passage on commercial-loan underwriting that shares "margin"
and "management" vocabulary but is not gross-margin commentary -- a real
example of BM25's lexical-only ceiling on a paraphrase-heavy query, the kind
of result the dense arm in Phase 3 exists to fix.

**q60 "why did operating expenses increase this quarter"** -- top hits are
onsemi and NVIDIA 10-Q MD&A passages explicitly discussing quarter-over-quarter
operating expense increases and their drivers (COVID disruption costs,
Mellanox integration costs). Plausible despite the natural-language phrasing,
because the content words ("operating," "expenses," "increase," "quarter")
carry the match; "why" and "did" are stopped.

Full detail (every query in `data/queries.jsonl`, top 100 each) is in
`data/runs/bm25.jsonl`; rebuild it with `uv run python scripts/search.py
--retriever ticker.retrieval.bm25:load_retriever`.

## Tests (`tests/test_bm25.py`, 8 new)

Tokenization on financial text (lowercasing, stopword removal, "Inc."
surviving with the period stripped, "$1.281 billion" and "U.S." losing their
single-character fragments to the `\w\w+` pattern), an empty query and an
all-stopword query both returning a flat zero-score result set instead of
crashing, scores identical across two independent index builds for three
different queries, and chunk_ids round-tripping from result position back to
known database rows (including confirming build order is `ORDER BY
chunk_id`, not insertion order). 152 tests pass repo-wide as of this report
(the rest pre-existing or added by concurrent work elsewhere in the repo
during this session, not this task), 2 pre-existing skips unrelated to this
work.

## Scope

This is the BM25 arm only. No dense retrieval, fusion, or ablation table was
built or touched. `pyproject.toml` picked up `sentence-transformers`,
`faiss-cpu`, and `scikit-learn` during this session from activity elsewhere
in the repo, not from this work -- the only dependency this task added was
`bm25s`.

## What I could not verify

- Byte-identical rebuild was checked on this machine only (`--verify-repeat`
  writes to a temp dir and diffs against the checked-in build); cross-machine
  reproducibility (different OS, different numpy build) is untested.
- Whether `lucene`'s advantage over `robertson` on this corpus's
  high-document-frequency terms actually changes retrieval quality rather
  than just the score's numeric range -- unmeasurable without judgments, and
  not claimed here.
- Retrieval latency was measured single-process, cold-cache-irrelevant
  (index already loaded), no concurrent load -- production-latency behavior
  under concurrent queries is untested.
