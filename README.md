# Ticker

Hybrid BM25 and dense retrieval over SEC filings, with a per sentence novelty
layer that separates language a company has never used before from the
boilerplate it reprints every year.

Search returns passages ranked by relevance. The novelty layer then tells you
which of those passages are actually new this period.

## Status

The system is built and the evaluation harness runs. The relevance judgments it
scores against are not written yet, so the retrieval comparison is unmeasured.
This section says which numbers exist and which do not, because the rest of the
repo is organised around not asserting the second kind.

Measured:

| quantity | value |
|---|---|
| corpus | 966 filings, 20 companies, 2 sectors, 2020 to 2025 |
| chunks indexed | 148,097 |
| sentences scored for novelty | 298,375 |
| novelty agreement with an independent diff (AUC) | 0.7132, CI [0.7013, 0.7244] |
| BM25 query latency | 1.5 ms mean |
| dense query latency | 54.2 ms mean, 164.1 ms p95 |
| tests | 370 passing |

Not measured yet:

* Whether dense retrieval beats BM25 on this corpus, or whether fusion beats
  either. The runs exist, the metrics code exists, the judgments do not.
* The two column table comparing standard relevance against novelty
  conditioned relevance, which is the result this project is built to produce.
* Whether the section extraction boundaries are correct. There are no approved
  fixtures, so every extraction rate currently reports the extractor grading
  itself.

## The novelty measure

For a sentence in a filing from firm f at time t:

```
novelty(s) = surprisal_firm(s) - surprisal_sector(s)
```

Both terms are interpolated Kneser Ney 5 gram models. The first is fit on
firm f's own filings before t. The second is fit on that firm's sector peers
before t, excluding f.

The subtraction is the point. Raw surprisal under a firm model flags anything
containing unusual English, including jargon that is unremarkable in context.
Subtracting a sector background cancels that. A sentence that is rare in
general and rare for this firm scores near zero. A sentence that is ordinary
English this firm has never written before scores high. That second thing is
what a reader actually wants.

Every fit takes an explicit `as_of` and filters on strict inequality. A model
scoring a document filed at t sees only documents filed before t, and that is
enforced in the data layer rather than by convention.

### Does it work

Two checks so far, neither of which needs a human label.

**Agreement with a sentence diff.** Align each section to the same firm's prior
period version, label every sentence inserted, modified or unchanged, and ask
whether novelty separates changed from unchanged. AUC is 0.7132, CI [0.7013,
0.7244] over 92,863 sentences, with a 100% join rate. The diff is computed from
text alone with no access to the surprisal code, so this is two independent
constructions agreeing. The threshold for abandoning the measure was an AUC
near 0.5.

**Concentration by item.** Novelty should be highest where the literature says
real change concentrates. Risk Factors sits above the Item 1 Business control
with disjoint intervals, which is the predicted result. MD&A does not separate
from the control, which is not. Items 3 and 7A average under 20 sentences per
section and are too short to read anything into.

That second check is a partial negative and it is reported as one.

## Corpus

Twenty companies across semiconductors and regional banks, six years, EDGAR
only. Two sectors rather than one because a per firm language model needs a
sector matched background to contrast against, and a sector matched control is
stronger than a corpus wide one.

| form | filings | sentences |
|---|---|---|
| 10-K | 120 | 114,667 |
| 10-Q | 360 | 144,659 |
| 8-K | 486 | 39,049 |

10-K items 1, 1A, 3, 7 and 7A. 10-Q Part I item 2 and Part II items 1 and 1A.
8-K exhibit 99.1 earnings releases.

There are no scraped earnings call transcripts. The readable ones are
copyrighted and a public repo that scrapes them is a liability. See `DATA.md`.

## How it works

Chunks are four sentence windows at stride two, carrying stable sentence ids so
a novelty score computed per sentence maps back onto a chunk without
recomputation.

**Lexical.** `bm25s` with the Lucene scoring variant, k1=1.5, b=0.75, no
stemmer. Naming the variant matters because production systems disagree about
five of them (Kamphuis et al., ECIR 2020). Lucene's IDF cannot go negative for
a term appearing in more than half the corpus, which "risk" and "quarter" both
do here. The index rebuilds byte identically, verified on the full corpus.

**Dense.** `BAAI/bge-base-en-v1.5` pinned to a resolved commit sha, never a
branch name, over a FAISS flat index. Flat means exact cosine with no
approximation and no recall knob, so the dense arm's numbers report the
embedding model rather than index hyperparameters. 461 MB, and the search is
faster than the query encode that feeds it.

**Fusion.** Reciprocal rank fusion at k=60, the published value, deliberately
not tuned. With 40 queries and no third split reserved for it, fitting k would
fit a free parameter on the same queries the result gets reported over. A
weighted variant tunes on a held out split and reports the held out number.

**Evaluation.** `ranx` for nDCG@10, MRR@10 and Recall@100 with bootstrap
confidence intervals and paired t tests. nDCG is cross checked against
`pytrec_eval` to four decimals as a test rather than a one off script, so it
re-checks on every run.

The harness was written before the second retriever existed. If evaluation
comes last it gets compromised to make the numbers look good, and everyone
reading the repo knows it.

## Query set and judging

Forty queries, graded 0 to 3, pooled from BM25, dense and a keyword seed list
at depth 20 each. The pool is 53.6 chunks per query, 2,144 in total.

Forty rather than the sixty originally planned. That is a deliberate trade and
it costs precision: every confidence interval is about 22% wider than the same
interval over sixty queries, and every paired test loses power by the same
factor. A non significant row should be read as underpowered rather than as a
null.

Judgments are human written. Nothing in this repo generates a relevance label,
and the paths that hold them are blocked at the tooling level. A model scoring
its own retrieval is not a result.

## Running it

```
uv sync
uv run python scripts/download_corpus.py
uv run python scripts/ingest_corpus.py
uv run python scripts/index.py
uv run python scripts/index_dense.py
```

The dense build encodes 148,097 chunks in about three and a quarter hours on an
M series laptop at 12.5 chunks per second. Everything else is minutes.

Then search, fuse and pool:

```
uv run python scripts/search.py \
    --retriever ticker.retrieval.bm25:load_retriever \
    --retriever ticker.retrieval.dense:load_retriever
uv run python scripts/fuse.py
uv run python scripts/pool.py \
    --bm25 ticker.retrieval.bm25:load_retriever \
    --dense ticker.retrieval.dense:load_retriever
```

Score novelty and validate it:

```
uv run python scripts/score_novelty.py --form 10-K --as-of filing
uv run python scripts/validate.py
```

Judge, then evaluate:

```
uv run python scripts/judge.py
uv run python scripts/evaluate.py
```

`scripts/judge.py` grades one chunk per keypress with no Enter, writes every
decision before showing the next chunk, and resumes where it left off.

## What to be skeptical of

**Time discipline covers novelty, not retrieval.** BM25 fits its IDF over the
whole corpus with no time predicate and the dense index embeds every chunk
regardless of date. Retrieval here is static ad hoc, not walk forward. Someone
who knows the leakage literature will spot this, so it is better stated than
defended later.

**The sector background is fit on a survivorship selected peer set.** The 20
companies were chosen by requiring continuous filing from 2020 through 2025,
which is a decision made with information from 2026. SVB Financial, Signature
Bank and First Republic are absent because they failed in 2023, so a surviving
bank's 2023 language is scored against a background containing no bank that
failed that year. The direction of the resulting bias is not measured and is
not asserted.

**The 40 queries were filtered partly on corpus term frequency.** Twenty of
sixty candidates were dropped, several on how much material the corpus held.
That shapes the evaluation set using the corpus the systems are scored on. Every
drop is listed with its reason in `reports/query_set.md`.

**The dense index rebuild is unverified at full scale.** Byte identical rebuild
is checked for BM25 on the whole corpus and for dense only on subsets, because
verifying it doubles a three hour build.

## Stack

Python 3.12 with uv. `edgartools`, `duckdb`, `bm25s`, `sentence-transformers`,
`faiss-cpu`, `ranx`. No vector database, no orchestration framework, no RAG
library. At this corpus size they add dependencies and hide the parts worth
showing.

## Layout

```
src/ticker/          library code
scripts/             one entry point per pipeline stage
tests/               370 tests
reports/             one report per phase, with the numbers
data/qrels/          human written relevance judgments
PLAN.md              the specification
DATA.md              corpus scope and the licensing position
REFERENCES.md        citations
```

## License

MIT, see `LICENSE`. Filing text is public domain from EDGAR.
