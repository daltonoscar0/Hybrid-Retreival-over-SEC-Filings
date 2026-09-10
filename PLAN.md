# Ticker

Hybrid retrieval over SEC filings with a per-sentence novelty layer.

Search that ranks passages by relevance, then tells you which of them are actually
new this period versus recycled boilerplate.

---

## 0. The one-paragraph version

BM25 plus dense retrieval fused with reciprocal rank fusion over a corpus of 10-K
and 10-Q item sections and earnings-release exhibits, evaluated on a hand-labeled
query set with nDCG/MRR and paired significance tests. On top of that, a novelty
score for every sentence: surprisal under a language model built only from that
firm's own prior filings, contrasted against surprisal under a background model, so
the score isolates "new for this company" from "rare English." Novelty is exposed
as a ranking signal and evaluated against a second, novelty-conditioned relevance
set, so the claim is measured rather than asserted.

---

## 1. Scope decisions to lock before writing code

These are the decisions that determine whether this finishes in five weeks or five
months. Make them once, write them down, stop revisiting.

**Corpus size.** 20 companies, 6 years, 2 sectors (suggest semiconductors and
regional banks; both have rich risk-factor language and genuine period-over-period
change). That is roughly 20 x 6 x 5 = 600 filings, which after section extraction
and chunking gives 300k to 600k sentences. Large enough that BM25 versus dense
differences are real, small enough that per-company language models have data and
that a full re-index takes minutes rather than hours.

**Why those two sectors.** Company-specific LMs need within-firm repetition to
learn what boilerplate looks like. Two sectors also gives a natural background LM:
sector-matched rather than corpus-wide, which is a stronger control.

**Document universe.** Filings only, no scraped transcripts. See section 2 for why
this is the right call and how to say so.

**Unit of retrieval.** Sentence-window chunk (4 sentences, stride 2), carrying
stable sentence IDs so novelty scores computed per sentence map back onto the
chunk for display and re-ranking without recomputation.

**Time discipline.** Every novelty score for a document filed at time t uses only
documents filed strictly before t. This is the same purging discipline as the
walk-forward evaluation in Edge, applied to text. Enforce it in the data layer, not
by convention, so it cannot be violated by a later refactor.

---

## 2. Data, and the licensing call

EDGAR is free, public domain, and redistributable. Earnings call transcripts
mostly are not: the readable ones live behind Motley Fool and Seeking Alpha and are
copyrighted. A public repo that scrapes them is a liability, and anyone at a company
whose business is licensed transcript data will notice immediately.

So:

**Primary corpus (EDGAR, public domain, redistributable):**
- 10-K Item 1 (Business), Item 1A (Risk Factors), Item 3 (Legal Proceedings),
  Item 7 (MD&A), Item 7A (Market Risk)
- 10-Q Part I Item 2 (MD&A), Part II Item 1 (Legal), Item 1A (Risk Factors)
- 8-K Exhibit 99.1 earnings press releases, which are the free, redistributable
  stand-in for call content and carry the quarter's actual numbers and narrative

**Transcript-flavored data, if you want it, from licensed research datasets rather
than scraping:**
- ECTSum (EMNLP 2022), 2,425 transcripts with expert bullet summaries
- lamini/earnings-calls-qa on Hugging Face, CC-BY, with an open pipeline
- jlh-ibm/earnings_call, 188 transcripts with paired price series

Put a `DATA.md` in the repo that states this explicitly. "I used the redistributable
corpus and cited research datasets for the rest" is a signal about engineering
judgment, not a limitation, and it costs one paragraph to make.

**Access.** `edgartools` for company lookup, filing retrieval, and text extraction.
Raw `efts.sec.gov/LATEST/search-index` as the fallback when you need full-text
search across filers. Gotchas: CIKs must be zero-padded to 10 digits or the API
500s, the parameter is `locationCodes` not `locationCode`, no API key is needed but
a real `User-Agent` is mandatory and rate limits are enforced.

**Section extraction is the part that will eat your time.** 10-K item boundaries are
not marked up consistently. Budget real time for it, write regression tests against
20 hand-checked filings, and log extraction failures loudly rather than silently
emitting a truncated section. A section that silently loses its second half will
poison every downstream number and you will not notice for two weeks.

---

## 3. Phases

### Phase 0, Skeleton and data contract (1 day)

Repo layout, `uv` environment, DuckDB schema, and the frozen record types.

```
filings(accession, cik, ticker, form, filed_at, period_end, url)
sections(section_id, accession, item, text, char_start, char_end)
sentences(sentence_id, section_id, ordinal, text, filed_at)
chunks(chunk_id, section_id, sentence_ids[], text)
```

`filed_at` on the sentence table is denormalized deliberately. Every novelty query
filters on it, and you want that filter to be impossible to forget.

**Exit criteria:** schema created, one filing ingested end to end, `pytest` green.

### Phase 1, Corpus build (3-4 days)

Downloader with caching and rate limiting. Section extractor with per-item
regression tests. Sentence splitter tuned for financial prose: dollar amounts,
"Inc.", "U.S.", numbered list items, and tabular fragments all break naive
splitters. Chunker.

**Exit criteria:** 600 filings ingested, extraction success rate above 95% per item
type with the failures enumerated in a report, sentence count and length
distribution sane on inspection.

### Phase 2, Lexical baseline and the evaluation harness (4-5 days)

Build the harness before the second retriever. If evaluation comes last it gets
compromised to make the numbers look good, and everyone reading the repo knows it.

- BM25 over chunks with `bm25s`
- Query set: 60 queries, graded relevance 0-3, hand-verified

**Building the query set without it taking two weeks.** Pool candidates from BM25
plus a dense retriever plus a keyword seed list, take the top 20 from each per
query, dedupe, and judge the pool. This is standard TREC pooling and it is defensible;
judging every chunk is not feasible and nobody expects it. Judge in one sitting per
query to keep your criteria stable, and re-judge 10 queries a week later to report
your own intra-annotator agreement. That number belongs in the README.

Query examples that exercise both lexical and semantic matching:
- customer concentration risk
- goodwill impairment charge in the current period
- management commentary on gross margin compression
- newly disclosed litigation
- foreign exchange impact on reported revenue
- changes to revenue recognition policy
- data center capital expenditure plans
- cybersecurity incident disclosure
- supply chain disruption affecting component availability
- share repurchase authorization increase

- Metrics via `ranx`: nDCG@10, MRR@10, Recall@100, with bootstrap confidence
  intervals and paired t-tests between systems

**Exit criteria:** BM25 numbers on the board with CIs. A single command reproduces
the table.

### Phase 3, Dense retrieval and fusion (3 days)

- Embeddings: start with `bge-base-en-v1.5`. Evaluate a finance-adapted model
  (Fin-E5 or FinLang) as a second arm. FinMTEB's finding that general-benchmark
  performance predicts financial-task performance poorly is exactly why you measure
  this on your own corpus rather than trusting a leaderboard.
- Index: FAISS flat is fine at this scale. Do not reach for a vector database.
- Fusion: RRF with k=60 per Cormack et al., plus a weighted variant tuned on a
  held-out query split via `ranx.optimize_fusion`. Report both, and report the
  tuned version's held-out number, not its tuning-set number.

**Exit criteria:** the ablation ladder table, BM25 / dense / RRF / weighted fusion,
with significance markers. Including the cases where fusion does not beat the best
single system. Those cases are the most interesting rows in the table.

### Phase 4, The novelty layer (5-7 days)

This is the part nobody else's demo has. It is also the part where it is easy to
build something that produces a number that means nothing, so the contrast design
matters more than the model choice.

**The measure.** For sentence `s` in a document from firm `f` filed at time `t`:

```
surprisal_M(s) = -(1/|s|) * sum_i log P_M(w_i | w_<i)

novelty(s) = surprisal_{f,<t}(s) - surprisal_{bg,<t}(s)
```

where `LM_{f,<t}` is fit on firm f's filings before t, and `LM_{bg,<t}` is fit on
the sector's filings before t excluding f.

The contrast is the whole point. Raw surprisal under the company model flags any
sentence containing unusual English, including jargon that is unremarkable in
context. Subtracting the background model cancels that: a sentence that is rare
generally and rare for this firm scores near zero, while a sentence that is ordinary
English but has never appeared in this firm's filings scores high. That is the
quantity a reader actually wants, and it is the same conditioning logic as the
garden-path surprisal work, with firm history in the role of context.

**Model ladder. Do them in order and stop when the validation in Phase 5 is
satisfied.**

1. Interpolated Kneser-Ney 5-gram per firm, backing off to the sector model.
   Cheap, fast, fully transparent, and a legitimate baseline in its own right. Ship
   this first and make everything downstream work against it.
2. Small causal LM (GPT-2-small or minigpt) with a per-firm LoRA adapter, or a
   single model conditioned on a firm prefix token. Per-token NLL.
3. Only if 2 clearly beats 1 on the Phase 5 validations. Frequently it will not, and
   "the n-gram model was competitive" is a finding worth reporting rather than
   hiding.

**Normalization, and a trap to avoid.** For in-document highlighting, z-score
novelty within the document so the UI colors are comparable. For any
cross-document aggregation, use the raw contrast. Z-scored scores are not
comparable across documents and quietly aggregating them will produce a
document-level novelty index that measures nothing.

**Also compute the Lazy Prices baseline.** Align each section to the same firm's
prior-period version, compute normalized Levenshtein distance and TF-IDF cosine
similarity, and run a sentence-level diff to label each sentence
inserted / modified / unchanged. This is a 200-line addition, it reproduces the
canonical measure in this literature, and those diff labels become your silver
standard in the next phase.

**Exit criteria:** novelty scores for every sentence, computed under the time
constraint, with the n-gram and diff-based measures both in place.

### Phase 5, Validating that novelty means something (3-4 days)

Four checks, cheapest first. The first two are non-negotiable.

**5.1 Diff agreement (free, and it is the real test).** Sentences the alignment
diff labels as inserted or modified should score higher than unchanged ones. Report
AUC of novelty score against the binary changed/unchanged label. If this is near
0.5, the surprisal measure is broken and no amount of downstream framing rescues it.
Fix it here.

**5.2 Section concentration.** Novelty should concentrate where the literature says
real changes concentrate: legal proceedings, risk factors, and executive discussion.
Report mean novelty by item type. This is a sanity check that also happens to be a
nice figure.

**5.3 Human spot check.** Sample 100 sentences stratified by novelty decile, label
them novel or boilerplate blind to the score, report agreement. One hour of work,
and it is the number a skeptical interviewer will ask for.

**5.4 Market reaction (optional, and treat it carefully).** Does document-level
aggregate novelty correlate with absolute cumulative abnormal return around the
filing date? Use raw contrast scores, not z-scored. A weak positive correlation is a
real result. A null is also a real result and you report it as one. Do not turn this
into a trading-signal claim; the sample is 600 filings and the honest framing is
"consistent with, underpowered for."

**Exit criteria:** a validation section with four numbers, written before you build
the demo UI so the UI reflects what the measure actually does.

### Phase 6, Novelty as a ranking signal, evaluated honestly (2-3 days)

The tempting mistake: fold novelty into the relevance score and report an nDCG
improvement. It will not improve nDCG, because your relevance judgments are about
relevance, not novelty, and mixing them makes the ranker worse at the thing being
measured.

The correct design is two evaluation sets:

- **Standard qrels:** relevance as judged in Phase 2. Novelty reranking should be
  approximately neutral here. Show that it is.
- **Novelty-conditioned qrels:** for 25 of the 60 queries, re-judge under the
  instruction "relevant *and* new this period." Novelty reranking should improve
  nDCG here.

Then report both columns side by side. "Improves the novelty-conditioned metric,
neutral on the standard one" is a much stronger and much more credible claim than a
single inflated number, and it demonstrates you understand what your own metric
measures. In the product, ship novelty as a toggle and a per-sentence highlight
rather than as a silent term in the ranking function.

**Exit criteria:** the two-column table. This table is the artifact.

### Phase 7, Demo and writeup (4-5 days)

- FastAPI backend, single-page frontend. Search box, results with sentence-level
  novelty highlighting, a novelty toggle, and a filter by company and period.
- README with the two-column results table above the fold, the validation numbers,
  the data licensing note, and a short methods section.
- One figure: novelty distribution by item type. One figure: the ablation ladder.

**Exit criteria:** someone who has never seen the repo can run it and understand the
contribution in ninety seconds.

---

## 4. De-scope ladder

If time compresses, cut in this order. Everything above the line still constitutes a
complete, honest project.

1. Cut the market-reaction validation (5.4) entirely
2. Cut the neural LM, ship n-gram surprisal only
3. Cut the finance-adapted embedding arm, ship `bge-base-en-v1.5` only
4. Cut to 10 companies and one sector
5. Cut the frontend to a CLI plus a static results page

--- do not cut below this line ---

6. The evaluation harness, the two-qrel design, or the diff-agreement validation

Cutting anything in 6 turns this from a measured result into a demo, and the
measurement is the entire differentiator.

---

## 5. Stack

```
python 3.12, uv
edgartools          EDGAR access and text extraction
duckdb              chunk store and metadata
bm25s               lexical retrieval
sentence-transformers + faiss-cpu    dense retrieval
ranx                evaluation, fusion, significance testing
transformers + peft  per-firm LM adapters (phase 4 step 2 only)
fastapi + uvicorn   demo backend
```

Deliberately no vector database, no orchestration framework, no RAG library. At this
corpus size they add dependencies and hide the parts that are worth showing.

---

## 6. What this demonstrates, per target

- **Perplexity Search ML:** the posting is retrieval, ranking, and RAG. This is that,
  with a real evaluation harness and a ranking signal you designed and validated.
- **Aiera:** a working small version of their product with a signal layer on top.
- **Jane Street ML Engineer:** the leakage discipline, the paired significance
  testing, and the willingness to report null results are the actual content.
- **Apple SWE:** clean systems work, tested extraction, reproducible pipeline.

---

## 7. Resume line

> Built a hybrid BM25 + dense retrieval system over SEC filings with an LM-surprisal
> novelty ranking layer, evaluated on nDCG against hand-labeled queries with paired
> significance tests.
