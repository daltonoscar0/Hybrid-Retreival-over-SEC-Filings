# Ticker — references

## Retrieval and fusion

**Reciprocal rank fusion.** Cormack, Clarke, Buettcher (2009), *Reciprocal Rank
Fusion outperforms Condorcet and individual rank learning methods*, SIGIR '09,
758-759. https://doi.org/10.1145/1571941.1572114 — the two-page paper the whole
hybrid-search world cites. RRF sums 1/(k + rank) across rankings; k=60 is the
convention and comes from this paper.

**BM25S.** Lù (2024), *BM25S: Orders of magnitude faster lexical search via eager
sparse scoring*. https://arxiv.org/abs/2407.03618 — precomputes score contributions
at index time into a sparse matrix. Reproduces the five BM25 variants catalogued in
Kamphuis et al. (2020), so you can state exactly which variant you used.
Repo: https://github.com/xhluca/bm25s

**ranx.** Bassani (2022), *ranx: A Blazing-Fast Python Library for Ranking
Evaluation and Comparison*, ECIR. https://github.com/AmenRa/ranx — nDCG/MAP/MRR,
paired t-tests with significance superscripts in the output table, LaTeX export, and
`fuse` / `optimize_fusion` for tuned fusion. This library does about 80% of Phase 2
and Phase 3's evaluation work.

**pytrec_eval.** Van Gysel & de Rijke (2018). https://arxiv.org/abs/1805.01597 —
the trec_eval interface, if you want to cross-check ranx's numbers against the
canonical implementation. Worth doing once and mentioning in the README.

**ir-measures.** https://ir-measur.es/ — unified frontend over eight metric
providers, useful if a reviewer asks for a measure ranx does not implement.

## Financial text and the novelty literature

**Lazy Prices.** Cohen, Malloy, Nguyen (2020), *Lazy Prices*, Journal of Finance
75(4), 1371-1415. https://doi.org/10.1111/jofi.12885 (working paper:
https://ssrn.com/abstract=1658471, NBER w25084) — the foundational result. Firms
that change their filing language subsequently underperform those that do not; a
long-nonchangers / short-changers portfolio earned up to 188bp monthly alpha in
their sample. Their measures are normalized Levenshtein distance and TF-IDF cosine
similarity between consecutive filings. Changes in language about the executive
team, litigation, and risk factors were the most informative.

This paper is the reason the novelty layer is a defensible idea rather than a cute
one, and it is the citation that makes the project legible to a finance audience.
It also defines your Phase 4 baseline: your surprisal measure should be compared
against their edit-distance measure, not proposed in a vacuum. Note their finding
that 86% of textual changes carried negative sentiment, which is useful context for
interpreting what your high-novelty sentences turn out to be.

**FinMTEB.** Tang & Yang (2025), *FinMTEB: Finance Massive Text Embedding
Benchmark*, EMNLP 2025 Main. https://arxiv.org/abs/2502.10990 /
https://aclanthology.org/2025.emnlp-main.179/ — 64 finance datasets, 7 tasks. Three
findings matter for you: general-benchmark performance correlates poorly with
financial-task performance, domain-adapted models beat general ones, and bag-of-words
beat every dense model on financial STS. That last one is a direct argument for
taking your BM25 arm seriously rather than treating it as a formality.
Repo: https://github.com/yixuantt/FinMTEB — includes Fin-E5, their adapted model.

**ECTSum.** Mukherjee et al. (2022), EMNLP 2022 Main.
https://arxiv.org/abs/2210.12467 — 2,425 earnings call transcripts paired with
expert bullet summaries derived from Reuters coverage.
Data and code: https://github.com/rajdeep345/ECTSum

## Data access

**EDGAR full-text search (EFTS).** `https://efts.sec.gov/LATEST/search-index` —
free, no key, no auth, covers 2001-present including exhibits. Requires a real
`User-Agent`. Boolean support: implied AND between terms, quoted exact phrases,
`-term` exclusion, OR clauses, wildcards. No natural-language search.
FAQ: https://www.sec.gov/edgar/search/efts-faq.html

Known gotchas worth writing into your client wrapper:
- CIKs must be zero-padded to 10 digits. `ciks=320193` returns a 500;
  `ciks=0000320193` works. Always `str(cik).zfill(10)`.
- The parameter is `locationCodes`, plural. The singular form is silently ignored
  and you get unfiltered results with no error.
- Multi-CIK is comma-separated in a single parameter.

**EDGAR submissions API.** `https://data.sec.gov/submissions/CIK##########.json` —
filing metadata per company: form type, filing date, accession number, document
index. This is how you enumerate what to download.

**edgartools.** https://github.com/dgunning/edgartools (MIT,
`pip install edgartools`) — typed objects over EDGAR, `filing.text()` for clean
extraction, XBRL parsing, 20+ form types. Requires an `EDGAR_IDENTITY` env var,
which is how it satisfies the SEC's User-Agent requirement.
Docs: https://edgartools.readthedocs.io/

## Datasets

- **ECTSum** — https://github.com/rajdeep345/ECTSum (2,425 transcripts + summaries)
- **lamini/earnings-calls-qa** —
  https://huggingface.co/datasets/lamini/earnings-calls-qa (CC-BY, open pipeline at
  https://github.com/lamini-ai/lamini-earnings-calls, so you can extend it to more
  calls yourself rather than scraping)
- **jlh-ibm/earnings_call** —
  https://huggingface.co/datasets/jlh-ibm/earnings_call (188 transcripts, 2016-2020
  NASDAQ, with paired stock prices and sector index — the price pairing makes this
  the convenient one for validation 5.4)

## Embedding models to evaluate

- `BAAI/bge-base-en-v1.5` — the default first arm
- `BAAI/bge-large-en-v1.5` — if the base model is the bottleneck
- Fin-E5 (from the FinMTEB repo) — finance-adapted e5-Mistral-7B; heavy, evaluate
  before committing to it
- FinLang — BGE fine-tuned on financial text; lighter middle option
- `BAAI/bge-reranker-v2-m3` — cross-encoder reranker for the top-50, if you add a
  reranking stage

Evaluate at least two on your own labeled query set. FinMTEB's central finding is
that you cannot read this choice off a leaderboard.
