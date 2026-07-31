# Data

## What is in the corpus

966 filings from 20 companies across two sectors, filed between 2020-01-01 and
2025-12-31.

| Form | Semiconductors | Regional banks | Total |
|---|---|---|---|
| 10-K | 60 | 60 | 120 |
| 10-Q | 180 | 180 | 360 |
| 8-K with EX-99.1 | 243 | 243 | 486 |
| **Total** | **483** | **483** | **966** |

Semiconductors: INTC, NVDA, TXN, QCOM, AVGO, MU, ADI, NXPI, MCHP, ON.
Regional banks: ZION, RF, HBAN, KEY, FITB, CFG, WAL, EWBC, CFR, CMA.

Extracted item sections are 10-K Items 1, 1A, 3, 7, and 7A; 10-Q Part I Item 2
and Part II Items 1 and 1A; and the whole of each 8-K Exhibit 99.1. Per-item
extraction rates and the enumerated failures are in `reports/phase1.md`.

Two sectors rather than one because the novelty measure in Phase 4 is a
contrast against a sector-matched background model. A background fit on the
whole corpus would mix bank language into a semiconductor firm's control, which
makes the control weaker than it needs to be.

## Licensing, and why there are no earnings call transcripts

EDGAR filings are public domain and redistributable. Earnings call transcripts
generally are not. The readable ones are published by Motley Fool and Seeking
Alpha under copyright, and a public repository that scrapes them is a
liability rather than a feature.

The 8-K Exhibit 99.1 earnings press release is the redistributable stand-in. It
carries the quarter's reported numbers and management's own narrative framing,
which is the part of a call that matters for this project, and it is public
domain. Exhibits were scoped to 8-Ks filed under Item 2.02, Results of
Operations and Financial Condition. Selecting on the presence of an EX-99.1
alone would also pull in dividend declarations, merger announcements, and debt
offerings, which are press releases but not earnings releases.

If transcript text is wanted later, these are licensed research datasets rather
than scraped pages:

- ECTSum (Mukherjee et al., EMNLP 2022), 2,425 transcripts with expert bullet
  summaries
- lamini/earnings-calls-qa on Hugging Face, CC-BY, with an open pipeline
- jlh-ibm/earnings_call, 188 transcripts with paired price series

## Access

Filings are retrieved with `edgartools`, which requires an `EDGAR_IDENTITY`
environment variable and uses it to satisfy the SEC's User-Agent requirement.
Three details cost time if you do not know them. CIKs must be zero-padded to ten
digits or the API returns HTTP 500. The full-text search parameter is
`locationCodes`, plural; the singular form is accepted, ignored, and returns
unfiltered results without an error. Rate limits are enforced.

Every filing is cached on disk under `data/raw/`, keyed by accession, so the
download sweep is resumable and the corpus rebuilds from cache without touching
the network.

## Known limitations

**Survivorship.** The 20 companies were picked in 2026 from firms still listed
and filing continuously under a single CIK for the whole window. Any company
that was acquired, delisted, or failed during the window is therefore absent.
For semiconductors that excludes Xilinx and Maxim Integrated, both acquired, and
Marvell, which orphaned its pre-2022 CIK on redomiciling. For regional banks it
excludes the 2023 failures, Silicon Valley Bank, Signature Bank, and First
Republic, along with several merged filers. The exclusion is not fixable
without breaking the six-years-of-filings-per-firm assumption that the per-firm
language models depend on, so it is documented rather than corrected. It matters
most for the regional bank sector, where the missing firms are missing precisely
because of the stress episode that generated the most interesting disclosure
language in the window.

**Nine 10-K filings have no machine-readable narrative.** All nine are regional
banks that incorporate their business description, risk factors, and MD&A by
reference to an annual report filed as a separate exhibit, leaving the HTML
primary document as financial statement tables and XBRL. The split is not a
judgment call. Measuring non-table characters per 10-K, these nine fall between
15,522 and 60,907, and the next filing above them has 155,320 against a corpus
median of 383,652. The 94,000-character gap separates them cleanly.

| Ticker | Filed | Non-table characters |
|---|---|---|
| FITB | 2021, 2022, 2023 | 15,522 to 15,732 |
| ZION | 2020 | 21,319 |
| HBAN | 2020 | 30,766 |
| RF | 2020 | 48,910 |
| EWBC | 2020 | 51,504 |
| KEY | 2020 | 57,330 |
| FITB | 2020 | 60,907 |

No amount of section-extraction work recovers text that is not in the document.
These are reported as extraction failures rather than backfilled from another
source, and every rate in `reports/phase1.md` is given both including and
excluding them so the exclusion cannot hide inside a headline number. Six of the
nine are 2020 filings covering fiscal 2019, which is the earliest year in the
window and therefore contributes mostly to the prior side of the novelty
contrast rather than to scored documents. The three mid-window FITB filings are
the more costly loss, because they thin out the per-firm language model for that
issuer.

**Some 10-Q items are absent by the filer's choice, not by extraction failure.**
Part II Item 1, Legal Proceedings, is omitted entirely from 23 quarterly filings
by Analog Devices and Texas Instruments. Checked directly, the phrase "legal
proceedings" appears in those documents only inside running prose, never as a
heading, so there is no section to extract. These are counted separately from
extraction misses. Reporting them as failures would understate extraction; folding
them silently into the denominator would overstate it.

**Sentence and chunk counts depend on the splitter.** The splitter is tuned for
financial prose, including dollar amounts, abbreviations such as Inc. and U.S.,
numbered list items, and tabular fragments. Filings are table-heavy, and a
splitter that shreds a table produces a spike of one-to-three-token sentences
that would contaminate the language models in Phase 4. The distribution is
checked for that spike rather than assumed to be clean.
