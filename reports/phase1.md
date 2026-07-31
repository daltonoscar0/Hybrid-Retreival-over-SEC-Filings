# Phase 1: corpus build

## What was built

A download sweep over the locked 20-company universe, an item-boundary section
extractor, a financial-prose sentence splitter, a chunker, and an ingest driver
that rebuilds the DuckDB from the on-disk cache. Section extraction detail is in
`reports/phase1_extraction.md`; this report covers the corpus that came out.

## Corpus

966 filings, 20 companies, two sectors, filed 2020-01-06 through 2025-12-18.

| | Semiconductors | Regional banks |
|---|---|---|
| 10-K | 60 | 60 |
| 10-Q | 180 | 180 |
| 8-K with EX-99.1 | 243 | 243 |

| Table | Rows |
|---|---|
| filings | 966 |
| sections | 2,087 |
| sentences | 295,070 |
| chunks | 146,449 |

The 8-K sweep was scoped to filings under Item 2.02, Results of Operations,
rather than to any filing carrying an EX-99.1. Selecting on the exhibit alone
also returns dividend declarations, merger announcements, and debt offerings.

295,070 sentences is below PLAN's 300k to 600k estimate. The estimate assumed
whole filings; the corpus stores five 10-K items, three 10-Q items, and the
earnings exhibit, not the full documents. The gap is the unextracted items, not
lost text.

## Extraction

| Item | ok | Raw | Extractable | Adjusted |
|---|---|---|---|---|
| 10-K 1 / 1A / 3 / 7 / 7A | 111 each | 92.5% | 111 | 100% |
| 10-Q Part I Item 2 | 357 | 99.2% | 359 | 99.4% |
| 10-Q Part II Item 1 | 334 | 92.8% | 337 | 99.1% |
| 10-Q Part II Item 1A | 355 | 98.6% | 358 | 99.2% |
| 8-K EX-99.1 | 486 | 100% | 486 | 100% |

These are census counts over the whole corpus, not estimates from a sample, so
no confidence interval applies. Every failure is enumerated by accession in
`reports/phase1_extraction.md`.

Two classes of filing are excluded from the adjusted column, both enumerated.
Nine 10-K filings, all regional banks, carry no item narrative in the filed HTML
because they incorporate it by reference to a separate exhibit. The separation is
empirical rather than chosen: counting non-table characters, those nine span
15,522 to 60,907, the next filing has 155,320, and the median is 383,652.
Twenty-three quarterly filings from ADI, TXN, and RF omit Part II Item 1 or 1A
outright, verified by reading the documents rather than inferred from the
extractor's silence.

The exit criterion is above 95% per item type. The adjusted column meets it. The
raw column does not, for 10-K items and for Part II Item 1, and both are reported
because the adjustment is the whole question.

## Sentence and chunk distributions

Median 28 tokens, mean 43.6.

| Tokens | Share |
|---|---|
| 1-3 | 1.0% |
| 4-7 | 1.7% |
| 8-15 | 12.0% |
| 16-40 | 58.7% |
| 41-80 | 18.9% |
| 81+ | 7.8% |

The 1-3 token bucket is the one that matters. A splitter shredding tables into
fragments produces a spike there, which would contaminate the Phase 4 language
models with text that is not prose. At 1.0% there is no spike.

Mean sentences per section behave the way the documents do. Risk Factors 350,
MD&A 354, Business 264, Market Risk 28, Legal Proceedings 7.7. Legal Proceedings
is short because filers routinely satisfy it with a cross-reference to a
financial statement note.

Chunks average 3.99 sentences against a 4-sentence window, with a minimum of 1
where a section is shorter than the window.

## Integrity checks

Run against the built database, not asserted from the code.

- 0 sentences with a null `filed_at`
- 0 sentences whose denormalized `filed_at` disagrees with the parent filing
- 0 orphan sections
- 0 chunks with an empty `sentence_ids` list
- 20 distinct CIKs, matching the locked universe

## What a reader should be skeptical of

`tests/fixtures/sections/expected/` is empty. Every extraction rate above rests
on the extractor's own output with no human-verified ground truth behind it. The
fixture regression test collects zero cases and skips. Twenty-one accessions have
candidate spans ready for review, and until those are approved the extraction
numbers are self-reported. This is the largest open weakness in Phase 1.

The nine no-narrative filings are a real corpus loss, not only a reporting
footnote. Six are 2020 filings covering fiscal 2019, the earliest year in the
window, which contribute mostly to the prior side of the novelty contrast. The
three mid-window FITB filings are worse, because they thin the per-firm language
model for that issuer specifically.

The universe was selected in 2026 from firms still listed and filing
continuously, so it is survivorship-biased by construction. This bites hardest on
regional banks, where the excluded firms are excluded precisely because of the
2023 stress episode that produced the most interesting disclosure language in the
window. `DATA.md` carries the detail.

Item 3 spans bound against the nearest table-of-contents-listed heading rather
than a true section break, because named litigation subheadings inside Item 3 are
structurally indistinguishable from a real boundary. That is an approximation and
is documented as one in the extractor.

## Open

- 21 fixture accessions awaiting human review via `scripts/review_fixtures.py`
- 9 filings whose narrative is recoverable only from a separate exhibit, not
  attempted
- 9 remaining 10-Q extraction misses across EWBC and FITB 2020 filings
