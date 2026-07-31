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
| sections | 2,086 |
| sentences | 298,375 |
| chunks | 148,097 |

The 8-K sweep was scoped to filings under Item 2.02, Results of Operations,
rather than to any filing carrying an EX-99.1. Selecting on the exhibit alone
also returns dividend declarations, merger announcements, and debt offerings.

298,375 sentences is below PLAN's 300k to 600k estimate. The estimate assumed
whole filings; the corpus stores five 10-K items, three 10-Q items, and the
earnings exhibit, not the full documents. The gap is the unextracted items, not
lost text.

## Extraction

| Item | ok | All filings | Raw | Extractable | Adjusted |
|---|---|---|---|---|---|
| 10-K 1 / 1A / 3 / 7 | 111 each | 120 | 92.5% | 111 | 100% |
| 10-K 7A | 110 | 120 | 91.7% | 111 | 99.1% |
| 10-Q Part I Item 2 | 357 | 360 | 99.2% | 360 | 99.2% |
| 10-Q Part II Item 1 | 334 | 360 | 92.8% | 337 | 99.1% |
| 10-Q Part II Item 1A | 355 | 360 | 98.6% | 358 | 99.2% |
| 8-K EX-99.1 | 486 | 486 | 100% | 486 | 100% |

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

## Two extraction repairs, and what changed because of them

Item 7 is the only target item that is never legitimately short. Its length
distribution over the 111 extractable 10-Ks has a hard gap: seven filings between
95 and 480 characters, the eighth at 16,752, median 74,971. Nothing in between.
Both ends of that gap were a structural problem in the filing, not a regex bug,
and before this phase both were emitted as valid sections.

**Comerica, six filings, incorporation by reference.** All six of CMA's 10-Ks
satisfy Item 7 with a pointer: "Reference is made to the sections entitled ... on
pages F-4 through F-39 of the Financial Section of this report." The narrative is
in the same document, after an index block mapping each financial-section title
to an F-page. The extractor now anchors on that index block's position and takes
the first year-Overview heading after it, ending at the first financial statement
or audit-report heading. Anchoring on the index block rather than on the pointer's
own wording is what makes it survive the roll-forward: "2019 Overview and 2020
Outlook" becomes "2024 Overview", and the index block's rendering picks up stray
spaces inside page numbers and years on five of the six years.

| accession | Item 7 before | Item 7 after |
|---|---|---|
| 0000028412-20-000034 | 455 | 183,265 |
| 0000028412-21-000054 | 480 | 226,260 |
| 0000028412-22-000067 | 439 | 194,519 |
| 0000028412-23-000094 | 439 | 196,761 |
| 0000028412-24-000185 | 439 | 212,208 |
| 0000028412-25-000108 | 438 | 203,750 |

**Regions 2021, one filing, joint presentation.** This is a different failure and
it was diagnosed separately. RF's 2021 10-K puts the Item 7 and Item 7A headings
back to back, 95 characters apart, and runs one combined narrative under both.
The next-heading end rule gave Item 7 those 95 characters and filed the entire
352,647-character MD&A under Item 7A, Market Risk. No text was lost; it was
attributed to the wrong item, which would have put a full MD&A into the Phase 5.2
novelty-by-item table as Market Risk. The combined span is now emitted under
Item 7, and Item 7A is reported as jointly presented rather than emitted over the
same offsets, because two sections sharing a span would double every sentence in
it through the chunker and into both language models.

The repaired 2021 span is 352,742 characters against 231,036 to 269,394 for RF's
four other years, which is the check that matters: the year that needed repair
now sits in the same range as the years that never did.

**Cost of the fix.** Item 7A drops from 111 to 110, which is the RF 2021 filing
no longer emitted separately. That is the conservative reading and it is reported
as a failure rather than hidden. Item 7 stays at 111 and its content is now
correct on 7 filings where it previously was not.

## Minimum section lengths

`ticker.sections.MIN_SECTION_CHARS` sets a floor per item. Below it the span is
recorded in the failure table instead of emitted. PLAN section 2 warns that a
section that silently loses its second half poisons every downstream number, and
before this the extractor emitted a 95-character Item 7 as a valid MD&A.

Every floor is read off the corpus's own length distribution. The two kinds of
item need very different ones and one global threshold would defeat the check.
Narrative items have observed minima of 25,114 (Item 1), 24,154 (1A), 16,752 (7),
and 14,081 (Part I Item 2). Cross-reference items are routinely one sentence
pointing at a financial statement note, or the word "None", with observed minima
of 128 (Item 3), 201 (7A), 44 (Part II Item 1), and 124 (Part II Item 1A). Those
are complete sections as filed.

| Item | Floor | Shortest observed |
|---|---|---|
| 1 | 5,000 | 25,114 |
| 1A | 5,000 | 24,154 |
| 3 | 100 | 128 |
| 7 | 2,000 | 16,752 |
| 7A | 150 | 201 |
| Part I Item 2 | 2,000 | 14,081 |
| Part II Item 1 | 30 | 44 |
| Part II Item 1A | 100 | 124 |
| EX-99.1 | 500 | 3,982 |

The module's contract is that per-filing problems are returned in `failures` and
never raised, because raising would abort a corpus sweep on one bad document.
A below-minimum span is a per-filing problem, so it is recorded there. What
"never write a short section as valid" buys is checked directly against the built
database rather than asserted: zero sections are below their item's floor.

263 extracted sections are under 1,000 characters and are a pointer rather than
prose. They are kept, because dropping them would be deleting real corpus, and
counted in `reports/phase1_extraction.md`, because a per-firm language model fit
partly on cross-reference sentences is learning how that filer words a pointer.

## Fixture set

24 candidate filings, covering all 20 CIKs, both forms, both sectors, and 2020
through 2025. The previous set drew 21 filings from 5 CIKs, 6 of them from one,
and reported a 100% fixture extraction rate against a corpus rate of 92.5%. That
gap is what an unrepresentative fixture set looks like: the five filers in it were
the ones the extractor already handled, so it could not have caught either failure
above.

Selection is 20 base filings, one per company with the form alternating by
position, plus 4 named edge cases: CMA's incorporation by reference, RF's 2021
joint presentation, and two 8-Ks for EX-99.1 coverage. INTC needs no edge slot;
its base draw is a 10-K, and every INTC 10-K uses the integrated-report layout.
One filing per company rather than one of each form: 40 filings is close to two
hours of span review against the one hour budgeted, so coverage of all 20 CIKs
was kept and per-company coverage of both forms was given up. The failure modes
seen so far are filer-specific rather than form-specific.

## Sentence and chunk distributions

Median 28 tokens, mean 43.6.

| Tokens | Share |
|---|---|
| 1-3 | 0.9% |
| 4-7 | 1.7% |
| 8-15 | 12.0% |
| 16-40 | 58.7% |
| 41-80 | 18.9% |
| 81+ | 7.8% |

The 1-3 token bucket is the one that matters. A splitter shredding tables into
fragments produces a spike there, which would contaminate the Phase 4 language
models with text that is not prose. At 0.9% there is no spike.

Mean sentences per section behave the way the documents do. MD&A 393, Risk
Factors 350, 10-Q MD&A 304, Business 264, 10-Q Risk Factors 95, earnings exhibit
80, Market Risk 19, 10-Q Legal 8.0, Legal Proceedings 7.7. MD&A overtakes Risk
Factors as the longest item because of the two repairs above; before them the
seven affected filings contributed a few hundred characters each.

Chunks average 3.99 sentences against a 4-sentence window, with a minimum of 1
where a section is shorter than the window.

## Integrity checks

Run against the built database, not asserted from the code.

- 0 sentences with a null `filed_at`
- 0 sentences whose denormalized `filed_at` disagrees with the parent filing
- 0 orphan sections
- 0 chunks with an empty `sentence_ids` list
- 0 sections below their item's minimum length
- 20 distinct CIKs, matching the locked universe

## What a reader should be skeptical of

`tests/fixtures/sections/expected/` is empty. Every extraction rate above rests
on the extractor's own output with no human-verified ground truth behind it. The
fixture regression test collects zero cases and skips. Twenty-four accessions have
candidate spans ready for review, and until those are approved the extraction
numbers are self-reported. This is the largest open weakness in Phase 1, and it is
larger now than it was, because the two repairs above added 1.5 million characters
of text to the corpus on the strength of the extractor's own judgment.

The repairs are checked by magnitude and by consistency, not by an approved span.
Three tests run against the raw cache: all six Comerica years above 50,000
characters, Regions 2021 above 300,000 with 7A absent, and every NVDA 10-K still
producing exactly its five items. Those rule out the repair failing to fire and
the repair firing where it should not. They do not establish that either boundary
is in the right place to the character. The CMA and RF fixtures are in the review
set for that reason.

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

The Comerica Item 7 span runs from the year-Overview heading to the first
financial statement heading. Item 7A's referenced subsections sit inside that
range, because that is how the filer laid the document out. Item 7A's own
extracted span is the 293-character pointer, so nothing is double-counted, but
Market Risk language for that filer is inside MD&A rather than under 7A and the
Phase 5.2 by-item table will show it there.

## Open

- 24 fixture accessions awaiting human review via `scripts/review_fixtures.py`
- 9 filings whose narrative is recoverable only from a separate exhibit, not
  attempted
- 9 remaining 10-Q extraction misses across EWBC and FITB 2020 filings
