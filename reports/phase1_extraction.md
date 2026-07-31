# Phase 1: section extraction

## Corpus-wide extraction results

Every cached filing from the locked 20-company universe: 120 10-K, 360 10-Q, 486 8-K. Not the 24-filing fixture set, which is 2.5% of the corpus and, per the last fixture set's 100% against a corpus 92.5%, not representative of it on its own.

Three rates per item. The raw rate divides by every filing of that form. The extractable count removes the filings under "Excluded: no recoverable narrative" and the (accession, item) pairs under "Confirmed legitimate item omissions", neither of which is a section this extractor could have found. The adjusted rate divides by that. All three are shown because a rate quoted only after removing cases from its own denominator is not checkable.

| item | form | ok | all filings | raw | extractable | adjusted |
|---|---|---|---|---|---|---|
| 1 | 10-K | 111 | 120 | 92.5% | 111 | 100.0% |
| 1A | 10-K | 111 | 120 | 92.5% | 111 | 100.0% |
| 3 | 10-K | 111 | 120 | 92.5% | 111 | 100.0% |
| 7 | 10-K | 111 | 120 | 92.5% | 111 | 100.0% |
| 7A | 10-K | 110 | 120 | 91.7% | 111 | 99.1% |
| Part I Item 2 | 10-Q | 357 | 360 | 99.2% | 360 | 99.2% |
| Part II Item 1 | 10-Q | 334 | 360 | 92.8% | 337 | 99.1% |
| Part II Item 1A | 10-Q | 355 | 360 | 98.6% | 358 | 99.2% |
| EX-99.1 | 8-K | 486 | 486 | 100.0% | 486 | 100.0% |

These are census counts over the whole corpus, not estimates from a sample, so no confidence interval applies.

### Excluded: no recoverable narrative

The filed HTML primary document carries no item narrative at all. These filers incorporate the business description, risk factors, and MD&A by reference to an annual report filed as a separate exhibit, leaving the primary document as financial statement tables and XBRL. Counting non-table characters per 10-K across the universe, these occupy the entire low tail at 15,522 to 60,907 characters, the next filing above them has 155,320, and the corpus median is 383,652. No threshold anywhere inside that gap changes the membership. See NO_NARRATIVE_ACCESSIONS in scripts/make_extraction_fixture.py.

- EWBC 10-K 0001069157-20-000016
- FITB 10-K 0000035527-21-000100
- FITB 10-K 0000035527-22-000119
- FITB 10-K 0000035527-23-000122
- FITB 10-K 0001193125-20-057751
- HBAN 10-K 0000049196-20-000010
- KEY 10-K 0000091576-20-000007
- RF 10-K 0001281761-20-000010
- ZION 10-K 0000109380-20-000092

### Cross-reference sections (extracted, and counted separately)

263 extracted sections are under 1000 characters and consist of a pointer to a financial statement note or another part of the document rather than narrative prose. These are complete sections as filed, not truncations, and they are kept: dropping them would be deleting real corpus. They are counted here because a per-firm language model fit partly on cross-reference sentences is learning how that filer words a pointer, which is worth knowing when reading the Phase 5.2 novelty-by-item table.

| item | sections | median chars |
|---|---|---|
| 3 | 50 | 200 |
| 7A | 33 | 244 |
| Part II Item 1 | 149 | 188 |
| Part II Item 1A | 31 | 184 |

### Confirmed legitimate item omissions (not misses, excluded from the rate)

"Start marker not found" here means the item's heading is absent from the filing, confirmed correct rather than a miss: see LEGITIMATE_OMISSION_ACCESSIONS in scripts/make_extraction_fixture.py for the per-filer evidence (alternating quarters with and without the heading present, extracting cleanly whenever it is). Not in the denominator or numerator of any item's rate above.

| ticker | form | accession | item |
|---|---|---|---|
| ADI | 10-Q | 0000006281-20-000013 | Part II Item 1 |
| ADI | 10-Q | 0000006281-20-000087 | Part II Item 1 |
| ADI | 10-Q | 0000006281-20-000123 | Part II Item 1 |
| ADI | 10-Q | 0000006281-21-000022 | Part II Item 1 |
| ADI | 10-Q | 0000006281-21-000169 | Part II Item 1 |
| ADI | 10-Q | 0000006281-21-000197 | Part II Item 1 |
| ADI | 10-Q | 0000006281-22-000020 | Part II Item 1 |
| ADI | 10-Q | 0000006281-25-000023 | Part II Item 1 |
| ADI | 10-Q | 0000006281-25-000125 | Part II Item 1 |
| ADI | 10-Q | 0000006281-25-000144 | Part II Item 1 |
| RF | 10-Q | 0001281761-21-000043 | Part II Item 1A |
| RF | 10-Q | 0001281761-21-000067 | Part II Item 1A |
| TXN | 10-Q | 0000097476-20-000017 | Part II Item 1 |
| TXN | 10-Q | 0000097476-20-000026 | Part II Item 1 |
| TXN | 10-Q | 0000097476-21-000013 | Part II Item 1 |
| TXN | 10-Q | 0000097476-21-000020 | Part II Item 1 |
| TXN | 10-Q | 0000097476-21-000032 | Part II Item 1 |
| TXN | 10-Q | 0000097476-22-000028 | Part II Item 1 |
| TXN | 10-Q | 0000097476-22-000040 | Part II Item 1 |
| TXN | 10-Q | 0000097476-22-000048 | Part II Item 1 |
| TXN | 10-Q | 0000097476-23-000025 | Part II Item 1 |
| TXN | 10-Q | 0000097476-23-000035 | Part II Item 1 |
| TXN | 10-Q | 0000097476-23-000041 | Part II Item 1 |
| TXN | 10-Q | 0000097476-24-000021 | Part II Item 1 |
| TXN | 10-Q | 0001628280-20-014630 | Part II Item 1 |

### Items folded into a jointly presented section (not misses)

The filer put two item headings back to back and ran one narrative under both. The combined span is emitted under the earlier item and the later one is not emitted at all, because two sections over the same offsets would double every sentence in them through the chunker and into both language models. No text is lost. These count against the later item's rate above, which is the conservative reading: the item has no span of its own.

| ticker | form | accession | item |
|---|---|---|---|
| RF | 10-K | 0001281761-21-000012 | 7A |

### Genuine misses (unresolved)

Everything else: a real heading exists that this extractor did not find, or the document's structure defeated Part I/II disambiguation. Counted as a failure in the table above.

| ticker | form | accession | item | reason |
|---|---|---|---|---|
| EWBC | 10-Q | 0001069157-20-000048 | Part I Item 2 | start marker not found |
| EWBC | 10-Q | 0001069157-20-000048 | Part II Item 1 | start marker not found |
| EWBC | 10-Q | 0001069157-20-000048 | Part II Item 1A | start marker not found |
| FITB | 10-Q | 0000035527-20-000077 | Part I Item 2 | could not locate a clean Part I / Part II split (found Part I=yes, Part II=no) |
| FITB | 10-Q | 0000035527-20-000077 | Part II Item 1 | could not locate a clean Part I / Part II split (found Part I=yes, Part II=no) |
| FITB | 10-Q | 0000035527-20-000077 | Part II Item 1A | could not locate a clean Part I / Part II split (found Part I=yes, Part II=no) |
| FITB | 10-Q | 0001193125-20-137598 | Part I Item 2 | start marker not found |
| FITB | 10-Q | 0001193125-20-137598 | Part II Item 1 | start marker not found |
| FITB | 10-Q | 0001193125-20-137598 | Part II Item 1A | start marker not found |

## Fixture set (own candidates, human review pending)

Candidate spans over 24 fixture filings, extractor's own output only -- not yet checked against tests/fixtures/sections/expected/, which is empty until a human reviews the .candidate.json files below. This is not the corpus-wide result above and is not a substitute for it. The previous fixture set reported 100% against a corpus rate of 92.5%, because its 21 filings came from 5 CIKs that the extractor already handled. This set covers all 20.

| item | ok | total | rate |
|---|---|---|---|
| 1 | 12 | 12 | 100% |
| 1A | 12 | 12 | 100% |
| 3 | 12 | 12 | 100% |
| 7 | 12 | 12 | 100% |
| 7A | 11 | 12 | 92% |
| EX-99.1 | 2 | 2 | 100% |
| Part I Item 2 | 10 | 10 | 100% |
| Part II Item 1 | 10 | 10 | 100% |
| Part II Item 1A | 10 | 10 | 100% |

### Fixture failures

| ticker | form | accession | item: reason |
|---|---|---|---|
| RF | 10-K | 0001281761-21-000012 | 7A: presented jointly with Item 7: the two headings are adjacent and one combined narrative follows both. The combined span is emitted under Item 7 and is not repeated here, because two sections over the same offsets would double every sentence in it. |

### Accessions awaiting human review

tests/fixtures/sections/expected/ is empty. Every accession below has a .candidate.json under tests/fixtures/sections/raw/ ready to check against the paired .txt dump.

- INTC 10-K 0000050863-20-000011
- NVDA 10-Q 0001045810-21-000131
- TXN 10-K 0000097476-22-000009
- QCOM 10-Q 0000804328-23-000023
- AVGO 10-K 0001730168-24-000139
- MU 10-Q 0000723125-25-000021
- ADI 10-K 0000006281-20-000156
- NXPI 10-Q 0001413447-21-000062
- MCHP 10-K 0000827054-22-000094
- ON 10-Q 0001628280-23-026196
- ZION 10-K 0000109380-21-000082
- RF 10-Q 0001281761-25-000063
- HBAN 10-K 0000049196-23-000020
- KEY 10-Q 0000091576-21-000114
- FITB 10-K 0000035527-24-000088
- CFG 10-Q 0000759944-23-000124
- WAL 10-K 0001212545-24-000092
- EWBC 10-Q 0001069157-25-000096
- CFR 10-K 0000039263-20-000010
- CMA 10-Q 0000028412-21-000140
- CMA 10-K 0000028412-23-000094
- RF 10-K 0001281761-21-000012
- NVDA 8-K 0001045810-22-000163
- WAL 8-K 0001212545-23-000109
