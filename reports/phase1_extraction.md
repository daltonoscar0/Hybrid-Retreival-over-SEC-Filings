# Phase 1: section extraction

## Corpus-wide extraction results

All 966 cached universe filings, not the 21-filing fixture set. Two rates are
given for every item. The raw rate divides by every filing of that form. The
adjusted rate excludes the filings under "Excluded: no recoverable narrative"
and the pairs under "Confirmed legitimate item omissions", neither of which is
a section this extractor could have found. Both are shown because a rate quoted
only after removing cases from its own denominator is not checkable.

| item | ok | all filings | raw | extractable | adjusted |
|---|---|---|---|---|---|
| 1 | 111 | 120 | 92.5% | 111 | 100.0% |
| 1A | 111 | 120 | 92.5% | 111 | 100.0% |
| 3 | 111 | 120 | 92.5% | 111 | 100.0% |
| 7 | 111 | 120 | 92.5% | 111 | 100.0% |
| 7A | 111 | 120 | 92.5% | 111 | 100.0% |
| EX-99.1 | 486 | 486 | 100.0% | 486 | 100.0% |
| Part I Item 2 | 357 | 360 | 99.2% | 359 | 99.4% |
| Part II Item 1 | 334 | 360 | 92.8% | 337 | 99.1% |
| Part II Item 1A | 355 | 360 | 98.6% | 358 | 99.2% |

### Excluded: no recoverable narrative

The filed HTML primary document carries no item narrative at all. These filers
incorporate the business description, risk factors, and MD&A by reference to an
annual report filed as a separate exhibit, leaving the primary document as
financial statement tables and XBRL.

The set is not a judgment call. Counting non-table characters per 10-K across
the universe, these nine occupy the entire low tail at 15,522 to 60,907, the
next filing above them has 155,320, and the corpus median is 383,652. The
94,413-character gap separates them cleanly, and no threshold anywhere inside
that gap changes the membership.

| ticker | form | accession | non-table chars |
|---|---|---|---|
| FITB | 10-K | 0000035527-23-000122 | 15,522 |
| FITB | 10-K | 0000035527-22-000119 | 15,654 |
| FITB | 10-K | 0000035527-21-000100 | 15,732 |
| ZION | 10-K | 0000109380-20-000092 | 21,319 |
| HBAN | 10-K | 0000049196-20-000010 | 30,766 |
| RF | 10-K | 0001281761-20-000010 | 48,910 |
| EWBC | 10-K | 0001069157-20-000016 | 51,504 |
| KEY | 10-K | 0000091576-20-000007 | 57,330 |
| FITB | 10-K | 0001193125-20-057751 | 60,907 |

Spot-checked rather than inferred from the character count alone. In EWBC
0001069157-20-000016, the strings "our business" and "we are subject to" appear
zero times, "competition" once, and "incorporated by reference" 34 times. Its
longest non-table lines are horizontal rule characters. There is no narrative in
the document to extract.

All nine are regional banks. Six are 2020 filings covering fiscal 2019.

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

### Genuine misses (unresolved)

Everything else: a real heading exists that this extractor did not find, or the document's structure defeated Part I/II disambiguation. Counted as a failure in the table above.

The five 2020 bank 10-Ks previously listed here (EWBC, HBAN, KEY, RF, ZION) were
reclassified into the no-narrative set above after the non-table character
analysis showed they contain no item text to find. That reclassification is why
the 10-K adjusted rate is 100% rather than 95.7%.

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

Candidate spans over 21 fixture filings, extractor's own output only -- not yet checked against tests/fixtures/sections/expected/, which is empty until a human reviews the .candidate.json files below. Not the corpus-wide result above; this is 4% of the corpus and, per experience with INTC, not representative of it on its own.

| item | ok | total | rate |
|---|---|---|---|
| 1 | 9 | 9 | 100% |
| 1A | 9 | 9 | 100% |
| 3 | 9 | 9 | 100% |
| 7 | 9 | 9 | 100% |
| 7A | 9 | 9 | 100% |
| EX-99.1 | 4 | 4 | 100% |
| Part I Item 2 | 8 | 8 | 100% |
| Part II Item 1 | 8 | 8 | 100% |
| Part II Item 1A | 8 | 8 | 100% |

### Fixture failures

None.

### Accessions awaiting human review

tests/fixtures/sections/expected/ is empty. Every accession below has a .candidate.json under tests/fixtures/sections/raw/ ready to check against the paired .txt dump.

- NVDA 10-K 0001045810-20-000010
- NVDA 10-K 0001045810-23-000017
- NVDA 10-K 0001045810-25-000023
- NVDA 10-Q 0001045810-21-000064
- NVDA 10-Q 0001045810-24-000264
- INTC 10-K 0000050863-20-000011
- INTC 10-K 0000050863-24-000010
- INTC 10-Q 0000050863-21-000030
- INTC 10-Q 0000050863-25-000074
- WAL 10-K 0001212545-21-000085
- WAL 10-K 0001212545-24-000092
- WAL 10-Q 0001212545-20-000163
- WAL 10-Q 0001212545-23-000149
- ZION 10-K 0000109380-22-000072
- ZION 10-K 0000109380-25-000040
- ZION 10-Q 0000109380-21-000192
- ZION 10-Q 0000109380-24-000134
- NVDA 8-K 0001045810-25-000115
- INTC 8-K 0000050863-25-000169
- WAL 8-K 0001628280-25-045685
- ZION 8-K 0000109380-25-000124
