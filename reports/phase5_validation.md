# Phase 5 validation: 5.1 diff agreement and 5.2 concentration by item

Both numbers here are computed without a human label. 5.3 and every
Phase 6 number are not, and are not in this report.

Scores read from `data/novelty/scores.jsonl`. Bootstrap 2000 resamples, seed 0.

## Corpus coverage

Forms scored: 10-K. 111 of 111 filings of those forms carry scores, across 20 of 20 companies.

## 5.1 Agreement with the Lazy Prices diff

AUC of the raw novelty contrast against the diff's binary changed label,
where changed is `inserted` or `modified` and unchanged is `unchanged`.
The diff is computed from text alone by `ticker.novelty.lazy_prices` with
no access to the surprisal code, so this is agreement between two
independent constructions.

| quantity | value |
|---|---|
| AUC | 0.7132 |
| 95% CI | [0.7013, 0.7244] |
| sentences | 92,863 |
| changed | 42,536 (45.8%) |
| unchanged | 50,327 |

The interval is a cluster bootstrap over sections, not over sentences.
Sentences inside one section share a firm, a filing date and one prior
alignment, so resampling them individually would treat correlated rows as
independent evidence and return an interval too narrow to be honest.

### Join coverage

These counts are the evidence that separates a measure that does not work
from a join that is broken. Both produce an AUC near 0.5.

| stage | count |
|---|---|
| sections in the score file | 554 |
| excluded, no earlier filing with this item | 100 |
| excluded, empty prior or current | 0 |
| aligned and used | 454 |
| sentences the diff labelled | 92,863 |
| of those, carrying a novelty score | 92,863 |
| join rate | 100.0% |

## 5.2 Mean novelty by item type

Raw contrast, sentence-level percentile bootstrap. The unit is the
sentence here because the quantity is a per-item corpus mean rather than
a ranking statistic.

| item | name | n | mean novelty | 95% CI |
|---|---|---|---|---|
| 1 | Business | 29,284 | -4.1011 | [-4.1430, -4.0597] |
| 1A | Risk Factors | 38,845 | -3.6917 | [-3.7222, -3.6608] |
| 3 | Legal Proceedings | 858 | -5.7575 | [-5.9911, -5.5202] |
| 7 | MD&A | 43,621 | -4.0594 | [-4.0863, -4.0301] |
| 7A | Market Risk | 2,059 | -3.3739 | [-3.5008, -3.2441] |

### Reading

The control is Item 1 Business, the most boilerplate-heavy item in the corpus, at -4.1011.

Set aside first: Item 3 at 7.7 sentences per section, Item 7A at 18.7 sentences per section. The substantial items run 264, 350, 393 sentences per section. A mean over a handful of sentences per
filing describes what the item is made of more than how novel it is.
Item 3 in this corpus is largely a cross-reference into the notes, so
its position at either end of the table is not evidence about whether
legal news is novel.

Against the control, for the substantial items:

- Item 1A Risk Factors (predicted by PLAN): above the control, intervals disjoint.
- Item 7 MD&A (predicted by PLAN): above the control, but the intervals overlap.

This is a partial result and the shortfall is the part worth stating. Item 7 MD&A is predicted by PLAN to carry more novelty than boilerplate and does not
separate from the control here. 5.2 is a sanity check rather than the
test of the measure, and 5.1 passed independently, so this does not
invalidate the measure. It does mean the by-item figure cannot be
presented as confirming the literature's prediction.

