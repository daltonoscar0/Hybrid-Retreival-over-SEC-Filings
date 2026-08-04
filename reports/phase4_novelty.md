# Phase 4: the novelty layer

## What was built

Three modules under `src/ticker/novelty/`, one driver, and one labeling tool.

| Piece | File | Tests |
|---|---|---|
| Interpolated Kneser-Ney 5-gram | `kneser_ney.py` | `tests/test_kneser_ney.py` (29) |
| The contrast and its guardrails | `score.py` | `tests/test_novelty_score.py` (27) |
| Lazy Prices diff baseline | `lazy_prices.py` | `tests/test_lazy_prices.py` (29) |
| Corpus scoring pass | `scripts/score_novelty.py` | `tests/test_score_novelty.py` (7) |
| Blind spot-check tool | `scripts/spotcheck.py` | `tests/test_spotcheck.py` (25) |

Step 2 of PLAN's model ladder is cut, item 2 on PLAN section 4's de-scope
ladder. There is no neural language model and none was started.

## The measure

`novelty(s) = surprisal_firm(s) - surprisal_background(s)`, both terms in bits
per token, both models fit strictly before the same `as_of`, both over the same
vocabulary. The firm model has the sector background wired in as its parent, so
an n-gram the firm has never written falls through to what its peers write
rather than to a uniform floor.

Kneser-Ney is written out rather than imported. `nltk` is not in PLAN's stack,
and the stronger reason is that this file is the measurement instrument: novelty
is a difference of two numbers it produces, and a difference hides most of the
ways a language model can be quietly wrong. The one property that matters is
continuation counts, and the one common way of getting it wrong is backing off
to raw unigram frequency, which is stupid backoff wearing a Kneser-Ney label. It
produces a number, the number varies across sentences, and nothing downstream
complains. `tests/test_kneser_ney.py` pins it with a pair of words at equal raw
frequency and different continuation counts, written so that a raw-frequency
backoff returns exactly equal probabilities where the correct model does not.

Chen and Goodman's three-discount modified Kneser-Ney is not implemented. One
absolute discount per order, estimated as Ney's `n1 / (n1 + 2*n2)`. A systematic
estimation improvement applies to both terms of a contrast and largely cancels,
and PLAN's ladder says to ship the transparent thing first.

## The three invariants, and where each is enforced

**Invariant 1, time discipline.** `fit` takes `as_of` with no default and raises
naming the offending sentence id if any training sentence is dated at or after
it. `novelty_raw` raises in the other direction, if the sentence being scored is
dated before `as_of`, since that sentence may be in the model's training data
and the score would be the model recognising itself. The two predicates are
complementary, so nothing can be both trained on and scored. That was the first
thing the audit tried to break and it held.

**Invariant 2, the contrast.** Nothing in `score.py` returns a bare surprisal or
accepts one. `RawNovelty` deliberately does not carry its two component
surprisals as fields, because a field is a reach.

**Invariant 3, z-scores are within-document only.** Enforced by type, not by
comment. `novelty_zscored_for_display` returns `DisplayNovelties`;
`document_novelty_index` accepts only `DocumentNovelty` and rejects the display
types by isinstance before its generic check; `DisplayNovelty` carries a
sentence id and a z-score and no path back to the contrast. The mistake is a
`TypeError` at the call site. Two tests assert the mistake fails rather than
asserting the correct path works.

The auditor looked for a route around this and did not find one. The consumer
that persists scores writes `novelty_raw` and never constructs a
`DocumentNovelty`, so it has no z-score to leak.

## The Lazy Prices baseline

Normalized Levenshtein distance, TF-IDF cosine, and a sentence-level diff
labeling each current sentence inserted, modified or unchanged, aligned against
the same firm's most recent earlier filing carrying the same item.

It imports nothing from `kneser_ney` or `score`, directly or transitively, and
reimplements its own tokenizer and its own prior-period lookup to keep it that
way. Phase 5.1 validates the surprisal measure against these diff labels, so a
shared tokenizer or a shared alignment would make that AUC partly a measurement
of the shared code: a bug in the shared part would move the measure and its own
validator in the same direction and never show up in the number.

TF-IDF is fit on the pair, not on a corpus. A corpus fit would buy a real rarity
weighting and cost two things. Document frequencies over the whole corpus are
taken over filings from after t, which is the leak invariant 1 exists to
prevent. And the validator of a leak-sensitive measure should be leak-free by
construction rather than by care: a pairwise fit depends on nothing but the two
strings, so there is no `as_of` to get wrong and no corpus version to record.

The modified threshold is 0.40, chosen as the middle of a measured trough over
68 section pairs and 25,136 sentences. Sliding it across the whole trough
relabels 4.8% of current sentences, and 5.1's label is binary with inserted and
modified both counting as changed, so no threshold anywhere in [0, 1) moves that
number at all.

## Cost, measured

| | |
|---|---|
| firm model, 10,634 sentences, plus sector background, 101,553 sentences | 17 s |
| scoring, once the pair is built | 1,416 sentences/s |

The background fit is the entire cost. Scoring is free by comparison. That
shaped two decisions and one bug.

It shaped the `--as-of` grid, which is where the audit found the serious
problem. Snapping the cutoff to the calendar quarter cuts the background fits
from one per filing to at most 48. It does not leak. It does bias 84% of
sentences toward false novelty, because a filing loses sight of the same firm's
earlier filings in the same quarter, and the dominant such pair is the earnings
8-K followed days later by the periodic report that recycles its language. The
default is now the exact cutoff. `reports/leak_audit.md` has the full finding.

It also shaped how the pass holds models. The first version cached every pair it
built, keyed by (firm, sector, cutoff). Fifty-one filings in, holding fifty-one
five-gram models over six-figure sentence counts, the OS killed the process.
Filings are now ordered firm-first so every filing sharing a pair is
consecutive, and exactly one pair is alive at a time.

## Scope of the scoring pass, and what it does not cover

The run is scoped to 10-K filings: 120 filings, one background fit each under
the exact cutoff, covering roughly 115,000 of the corpus's 298,375 sentences.
That is the scope, not a sample, and the 10-Qs and earnings exhibits are simply
not scored yet. `scripts/score_novelty.py` without `--form` covers everything
and costs proportionally more.

At the time of writing the pass is still running. The distribution table, the
mean-by-item preview, and the spot-check input file all wait on it.

## What a reader should be skeptical of

The sector background's membership was chosen with future information. The 20
CIKs were selected in 2026 by requiring continuous filing from 2020 through
2025, so every background model, at every `as_of`, is fit on firms known in
2026 to have survived. The strict `<` filter governs which of their sentences
the fit may read; it has nothing to say about which firms are eligible in the
first place. For regional banks the effect is concrete: SVB Financial,
Signature Bank and First Republic are excluded because they failed in 2023, so
a 2023 bank filing's control contains only banks that came through that year.
The bias direction is not measured. Excluding those firms keeps distress
language out of the background and pushes the contrast down; including them
would have pushed it up. `src/ticker/universe.py` records the selection rule
and why it is not fixable without breaking the six-year per-firm assumption.


Nothing here has been validated. Phase 5.1 is the test of whether this measure
means anything, and it has not run. Until the diff-agreement AUC exists, this is
a measure with a defensible construction, a tested implementation, an audit it
survived, and no evidence that its output tracks anything real. The construction
being careful is not the same as the number being meaningful, and PLAN says so
plainly: if the AUC comes back near 0.5, the measure is broken and no amount of
downstream framing rescues it.

The spot-check tool has never been run against a real score file. It has been
exercised end to end against a synthetic 1,000-sentence DuckDB, 100 items across
all ten deciles, 100 labels written, screens asserted free of the score, the
decile, and the rank. The join against the real corpus is untested.

Digits are kept as tokens rather than normalized to a placeholder, on the
argument that a figure nobody has filed before is unseen by both models so the
contrast cancels most of it. That argument is plausible and unverified. If the
top of the novelty distribution turns out to be mostly numbers, this is the
first thing to change.

The sector background excludes the firm but includes every other firm in the
sector, at every date before the cutoff. For a firm with an unusual business
inside its sector, that background is a weaker control than it looks.
