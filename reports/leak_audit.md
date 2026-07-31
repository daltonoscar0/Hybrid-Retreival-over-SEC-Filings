# Leakage and evaluation-integrity audit

Adversarial read of every module that fits a model, builds an index, or scores
a system, against the repo's invariants. Run at the end of Phase 4, before
any of it is used to produce a number. Read-only: the auditor did not edit.

Six findings. One CRITICAL, one HIGH, two MEDIUM, two LOW. All four of CRITICAL
through MEDIUM are fixed below. Both LOWs are recorded and not fixed.

Two of the findings are the interesting kind: the code was correct and the
result would still have been wrong.

---

## CRITICAL. The tuned fusion run escaped into the directory the results table globs

`scripts/fuse.py`, `src/ticker/evaluation.py`

`ticker.fusion` enforces the held-out discipline three ways. `tune_weights`
returns weights and deliberately no score, so the tuning-set number cannot be
reported by accident. `held_out_score` takes both splits and raises on any
overlap. `scripts/fuse.py` prints the tuning-set number next to the held-out one
and labels it as not a result.

None of that mattered, because the tuned artifact left the process as a file.
`fuse.py` wrote `data/runs/wsum.jsonl`; `evaluation.discover_runs` globs
`*.jsonl` under that directory and `scripts/evaluate.py` scores everything it
finds over every judged query. The sequence is exactly the one RUN.md B5 asks
for: fuse, then evaluate. `reports/results.md` would have carried a `wsum` row
whose score covered the twenty queries that chose its weights, with a bootstrap
CI and a significance marker against the untuned arms.

It had not happened yet only because there are no judgments, so `results.md` is
a stub. No refactor was needed to trigger it.

**Fix.** The tuned run is written to `data/runs/tuned/wsum.jsonl`. The glob is
not recursive, so the ablation ladder cannot pick it up. Its number is reported
only in `reports/fusion_tuning.md`, only on the held-out split, and only
alongside every other arm scored on that same split, which `fuse.py` already
computed. `tests/test_fusion.py::test_the_tuned_run_is_not_discoverable_by_the_ablation_ladder`
pins it.

The general lesson is worth keeping. The contamination route was the filesystem,
not the call graph, so every in-process guard was satisfied.

---

## HIGH. The quarter as_of grid biased 84% of sentences toward false novelty

`scripts/score_novelty.py`

The scoring pass could snap each filing's cutoff to the first instant of its
calendar quarter, to avoid refitting the expensive sector background model once
per filing. The docstring argued this was safe because the training set is then
a prefix of what the exact cutoff would allow, never a superset, so the worst it
could do was understate the contrast.

The no-leak half of that is correct and the auditor could not break it. `fit`
rejects a sentence at or after `as_of`, `novelty_raw` rejects one before it, the
two predicates are complementary, and nothing is both trained on and scored.

The direction claim was wrong. Consider a firm filing its earnings 8-K on
January 20 and the 10-K whose MD&A recycles that language on February 10. Both
snap to January 1, so the firm model scoring the 10-K has not seen the 8-K,
although the 8-K predates it. The recycled text is unseen by the firm model, so
firm surprisal is high; the sector background has not seen it either, so the
subtraction does not cancel it; the contrast reads new for text the firm
published three weeks earlier. Under the exact cutoff the firm model would have
seen it and scored it strongly negative.

Measured on the corpus:

| | count |
|---|---|
| filings that are second or later within their own (firm, calendar quarter) | 486 of 966 |
| by form | 349 10-Q, 113 10-K, 24 8-K |
| sentences in those filings | 251,636 of 298,375 (84.3%) |

The bias lands precisely on the recycled boilerplate the measure exists to
detect, and it feeds 5.1's AUC, 5.2's by-item table, and the spot-check deciles.

**Fix.** `--as-of` defaults to `filing`, the exact cutoff. The docstring now
states the direction of the error correctly rather than backwards. `quarter`
stays available and documented, because a corpus-wide pass over all three forms
is hours of background fits, and anything feeding a Phase 5 validation is told
to use the default. The scores in `data/novelty/scores.jsonl` were regenerated
under the exact cutoff; the quarter-grid run was discarded.

For the 10-K scope this costs nothing. One 10-K per firm per year means one
filing per (firm, quarter) already, so the grid was buying no reuse there.

---

## MEDIUM. quarter_floor mixed two calendars

`scripts/score_novelty.py`

DuckDB returns `filed_at` as a TIMESTAMPTZ in the session timezone, which here
is America/New_York and not UTC. `quarter_floor` read `.year` and `.month` off
that and stamped the result `timezone.utc`. West of UTC the floor still lands
before the filing and nothing shows.

East of UTC it does not. A filing at 2020-03-31 17:00 Eastern renders as
2020-04-01 06:00 in a Tokyo session, the floor becomes 2020-04-01 00:00 UTC,
which is after the filing. That filing's own sentences then satisfy
`filed_at < as_of`, get fitted into its own firm model, and are silently dropped
from the output by the guard. Which sentences get scored would depend on the
`TZ` of the process.

**Fix.** `moment.astimezone(timezone.utc)` as the first statement, with a test
that a filing late on the last day of a quarter in a UTC+9 session still floors
to a cutoff before itself.

---

## MEDIUM. The incorporation-by-reference repair had no disjointness guard

`src/ticker/sections.py`

The joint-presentation repair drops Item 7A before appending the combined span
and says why. The incorporation-by-reference repair relocates Item 7 to a span
far later in the document and left every other item where it was, with no
intersection check. An item whose end boundary is the next item-label heading
overruns if there is no such heading between it and the F-pages, and its span
would then contain the relocated Item 7. Both would be inserted, doubling every
sentence between them into the chunker, both indexes, and both language models.

Measured, this does not occur: zero overlapping character ranges across all 966
filings, and all seven repaired filings are disjoint. No number is wrong.

**Fix.** The repair now drops any section overlapping the recovered span and
reports it as a failure. Two tests, one on a synthetic filing constructed so the
overlap actually happens, one asserting no two sections overlap in any repaired
filing. Re-run over the seven real repaired filings: the guard fires on none of
them and every span is unchanged, so the corpus did not need rebuilding.

---

## Re-audit of the four fixes

All four verified CLOSED against the code and against the tests, run rather than
read. The re-audit found three things the first pass could not have.

**The overlap test was vacuous.** The first version of
`test_incorporation_repair_drops_an_item_that_overlaps_the_recovered_span`
deleted the heading that bounds Item 7A, which makes 7A fail with "no closing
boundary" and never be emitted, so there was no section left to overlap. Both
its assertions held with the guard deleted outright. Producing a real overlap
needs two conditions at once: the joint-presentation repair must decline, which
means a non-target heading between the 7 and 7A headings, and 7A must keep a
boundary past the recovered narrative, which means a later heading added. The
test now constructs that, asserts on the failure text rather than only on the
absence of overlaps, and is paired with a second test that pins the fixture
itself by checking the overlap exists before the guard runs. Confirmed by
disabling the guard: the test fails.

**A legacy tuned run would still sit in the glob.** Moving the write does not
move a file someone already has. `scripts/fuse.py` now refuses to run while
`data/runs/wsum.jsonl` exists, rather than deleting a file it did not create.

**`quarter_floor` accepted a naive datetime.** `.astimezone` on a naive value
assumes system local time instead of raising, which is the same bug by a
different route. It now raises.

---

## LOW, recorded and not fixed

**`held_out_score` trusts the `train_ids` its caller hands it.** Nothing binds
them to the ids `tune_weights` actually optimized over, and `tune_weights`
narrows internally through `usable_query_ids`, so the two can already differ.
`scripts/fuse.py` passes the same list and the check does fire, so the current
call is correct. A caller that re-splits gets a contaminated number and no
exception. The fix is for `tune_weights` to return the weights and the ids
together in one object that `held_out_score` takes instead of a loose list.

**The 40 queries were selected partly on corpus term frequency.** Twenty of the
sixty candidates were dropped, several on substring counts over all 148,097
chunks: "going concern" 16 hits, "funded status" 54, "advanced packaging" 172.
That shapes the evaluation set using the corpus the systems are scored on. It
happened before pooling and before any judgment, every drop has a recorded
reason in `reports/query_set.md`, and the reasons are thin-material rather than
low-scoring. It is defensible and it belongs in the README as a sentence rather
than as a silence.

---

## One scoping disagreement between the two spec documents

BM25 fits its IDF over the whole corpus with no time predicate, and the dense
index embeds every chunk regardless of date, and both then score documents from
every date against that. Under the literal wording of the repo's invariant 1,
"any model or statistic scoring a document filed at t is fit only on documents
filed strictly before t", that is a corpus-wide fit applied across periods.
Under PLAN.md section 1, which scopes time discipline to "every novelty score",
it is out of scope, and retrieval here is static ad hoc rather than walk-forward.

No change made. The two documents disagree and someone will ask, so it is
written down rather than left for them to find.

---

## What the audit covered, and what it did not

Traced clean and reported as such:

- Invariant 4. Nothing under `src/ticker/retrieval/`, `fusion.py`, `pooling.py`,
  `evaluation.py`, or any of the retrieval scripts imports or reads a novelty
  score. Two textual matches exist repo-wide, both in prose.
- Invariant 3. `DisplayNovelty` carries only a sentence id and a z-score, with
  no path back to the contrast. `document_novelty_index` rejects both display
  types by isinstance before its generic check. The one consumer that persists
  scores writes `novelty_raw` and never constructs a `DocumentNovelty`, so it
  has no z-score to leak. No route found.
- No fitting function has a defaulted `as_of`. No `<=`, `>=` or `BETWEEN`
  against `filed_at` or `as_of` anywhere in `src` or `scripts`.
- The target firm is excluded from its background in SQL, not by a post-filter.
- Lazy Prices fits TF-IDF per pair, imports nothing from the surprisal code, and
  clamps its alignment to `min(as_of, filed_at)` with strict `<`.
- Only two functions append under the human-only paths, both reached solely from
  a human-driven CLI.

Not covered, and stated so rather than implied: the auditor ran no tests, so the
guards above are verified to exist rather than verified to fire. It did not read
the ingest path closely, so "`sentences.filed_at` always equals
`filings.filed_at`" rests on the schema comment and on a zero-disagreement query
against the built database rather than on the insert code. It did not read the
judging or agreement modules.
