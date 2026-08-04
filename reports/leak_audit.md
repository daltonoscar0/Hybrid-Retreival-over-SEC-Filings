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
finds over every judged query. The sequence is the intended one: fuse, then
evaluate. `reports/results.md` would have carried a `wsum` row
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

---

# Re-audit, Phase 5

Run again after the Phase 5 validation layer was written, against the same
invariants. Four of the prior fixes were re-verified and hold. The two LOWs
recorded and not fixed above are still open and still recorded. Eight new
findings; six are fixed below, two are documentation and are also fixed.

Nothing here changed a number, because no number had been produced yet:
`reports/results.md` is still a stub and Phase 5 had not run to completion.
That is timing, not innocence. Two of the findings reconstruct the original
CRITICAL's exact failure mode through routes the original fix does not block.

## The regression test for the CRITICAL did not test the fix

`tests/test_fusion.py`, `scripts/fuse.py`

The prior audit closed the CRITICAL by moving the tuned run to
`data/runs/tuned/` and pinning it with
`test_the_tuned_run_is_not_discoverable_by_the_ablation_ladder`. That test
constructs the tuned path itself and never calls `fuse.py`. What it pins is
that the glob is non-recursive and that `TUNED_SUBDIR` is a non-empty string.

Demonstrated rather than argued: reverting `scripts/fuse.py` to write
`wsum.jsonl` straight into the globbed directory, which restores the CRITICAL
in full, leaves that test green. Only the legacy-file refusal at the top of
`main` would have caught it, and that refusal had no test at all.

**Fix.** `test_fuse_main_writes_the_tuned_run_outside_the_glob` drives `main()`
through `sys.argv` over a real two-system run set and a real qrels file, then
asks `discover_runs` what it can see. It fails against the reverted line.
`test_fuse_main_refuses_to_run_when_a_legacy_tuned_run_sits_in_the_glob` covers
the refusal.

## One CLI flag reproduced the CRITICAL

`scripts/evaluate.py`, `src/ticker/evaluation.py`

`--runs-dir` is free. `scripts/evaluate.py --runs-dir data/runs/tuned` globs the
tuned directory, finds `wsum.jsonl`, and scores it over every judged query
including the ones that chose its weights, writing to `reports/results.md`
because the output path routes on the qrels stem. The subdirectory protects the
default value of a flag, which is the same class of convention that moving the
file was meant to replace.

**Fix.** `fuse.py` writes a `.held-out-only` marker into the tuned directory and
`discover_runs` raises on any directory carrying it. The refusal is a property
of the directory rather than of a flag's default.
`test_pointing_the_ablation_ladder_at_the_tuned_directory_is_refused` covers it.

## The invariant 3 type firewall did not survive serialization

`scripts/score_novelty.py`, `src/ticker/validation.py`

In process, invariant 3 holds by type: the display z-score is its own class and
the aggregation path rejects it by isinstance. On disk both a z-score and a raw
contrast are a float under `novelty`. The prior audit's clean finding here was
correct when written and became incomplete when Phase 5 added the first
cross-document reader of that file.

**Fix.** `score_novelty.py` writes `score_kind: raw_contrast` and
`validation.load_scores` raises on anything else or on its absence. The scoring
pass was restarted rather than the column backfilled: a provenance marker
written by a migration asserts something the writer never observed.

## Survivorship in the sector background was disclosed in one docstring

`src/ticker/universe.py`, `src/ticker/novelty/score.py`

The 20 CIKs were chosen in 2026 by requiring continuous filing from 2020
through 2025, so every sector background, at every `as_of`, is fit on firms
known in 2026 to have survived. The strict `<` filter governs which sentences
the fit may read and says nothing about which firms are eligible. For regional
banks this is concrete: SVB Financial, Signature Bank and First Republic are
excluded because they failed in 2023, so a 2023 bank filing's control contains
only banks that came through that year.

This is future information selecting the corpus that one of the two terms in
every novelty contrast is fit on. It is locked by PLAN section 1's corpus
decision and unfixable without
breaking the six-year per-firm assumption, so the finding is the disclosure
gap. It appeared in `universe.py`'s docstring and in no report.

**Fix.** Stated in `reports/README-notes.md` and in `phase4_novelty.md`'s
skepticism section, with the bias direction left unasserted because it is not
measured. Excluding the failed banks keeps distress language out of the
background and pushes the contrast down; including them would have pushed it
up. Which dominates is unanswered.

## A partial score file produced a clean-looking result

`scripts/validate.py`, `src/ticker/validation.py`

`JoinDiagnostics` counts rows lost inside the sections a score file contains,
so a firm the scoring pass never reached is structurally invisible to it: its
sections are not in the file to be counted as missing. The interrupted pass
that `data/novelty/scores.jsonl` came from was missing three companies
entirely, and would have produced a 100% join rate over 18 of 20 firms.

**Fix.** `validation.coverage` reconciles the file against the `filings` table,
scoped to the forms the file actually contains so a deliberate 10-K-only scope
does not read as incomplete. `validate.py` prints the shortfall, names the
unscored filings, and exits rather than writing a report unless
`--allow-partial` is passed.

## The 5.2 bootstrap reseeded inside its own loop

`src/ticker/validation.py`

`random.Random(seed)` sat inside the per-item loop, so every item drew an
identical stream of resample indices and two items with equal n got perfectly
correlated intervals. Each CI was individually valid; the table exists to be
read across items and would not have supported that.

**Fix.** Seeded once for the table.
`test_two_items_with_equal_n_get_independent_resamples` covers it.

## Two documentation defects

`src/ticker/novelty/score.py` claimed there was "exactly one place in the
codebase that writes a `filed_at <` predicate". There are three:
`db.prior_sentences`, `db.background_sentences`, and
`lazy_prices.align_prior_section`. All three are correct; the comment would
have sent an auditor of the time filter to one of them. Corrected to name all
three.

`reports/README-notes.md` described the query-set drops as "thin-material
rather than low-scoring". The recorded reason for q58 is that a query with
almost no relevant material gives a degenerate per-query nDCG, which is a
reason about the metric the systems are scored with. The note now states this.
The same paragraph quoted the substring counts as counts over 148,097 chunks;
they were taken over a 146,449-chunk build predating the Item 7 span repairs.
`query_set.md` keeps its original figures, now labelled as the pre-repair
record, and `pooling.py` and `retrieval/dense.py` carry the current count.

## Still open, deliberately

The two LOWs from the first audit. `fusion.held_out_score` trusts its
`train_ids` argument, and the query set was selected partly on corpus term
frequency. Both recorded, neither fixed.

Corpus-wide BM25 IDF and date-blind dense embedding were re-verified as the
PLAN section 1 scope decision rather than a leak. The disclosure still lives
only in `reports/README-notes.md`, which feeds a README that does not exist
yet, and neither retrieval module mentions it.

## What this audit did not do

It ran no tests. Every guard it reported was verified to exist rather than
verified to fire; the one exception is the reverted-line experiment above,
which was run afterwards to confirm the claim about the fusion test. It did
not read the ingest path, `sections.py` in full, or the agreement modules.
