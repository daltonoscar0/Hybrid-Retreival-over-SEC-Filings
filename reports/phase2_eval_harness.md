# Phase 2: evaluation harness and judging tool

Built before any retriever, per PLAN.md section 3's Phase 2 note: an evaluation
harness written after the systems it measures gets tuned to flatter them, and
everyone reading the repo knows it. Nothing here builds BM25 or dense
retrieval. `scripts/evaluate.py` runs and produces a well-formed table with
zero systems and zero judgments on the table today; that is deliberate, and
is how the retrieval agent is unblocked without waiting on human judging.

## What exists

| Piece | File | Tests |
|---|---|---|
| Query set (40 queries) | `data/queries.jsonl` | -- |
| Retriever interface | `src/ticker/retrieval_types.py` | -- |
| Pooling | `src/ticker/pooling.py`, `scripts/pool.py` | `tests/test_pooling.py` (8) |
| Qrels JSONL I/O | `src/ticker/qrels.py` | `tests/test_qrels.py` (10) |
| Judging session | `src/ticker/judging.py`, `scripts/judge.py` | `tests/test_judging.py` (13) |
| Agreement / kappa | `src/ticker/agreement.py`, `scripts/agreement.py` | `tests/test_agreement.py` (9) |
| Metrics / CIs / significance | `src/ticker/evaluation.py`, `scripts/evaluate.py` | `tests/test_evaluation.py` (16) |

132 tests pass repo-wide (`uv run pytest -q`), 56 of them new in this phase, 2
pre-existing skips unrelated to this work (section-extraction fixtures not
yet reviewed). New dependencies added with `uv add`: `ranx`, `pytrec-eval`.

## Query set

`data/queries.jsonl`, 40 rows, one JSON object per line:
`{query_id, text, sector, note}`. `sector` is `semiconductors`,
`regional_banks`, or `both`, for stratifying later work (e.g. picking the 25
queries for Phase 6's novelty-conditioned re-judge). `note` is one line on
what the query is meant to exercise (exact-phrase lexical match, paraphrase
requiring semantic matching, a recency/novelty qualifier, sector jargon, a
natural-language question form, etc.), not a relevance judgment.

Composition: the 10 queries listed in PLAN.md Phase 2 verbatim, 15
semiconductor-specific, 15 regional-bank-specific, 20 cross-sector general.
Sector breakdown: 28 `both`, 17 `semiconductors`, 15 `regional_banks`.

These are candidates. PLAN.md is explicit that the query set is hand-verified,
the same status as the section-extraction fixtures -- read them, edit the
text, drop or add queries, and change `sector`/`note` as needed before
pooling for real.

## Pooling (`scripts/pool.py`)

Standard TREC pooling: top-20 from each wired retriever plus a keyword-seed
search, deduped by `chunk_id`, order shuffled with a seed tied to
`(base_seed, query_id)` so judging order carries no rank or source signal.
The justification (why not judge all 146,449 chunks, and the caveat that a
system outside the pool at judging time is scored unfairly against it) is in
`ticker.pooling`'s module docstring.

Retrievers are wired by dynamic import: `--bm25 module.path:callable` and
`--dense module.path:callable`, each returning an object with `.search(query,
k)`. Neither exists yet. Run today, `scripts/pool.py` pools from the keyword
seed alone and prints a loud warning that the pool is incomplete and
shouldn't be judged yet:

```
uv run python scripts/pool.py
no --bm25 or --dense retriever wired; pooling from the keyword seed only. ...
pooled 60 queries -> data/pool/pool.jsonl
pool size: min 20  max 20  mean 20.0  total judgments if all pools are graded: 1200
```

Every one of the 60 queries hit the `keyword-k=20` cap, i.e. the naive
substring search alone already finds 20+ matching chunks for every query in
this corpus -- expected, since the corpus is dense in financial boilerplate
that repeats these terms, and exactly why a pure keyword-seed pool is not
sufficient to judge against; it has no notion of relevance ranking, only
match count. Full 60-query pooling from the keyword seed alone took 2m53s
against the 146,449-chunk corpus (down from 6m23s after switching the match
from `ILIKE '%term%'` to `contains(lower(text), term)`, same semantics,
roughly 2.4x faster measured on this table). This is a one-time batch step,
not on the judging critical path.

## Judging tool (`scripts/judge.py`)

One chunk at a time: query text, on-screen instruction, chunk text, firm
ticker, form, item, filed date, period end. Single keystroke, no Enter: `0-3`
grades and advances, `s` skips and advances, `q` quits. Every keystroke
appends one line to the qrels file immediately, so killing the process loses
at most the item on screen, and re-running resumes at the first unjudged
pair. `--redo` re-presents already-judged pairs (last line wins on load, so a
correction doesn't require editing the file). `--query QID` restricts a
session to one or more queries.

Verified end-to-end against a scratch output path (never `data/qrels/`) with
a real pty driving single keystrokes against the live corpus: grading,
skipping, resuming, and the final summary all behave as designed. Per-item
latency after the query's pool is fetched (one batched DB query per query,
not per chunk) is effectively the terminal's own keystroke latency.

`--rejudge N` samples N already-graded (non-skip) pairs from the target qrels
file and re-judges them blind into a sibling `*.rejudge.jsonl` file (e.g.
`standard.jsonl` -> `standard.rejudge.jsonl`), never merged into the primary
file. `scripts/agreement.py` computes Cohen's kappa between the two passes,
treating a skip as its own category rather than dropping the pair, and prints
observed/expected agreement plus a confusion matrix.

`--instruction novelty` swaps the on-screen instruction to "relevant AND new
this period" and writes to `data/qrels/novelty.jsonl` -- same tool, same pool
file, separate output, never merged with the standard file.

## Evaluation (`scripts/evaluate.py`)

Metrics via `ranx`: nDCG@10, MRR@10, Recall@100, mean and a 2000-resample
percentile bootstrap CI per system, paired two-sided Student's t-test
significance markers between every system pair (`ranx.compare`, matching
PLAN.md and REFERENCES.md). One command writes both `reports/results.md` and
a LaTeX twin.

Retrieval systems are decoupled from this harness by a file contract, not a
shared library: any system writes `data/runs/<name>.jsonl`, one line per
`{query_id, chunk_id, score}`. `scripts/evaluate.py` discovers every
`*.jsonl` under `--runs-dir` and scores it. This is the interface the
retrieval agent needs; nothing else in this repo constrains how a run file
gets produced.

Verified against synthetic qrels/runs (not real judgments) end to end: a
deliberately strong system and a deliberately weak one over 8 synthetic
queries produced a correctly ordered table with populated CIs and a
significant win recorded for the strong system, in both the markdown table
and the LaTeX (`ranx`'s own boldface-best-plus-superscript table).

Runs with no judgments or no run files present today and writes a
well-formed stub table (headers, no rows, an explanation) rather than
failing, so `scripts/evaluate.py exists and runs` is true right now for the
retrieval agent regardless of judging progress.

## pytrec_eval cross-check

Deliverable 7. Wired once as a standing test rather than a one-off script run
(`tests/test_evaluation.py::test_ranx_agrees_with_pytrec_eval_on_ndcg10` and
its imperfect-run counterpart), so it stays proven on every future test run,
not just the day it was written. `ticker.evaluation.ndcg10_cross_check`
computes mean nDCG@10 for one system both ways -- `ranx.evaluate` and
`pytrec_eval.RelevanceEvaluator({"ndcg_cut.10"})` -- on identical synthetic
qrels and run data. Both tests assert agreement to 4 decimals; measured
agreement was exact to floating-point precision (both use the same
Jarvelin-Kekalainen linear-gain formula), not merely 4-decimal-close.

## What exists in data/qrels and data/spotcheck

Nothing. Both directories are empty (`data/qrels/`, `data/spotcheck/`). No
grade, real or synthetic, has been written to either path by this work --
every test and every smoke-test run above used `tmp_path` or `/tmp` scratch
files, never the real qrels location, per the hard constraint on this task.

| File | Judgments |
|---|---|
| `data/qrels/standard.jsonl` | 0 (does not exist) |
| `data/qrels/standard.rejudge.jsonl` | 0 (does not exist) |
| `data/qrels/novelty.jsonl` | 0 (does not exist) |
| `data/qrels/novelty.rejudge.jsonl` | 0 (does not exist) |

## What the human must sit down and do

1. **Read and edit `data/queries.jsonl`.** 60 candidate queries, hand-verify
   per PLAN.md: fix wording, drop weak queries, add ones I missed. ~20-30
   minutes.
2. **Wait for BM25 and dense retrieval to land**, then run `scripts/pool.py`
   with both wired. Judging the current keyword-only pool would grade a pool
   with no relevance ranking behind it and should not be treated as the real
   pass.
3. **Judge the standard pass**: `uv run python scripts/judge.py`. Pool size
   is unknown until real retrievers are wired (keyword-only alone already
   hits the 20-per-query cap on every query; a three-source pool after
   dedup will likely land somewhere in the 30-55 per query range based on
   typical TREC pool overlap, so roughly 1800-3300 judgments total across 60
   queries). At a few seconds per item with the single-keystroke design,
   that is very roughly 3-6 hours, likely two sittings rather than one, which
   is a real tension against PLAN's "judge in one sitting per query to keep
   criteria stable" -- budget query-sized breaks, not mid-query ones.
4. **Rejudge a sample for kappa**: `uv run python scripts/judge.py --rejudge
   30`, then `uv run python scripts/agreement.py`. ~15-20 minutes. That kappa
   number is the one PLAN says belongs in the README (there is no root
   README yet -- that is Phase 7's deliverable -- so it is captured here for
   now).
5. **Novelty-conditioned pass** on the 25 queries PLAN Phase 6 calls for:
   `uv run python scripts/judge.py --instruction novelty --query <id> ...`
   (repeat `--query`, or judge all 60 and use whichever 25 turn out most
   relevant to novelty once Phase 4's novelty layer exists). Roughly half
   the volume of step 3 if scoped to 25 queries, so another 1.5-3 hours.
6. **Run `scripts/evaluate.py`** once BM25/dense run files and real qrels
   both exist, to get the actual Phase 2 table.

Total human time once retrieval exists: order of a full day of judging, split
across sittings, plus under an hour for setup and agreement checking -- this
is the "two days, not two weeks" the tool was optimized for, not zero days.

## What I could not verify

- Real per-item judging speed with a human reader, as opposed to a scripted
  pty sending keys the instant a prompt appears. The pty smoke test proves
  the mechanics (one keystroke, immediate write, correct resume) but not
  reading speed against real filing text.
- Pool size and composition once BM25 and dense retrieval exist -- today's
  numbers are keyword-seed-only and will change, likely upward in overlap
  and downward in total unique chunks per query, once two more sources are
  pooled in.
- Whether 60 queries and a ~25-query novelty subset land the standard/
  novelty split PLAN Phase 6 expects; that split is explicitly a human call
  via `sector` and hand review, not something this tool decides.
