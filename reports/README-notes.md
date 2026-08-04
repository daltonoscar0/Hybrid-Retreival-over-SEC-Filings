Notes for the README the human writes. Facts and numbers only, in the voice the
README should use, ordered roughly as the README will need them. Not a draft.

## The metric implementation is cross-checked against a second library

nDCG@10 is computed with `ranx`. `pytrec_eval`, the Python binding to the
official TREC evaluation tool, computes it a second way over the same qrels and
the same run, and the two are asserted equal to four decimals. This is a test,
`tests/test_evaluation.py::test_ranx_agrees_with_pytrec_eval_on_ndcg10`, not a
one-off script run, so it re-checks on every suite run rather than having been
true once. There is a second case for an imperfect run, because a run that
retrieves everything correctly agrees under almost any nDCG variant and would
not separate the two implementations.

What this rules out: a wrong gain function, a wrong discount, or a wrong
handling of a query whose judged set is smaller than the cutoff. What it does
not rule out: both libraries agreeing on a convention that does not match some
third one. `ranx` uses the Jarvelin-Kekalainen linear gain, matching
`pytrec_eval`'s `ndcg_cut`.

## Which BM25

`bm25s` with the `lucene` scoring variant, k1=1.5, b=0.75, no stemmer.

Kamphuis, de Vries, Boytsov and Lin (ECIR 2020), "Which BM25 Do You Mean? A
Large-Scale Reproducibility Study of Scoring Variants", catalogue five variants
that production systems disagree on: Robertson, ATIRE, BM25L, BM25+, and Lucene.
Naming which one is in use is the point of citing them. Lucene's IDF term,
log(1 + (N - df + 0.5) / (df + 0.5)), cannot go negative for a term in more than
half the corpus, which "customer", "risk" and "quarter" all are here; Robertson's
original can, and needs a floor special case. Lucene is also what Elasticsearch
and Solr ship.

No stemmer, for reproducibility as much as quality: `bm25s`'s stemmed-vocabulary
path iterates a Python set of token strings to assign ids, and set iteration
order for strings depends on `PYTHONHASHSEED`, so the on-disk index would not be
byte-identical across runs.

The index rebuilds byte-identically. `scripts/index.py --verify-repeat` rebuilds
into a scratch directory and diffs all six artifacts plus every non-timing
manifest field. Verified on the full 148,097-chunk corpus.

## Which embedding model, and pinned how

`BAAI/bge-base-en-v1.5`, revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`,
FAISS `IndexFlatIP` over L2-normalized vectors, which is exact cosine with no
approximation and no training step.

The revision is a resolved commit sha, never the string "main". A branch name in
the manifest would let a moved upstream repo silently change what a saved index
means. The retriever reads the model name, the revision, and the query
instruction prefix out of the index manifest rather than from module constants,
so the index and the thing querying it cannot drift apart.

The finance-adapted second embedding arm PLAN section 3 called for is cut. So is
the neural language model, and so is the 5.4 market-reaction validation. Those
are de-scope decisions taken deliberately, in the order PLAN section 4 lists
them, not things that were attempted and failed.

## Two things about the evaluation setup that belong in the README, not in a footnote

**The 40 queries were selected partly on corpus term frequency.** Twenty of
sixty candidates were dropped, several on substring counts over the corpus:
"going concern" hit 16, "funded status" 54, "advanced packaging" 172. Those
counts were taken over a 146,449-chunk build of the corpus, before the Item 7
span repairs in `e2572a7`; the corpus now holds 148,097 chunks, so the counts
are indicative rather than exact.

That shapes the evaluation set using the corpus the systems are scored on. It
happened before pooling and before any judgment, and every drop is listed with
its reason in `reports/query_set.md`.

State the reasons accurately, because they are not all the same. Most drops
were for thin material. At least one, q58 "going concern doubt", was recorded
as dropped because a query with almost no relevant material gives a degenerate
per-query nDCG, which is a reason about the metric the systems are scored with
and not only about how much material exists. q48 and q49 are the same shape one
step softer. The decision is still defensible, and low-base-rate queries were
deliberately retained through q59 (284 chunks) so the precision case is still
tested. But a README sentence claiming the reasons were thin-material rather
than metric-driven would be contradicted by the repo's own record.

**Time discipline is scoped to novelty, not to retrieval.** BM25 fits its IDF
over the whole corpus with no time predicate, and the dense index embeds every
chunk regardless of date, and both then score documents from every date against
that. The repo's own invariant 1, read literally, covers it; PLAN.md section 1 scopes
the discipline to "every novelty score" and treats retrieval as static ad hoc
rather than walk-forward. The two documents disagree. The design follows PLAN.
A reader who knows the leakage literature will spot it, so it is better stated
than defended later.

**The sector background is fit on a survivorship-selected peer set, and that
is future information.** The 20 CIKs were chosen in 2026 by requiring
continuous filing under one CIK from 2020 through 2025. Every novelty score
subtracts a sector background fit on those firms, so a filing from 2020 is
scored against a peer set whose membership was decided using information from
2026. The strict `<` time discipline holds inside the fit and does not reach
the question of which firms are in it.

The regional-bank case is the one to state plainly. SVB Financial, Signature
Bank and First Republic are absent because they failed in 2023. The background
against which a surviving bank's 2023 language is judged therefore contains no
bank that did not survive that year, in precisely the period the measure would
most likely be asked about. The semiconductor side excludes Xilinx and Maxim
Integrated as acquired, and Marvell for a 2021 redomicile that orphaned its
earlier CIK.

The direction of the resulting bias is not measured and should not be asserted.
Excluding the failed banks removes distress language from the background, which
raises background surprisal for that language and pushes the contrast down; had
those firms been present, their early-2023 filings would have lowered it and
pushed the contrast up. Which effect dominates is an empirical question nobody
here has answered.

This is documented rather than fixed, and `src/ticker/universe.py` says so:
including firms whose filings stop mid-window breaks the six-years-per-firm
assumption the per-firm models depend on. The point is that any README sentence
asserting time discipline over novelty without qualification is false at the
corpus-construction layer, which is upstream of every guard the code enforces.

## Query set size: 40, not the 60 PLAN specified

PLAN.md section 3 Phase 2 calls for 60 queries. The set is 40. This is a
deliberate de-scope decision, on the stated grounds that 40 keeps
the paired tests adequately powered and saves a day of hand judging. It is not
an accident and it is not a shortfall against a target that was still live.

The ten PLAN section 2 examples are all present. Which 20 candidates were cut
and why is in `reports/query_set.md`, along with the sector and
lexical/semantic breakdown of the surviving 40. Query ids were not renumbered,
so the ids run q01 to q60 with gaps.

40 queries costs precision. Every metric in the results table is a mean over
queries, so its standard error scales as 1 over the square root of the query
count. Dropping from 60 to 40 multiplies every standard error by sqrt(60/40),
which is 1.22: every bootstrap confidence interval in this repo is about 22%
wider than the same interval computed over a 60-query set. Every paired t-test
between two systems loses power by the same factor. A difference that 60
queries would detect at 80% power has to be roughly 22% larger before 40
queries detect it at 80% power. The change in critical value from 59 to 39
degrees of freedom (2.001 to 2.023 at the two-sided 5% level) is small next to
that and is not the reason.

The practical consequence is that a non-significant row in the ablation ladder
is weaker evidence of no difference here than it would be at 60 queries, and
should be read as underpowered rather than as a null.

## Claims in this repo that are not yet backed by a number

Kept as a running list. Every line here is something the README
must not assert until the number exists.

Blocked on the human labeling pass:

- Every cell of the ablation ladder. `reports/results.md` is a stub.
- Whether fusion beats either single arm, and whether the weighted variant
  beats RRF. The machinery runs; the qrels do not exist.
- The intra-annotator kappa.
- Both columns of the Phase 6 two-qrel table, which is the artifact.

Still unverified for the dense index:

- Byte-identical rebuild. Verified for BM25 on the full corpus, verified for
  dense only on subset builds. `--verify-repeat` exists and doubles a 3 h 17 m
  build, which is why it has not been run at full scale.

Measured, and available to assert:

- The dense index is built over all 148,097 chunks, and its `corpus_sha256`
  matches the BM25 manifest, so the two arms index the same corpus.
- The three-source judging pool is 53.6 chunks per query, 2,144 in total. The
  dense arm contributes 16.2 per query that neither other source surfaced.
- Novelty covers all 298,375 sentences across 10-K, 10-Q and 8-K, every row
  with `as_of == filed_at`.
- Validation 5.1: AUC 0.7132, CI [0.7013, 0.7244], join rate 100%. PLAN's kill
  condition of an AUC near 0.5 is not met.
- Validation 5.2 is a partial negative. Risk Factors separates from the Item 1
  Business control with disjoint intervals; MD&A does not, and PLAN predicts it
  should. Items 3 and 7A are too short per section to interpret.

Blocked on Phase 5.3, which needs labels:

- Agreement between the blind spot-check labels and the novelty deciles.

Asserted from construction rather than measurement:

- That the extraction boundaries are right. Zero approved fixtures exist, so
  every extraction rate in `reports/phase1.md` is the extractor grading itself.
  The two short-Item-7 repairs are checked by magnitude and by consistency with
  the same filer's other years, not against an approved span.
- That the sector-matched background is a better control than a corpus-wide one.
  PLAN argues it; nothing here has measured the difference.
- That `bge-base-en-v1.5` beats a finance-adapted model on this corpus, or
  loses to one. That arm is cut, so the question is open, not answered.
