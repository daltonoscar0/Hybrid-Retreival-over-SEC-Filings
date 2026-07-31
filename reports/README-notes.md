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

## Query set size: 40, not the 60 PLAN specified

PLAN.md section 3 Phase 2 calls for 60 queries. The set is 40. This is a
deliberate de-scope taken under RUN.md B2, on the stated grounds that 40 keeps
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
