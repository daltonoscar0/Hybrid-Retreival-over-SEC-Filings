"""Rank and score fusion over run files, plus the held-out weight-tuning split.

Two fusion methods, and why there are two
------------------------------------------
`rrf` combines rankings. `weighted` combines scores. They fail differently,
so the ablation ladder reports both rather than picking one.

Reciprocal rank fusion (Cormack, Clarke, Buettcher, "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods", SIGIR 2009,
758-759) scores a document as the sum over systems of `1 / (k + rank)`, with
`rank` 1-based inside each system's own ranking and a system that never
retrieved the document contributing nothing. It reads ranks only, so it needs
no score normalization, no training data, and no assumption that two systems
put their scores on comparable scales. That is why it is the fusion arm that
runs before a single relevance judgment exists.

`k = 60` is the value Cormack et al. report. They chose it on TREC data. It
is not tuned here and should not be: with 40 queries and no third split
reserved for it, fitting `k` would be fitting a free parameter on the same
queries the result is reported over, which is the exact failure the held-out
machinery below exists to prevent. Using the published constant and saying
where it came from is the honest version. What `k` does is flatten the
reciprocal curve. At `k=60`, rank 1 contributes `1/61 = 0.0164` and rank 10
contributes `1/70 = 0.0143`, a 14 percent gap; at `k=0` rank 1 would
contribute ten times rank 10. So one system's top hit cannot carry the fused
ranking by itself, and agreement across systems decides instead.

Why weighted fusion needs a normalization step that RRF does not
-----------------------------------------------------------------
A BM25 score is an unbounded sum of per-term IDF contributions; on this
corpus a strong match scores in the tens. A dense score from a normalized
bi-encoder is a cosine similarity, bounded by [-1, 1] and in practice packed
into roughly [0.6, 0.9]. Adding the two raw is not fusion. BM25 would decide
every ranking and the dense arm would perturb it by less than the gap between
adjacent BM25 ties. `weighted` therefore min-max normalizes each system's
scores into [0, 1] first, which is what makes a weight of 0.5 mean "half the
say" rather than "half of a number ten times larger than the other one".

The normalization is per query, never pooled across queries. BM25's scale
moves with query length and with how rare the query's terms are, so a
two-word query over rare terms and a six-word query over common ones are not
on the same scale even within one system. A corpus-wide min-max would let the
first query's documents sit near 1.0 and the second's near 0.2, and the
weighted sum would then be reporting which query was asked. Per-query min-max
restricts the comparison to the only one that is meaningful: this document
against the other documents this system returned for this query.

Two properties of min-max to know before reading the numbers. The
worst-scoring document in each per-query list normalizes to exactly 0 and so
contributes nothing, and a document a system never retrieved also contributes
0, so after fusion the two are indistinguishable. Both are standard for score
fusion, and both are reasons to report RRF next to it rather than instead of
it.

Tuning without reporting the tuning-set number
-----------------------------------------------
`weighted` has free parameters, so it has to be tuned, and a tuned system
scored on the queries it was tuned on reports a number that is partly a
measure of its own tuning. This module splits that into pieces that cannot be
collapsed into one call. `split_queries` makes the partition, deterministically
from a seed. `tune_weights` runs `ranx.optimize_fusion` on the train ids and
returns weights and nothing else -- there is no metric anywhere in its return
value, so no caller can print the tuning-set score by mistake.
`held_out_score` scores a fused run and raises if the ids it is handed overlap
the ids that were tuned on. Getting the tuning-set number into a report takes
a deliberate `score_run(..., train_ids, ...)` call and a deliberate decision
to label it a result.

Weights are keyed by system name, not passed positionally. A list of weights
lined up against a list of runs is one reordering away from giving BM25 the
dense arm's weight and reporting the result as a win, and nothing in the types
would catch it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Mapping

from ranx import Qrels, Run, evaluate as ranx_evaluate, optimize_fusion

# query_id -> chunk_id -> score, the in-memory shape of a run file (see
# ticker.evaluation for the on-disk contract).
RunDict = dict[str, dict[str, float]]

DEFAULT_RRF_K = 60
DEFAULT_NORM = "min-max"
DEFAULT_METRIC = "ndcg@10"
DEFAULT_WEIGHT_STEP = 0.1

# Matches ranx's min-max implementation, which floors the denominator at this
# value so a per-query list whose scores are all equal maps to zeros instead of
# dividing by zero. Kept identical so weights tuned by ranx.optimize_fusion
# apply unchanged to a run fused by `weighted` here.
MIN_MAX_EPS = 1e-9


def _sorted_run(run: RunDict) -> RunDict:
    """Query ids ascending, chunks by descending score with chunk_id breaking
    ties, so a fused run serializes byte-identically across runs and across
    machines. Score order is what evaluation reads; the tie break only fixes
    what would otherwise be dict insertion order."""
    return {
        query_id: dict(
            sorted(run[query_id].items(), key=lambda item: (-item[1], item[0]))
        )
        for query_id in sorted(run)
    }


def rrf(runs: Mapping[str, RunDict], k: int = DEFAULT_RRF_K) -> RunDict:
    """Reciprocal rank fusion over `{system_name: run}`.

    See the module docstring for the citation and for why `k` stays at the
    paper's 60. Ranks come from sorting each system's per-query scores
    descending with chunk_id breaking ties, so two systems that return the
    same tied scores in different orders still fuse to the same result.
    """
    if k <= 0:
        raise ValueError(f"rrf k must be positive, got {k}")

    fused: RunDict = {}
    for run in runs.values():
        for query_id, scores in run.items():
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            target = fused.setdefault(query_id, {})
            for rank, (chunk_id, _score) in enumerate(ranked, start=1):
                target[chunk_id] = target.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return _sorted_run(fused)


def normalize_run(run: RunDict, norm: str = DEFAULT_NORM) -> RunDict:
    """Per-query score normalization.

    `norm="min-max"` maps each query's scores into [0, 1]. `norm="none"`
    returns the scores untouched and exists for tests and diagnostics only;
    fusing unnormalized BM25 and cosine scores is the mistake the module
    docstring describes, not an option worth exercising on real runs.
    """
    if norm == "none":
        return {query_id: dict(scores) for query_id, scores in run.items()}
    if norm != "min-max":
        raise ValueError(f"unknown norm {norm!r}, expected 'min-max' or 'none'")

    normalized: RunDict = {}
    for query_id, scores in run.items():
        if not scores:
            normalized[query_id] = {}
            continue
        lo = min(scores.values())
        hi = max(scores.values())
        denominator = max(hi - lo, MIN_MAX_EPS)
        normalized[query_id] = {
            chunk_id: (score - lo) / denominator for chunk_id, score in scores.items()
        }
    return normalized


def weighted(
    runs: Mapping[str, RunDict],
    weights: Mapping[str, float],
    norm: str = DEFAULT_NORM,
) -> RunDict:
    """Weighted score fusion of `{system_name: run}` after per-query `norm`.

    `weights` is keyed by system name and must name every run exactly once;
    a mismatch raises rather than defaulting a missing system to zero, since
    silently dropping an arm from a fusion looks identical in the output to
    the arm being useless.
    """
    if set(weights) != set(runs):
        raise ValueError(
            f"weights must name exactly the fused systems: runs "
            f"{sorted(runs)}, weights {sorted(weights)}"
        )

    fused: RunDict = {}
    for name, run in runs.items():
        weight = float(weights[name])
        for query_id, scores in normalize_run(run, norm).items():
            target = fused.setdefault(query_id, {})
            for chunk_id, score in scores.items():
                target[chunk_id] = target.get(chunk_id, 0.0) + weight * score
    return _sorted_run(fused)


def split_queries(
    query_ids: Iterable[str], frac: float, seed: int
) -> tuple[list[str], list[str]]:
    """Partition query ids into (train, test); `frac` is the train share.

    Sorted before the shuffle, so the split depends on the seed and on the
    set of ids and on nothing else. Passing ids read out of a dict, a JSONL
    file, or a set gives the same partition. Both returned lists are sorted
    for stable reporting; the shuffle decides membership, not presentation.

    Both sides are forced non-empty: a `frac` that would round one side to
    zero gets clamped to a single query instead, because an empty test split
    would make `held_out_score` raise a confusing error far from the cause.
    """
    if not 0.0 < frac < 1.0:
        raise ValueError(f"frac must be strictly between 0 and 1, got {frac}")

    ids = sorted(set(query_ids))
    if len(ids) < 2:
        raise ValueError(f"need at least 2 query ids to split, got {len(ids)}")

    n_train = min(max(round(frac * len(ids)), 1), len(ids) - 1)
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:n_train]), sorted(shuffled[n_train:])


def usable_query_ids(
    qrels: Mapping[str, Mapping[str, int]],
    runs: Mapping[str, RunDict],
    query_ids: Iterable[str],
) -> list[str]:
    """The subset of `query_ids` that tuning can actually use: judged, and
    retrieved by every run.

    `ranx.optimize_fusion` asserts that every run's query ids match the
    qrels' exactly, so the caller has to intersect first. A query with no
    judgments contributes no signal to the tuning objective anyway, and a
    query one arm returned nothing for would make that arm's weight
    unidentifiable on that query.
    """
    return [
        query_id
        for query_id in sorted(set(query_ids))
        if qrels.get(query_id) and all(run.get(query_id) for run in runs.values())
    ]


def tune_weights(
    qrels: Mapping[str, Mapping[str, int]],
    runs: Mapping[str, RunDict],
    train_ids: Iterable[str],
    metric: str = DEFAULT_METRIC,
    *,
    norm: str = DEFAULT_NORM,
    step: float = DEFAULT_WEIGHT_STEP,
) -> dict[str, float]:
    """Grid-search fusion weights on the train split. Returns weights only.

    Wraps `ranx.optimize_fusion(method="wsum")`, which sweeps every weight
    vector on a `step` grid that sums to 1 and keeps the one maximizing
    `metric`. The returned dict is keyed by system name.

    No score is returned, deliberately. The best `metric` value found during
    the sweep is the tuning-set number, and it is the number a reader would
    mistake for a result. Score the returned weights with `held_out_score` on
    ids this function never saw.
    """
    if len(runs) < 2:
        raise ValueError(f"weighted fusion needs 2+ runs, got {sorted(runs)}")

    ids = usable_query_ids(qrels, runs, train_ids)
    if not ids:
        raise ValueError(
            "no train query is both judged and retrieved by every run, so "
            "there is nothing to tune on"
        )

    names = sorted(runs)
    ranx_qrels = Qrels.from_dict({query_id: dict(qrels[query_id]) for query_id in ids})
    ranx_runs = [
        Run.from_dict({query_id: runs[name][query_id] for query_id in ids}, name=name)
        for name in names
    ]
    best = optimize_fusion(
        ranx_qrels,
        ranx_runs,
        norm=norm,
        method="wsum",
        metric=metric,
        step=step,
        show_progress=False,
    )
    return {name: float(weight) for name, weight in zip(names, best["weights"])}


def score_run(
    qrels: Mapping[str, Mapping[str, int]],
    run: RunDict,
    query_ids: Iterable[str],
    metric: str = DEFAULT_METRIC,
) -> float:
    """Mean `metric` for `run` over the judged subset of `query_ids`.

    Scores whatever ids it is given, including tuning ids. `held_out_score`
    is the one to call for a number that goes in a report.
    """
    ids = [query_id for query_id in sorted(set(query_ids)) if qrels.get(query_id)]
    if not ids:
        raise ValueError("none of the given query ids has any judgment")

    ranx_qrels = Qrels.from_dict({query_id: dict(qrels[query_id]) for query_id in ids})
    retrieved = {query_id: run[query_id] for query_id in ids if run.get(query_id)}
    if not retrieved:
        return 0.0

    # make_comparable zero-fills the judged queries this run returned nothing
    # for, which is the correct score for them and matches ticker.evaluation.
    return float(ranx_evaluate(ranx_qrels, Run.from_dict(retrieved), metric,
                               make_comparable=True))


def held_out_score(
    qrels: Mapping[str, Mapping[str, int]],
    run: RunDict,
    train_ids: Iterable[str],
    test_ids: Iterable[str],
    metric: str = DEFAULT_METRIC,
) -> float:
    """Score `run` on `test_ids`, refusing any overlap with `train_ids`.

    Takes both splits so the disjointness is checked at the point the number
    is produced, not left to the caller having remembered. This is the only
    scoring entry point whose output is meant to be reported for a tuned
    system.
    """
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise ValueError(
            f"{len(overlap)} query id(s) appear in both the tuning and the "
            f"scoring split, e.g. {sorted(overlap)[:5]}. A tuned system "
            "scored on its own tuning queries is not a held-out number."
        )
    return score_run(qrels, run, test_ids, metric)


def write_run_jsonl(path: Path, run: RunDict) -> int:
    """Serialize a run to the `ticker.evaluation` contract; returns row count.

    The writer half of `ticker.evaluation.load_run_jsonl`, which predates any
    code that produces a run. It lives here rather than being copied into
    `scripts/search.py` and `scripts/fuse.py` so the two producers cannot
    drift apart on field names. Rows come out in `_sorted_run` order, so
    re-running a deterministic retriever overwrites the file with identical
    bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with path.open("w") as f:
        for query_id, scores in _sorted_run(run).items():
            for chunk_id, score in scores.items():
                f.write(
                    json.dumps(
                        {
                            "query_id": query_id,
                            "chunk_id": chunk_id,
                            "score": float(score),
                        }
                    )
                    + "\n"
                )
                rows += 1
    return rows
