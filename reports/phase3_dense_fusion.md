# Phase 3: dense retrieval and fusion

## What was built

`src/ticker/retrieval/dense.py` wraps `BAAI/bge-base-en-v1.5` and a FAISS flat
index behind the same `ticker.retrieval_types.Retriever` interface BM25 uses, so
`scripts/search.py`, `scripts/pool.py` and `scripts/evaluate.py` need no dense
specific code. `scripts/index_dense.py` builds and saves the index with a
manifest. `src/ticker/fusion.py` holds RRF, weighted fusion, and the held-out
tuning split; `scripts/fuse.py` is its CLI. Tests are in `tests/test_dense.py`
(15) and `tests/test_fusion.py` (32).

Per RUN.md the finance-adapted second embedding arm is cut. There is one dense
model and it is a constant in the module, not a parameter.

## State of the dense index at the time of writing

The index is not built. The build is running and needs about five hours on this
machine. Everything downstream of it is therefore also outstanding:
`data/runs/dense.jsonl`, the RRF and weighted fusion run files, the three-source
judging pool, and every row of the ablation ladder.

This is a throughput limit, not a defect, and the number behind it is measured
rather than estimated. See below.

## Two problems found while building it, both real

### A two-OpenMP-runtime segfault, at scale only

The first three full-corpus builds died the same way: a `SIGSEGV` inside
`libomp.dylib`, in `__kmp_fork_barrier`, partway through the encode. It
reproduced on `mps` and on `cpu`, and with `KMP_DUPLICATE_LIB_OK=TRUE` set. One
run instead deadlocked in `MPSStream::copy_and_sync`, which is the same conflict
with a different symptom.

`faiss-cpu` and `torch` each ship their own OpenMP runtime. `build_index`
imported both at the top of the function, so both were live during the encode,
and the thousands of OpenMP fork/join cycles across 2,315 batches eventually hit
the race. Moving `import faiss` to after the encode fixes it: torch's runtime is
then the only one live during the encode, and faiss afterwards does a flat copy
and a file write, neither of which needs threads. `faiss.omp_set_num_threads(1)`
is set as well.

Worth recording because of how it hid. `tests/test_dense.py` builds real indexes
with the real model and never crashed: at a few hundred vectors the race does not
get enough attempts. The test suite was green while the thing it tests could not
run on the corpus. A test that exercises the code path is not a test that
exercises the workload.

### Encoding throughput is the binding constraint

Measured on this machine (MacBook Air, 10 cores, 16 GB, fanless), over 256
chunks drawn from the corpus, mean chunk length 1,119 characters:

| configuration | chunks/s |
|---|---|
| mps, fp32, max_seq_length 512 | 7.9 |
| mps, fp32, 512, deterministic algorithms on | 6.8 |
| mps, fp16, 512 | 10.2 |
| mps, fp16, max_seq_length 256 | 22.5 |
| cpu, fp32, 512 | 4.7 |

At the shipped configuration, fp32 and the model's own 512-token limit, 148,097
chunks is about 5.2 hours.

That is not a misconfiguration. `bge-base-en-v1.5` is 110M parameters, so a
512-token forward pass is roughly 113 GFLOP, and 7.9 sequences per second is
about 890 GFLOPS, near 30% of this GPU's fp32 peak. The corpus is simply large
for the hardware.

The shipped configuration is the slowest of the five. fp16 would cut it to 4.1
hours and `max_seq_length 256` to 1.8, and neither was taken. Halving the
sequence limit would silently drop the tail of every chunk over 256 tokens,
which at a mean of 1,119 characters is most of them, and hiding a quality change
inside a speed decision is the thing this repo is organised against. fp16 was
left off because the manifest claims fp32 and the difference bought about an
hour.

## The pool with two of its three sources

Reported here rather than in the Phase 2 report because it is what the pool
looks like with the dense source missing, which is the state the unbuilt index
leaves it in.

| pool | min | max | mean | total judgments |
|---|---|---|---|---|
| keyword seed only | 20 | 20 | 20.0 | 800 |
| keyword seed plus BM25 | 25 | 40 | 37.4 | 1,497 |

The keyword-only pool hit its `keyword-k=20` cap on every one of the 40 queries,
so its "mean 20.0" is the cap and not a measurement. Adding BM25 nearly doubles
the pool, which says the two sources disagree about what is worth looking at on
almost every query. That disagreement is the argument for pooling and it is the
reason a keyword-only pool should not be judged.

The three-source number is not known and will not be until the dense index
exists. Judging the two-source pool now would fix a ceiling on measurable recall
that excludes, by construction, exactly the chunks the dense arm was added to
find.

`scripts/pool.py` and `scripts/judge.py` opened the corpus read-write, which in
DuckDB takes an exclusive file lock, so either one locked every other process
out. Both now use `db.connect_readonly`. This is not hypothetical tidiness: the
Phase 4 scoring pass runs for tens of minutes against the same file, and under
the old code it would have blocked a judging session outright.

## Fusion

`rrf(runs, k=60)` implements Cormack, Clarke and Buettcher (SIGIR 2009). `k` is
their published value and is deliberately not tuned: with 40 queries and no
third split reserved for it, fitting `k` would be fitting a free parameter on the
queries the result is reported over.

`weighted` min-max normalizes each system's scores per query before summing.
Per query, not pooled across queries: BM25's scale moves with query length and
term rarity, so a corpus-wide normalization would leave the fused score partly
reporting which query was asked.

The tuning discipline is enforced by the shape of the API rather than by care.
`tune_weights` returns weights and no score, so the tuning-set number cannot be
reported by accident. `held_out_score` takes both splits and raises on any
overlap. `scripts/fuse.py` writes the tuning-set number and the held-out number
side by side in `reports/fusion_tuning.md`, because the gap between them is the
only way a reader can tell whether the weights overfit.

Run today, `scripts/fuse.py` reports that it has one run file and needs two, and
writes nothing. Once the dense run file exists it will produce RRF immediately;
the weighted arm additionally waits on judgments, and the script says so and
exits 0 rather than inventing a qrels file.

## The ablation ladder

Empty. It needs relevance judgments and there are none. `scripts/evaluate.py`
writes a stub table saying exactly that.

This is the expected state at this point in the build and it is the reason the
harness was written before the retrievers: the table exists, the command that
fills it exists, and no number in it can be produced by anything other than a
human's labels.

## What a reader should be skeptical of

`tests/test_dense.py::test_real_model_matches_a_paraphrase_bm25_would_miss` fails.
It seeds three short chunks and asserts the model ranks "The Company recorded a
non-cash write-down of intangible assets" first for the query "goodwill
impairment charge"; it ranks "The board authorized an additional share
repurchase program" first instead. This has not been diagnosed. It is either a
badly chosen toy, three unrelated one-sentence documents and an abstract query
being a weak test of an embedding model, or a real defect in the query prefix
policy. It is left failing rather than deleted or loosened, because a test that
is quietly relaxed after it fails is worse than no test.

Byte-identical rebuild has not been checked for the dense index. `--verify-repeat`
exists and re-encodes into a scratch directory, and running it doubles a
five-hour build, so it has only been run on subset builds.

The retriever's query path imports faiss before it loads the model, the same
order that crashed the build. Forty queries is a few hundred OpenMP cycles
rather than tens of thousands, so it has not been observed to fail, and it has
not been fixed either.
