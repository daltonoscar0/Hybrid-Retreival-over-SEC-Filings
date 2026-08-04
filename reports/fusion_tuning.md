# Fusion

| system | queries | rows |
|---|---|---|
| bm25 | 40 | 4000 |
| dense | 40 | 4000 |

## RRF

`k = 60`, the value reported in Cormack, Clarke, Buettcher (SIGIR 2009). Chosen there on TREC data and not tuned on this corpus: with no third split reserved for it, fitting `k` would fit a free parameter on the same queries the result is reported over.

Fused 2 systems into `data/runs/rrf.jsonl` (6664 rows). RRF reads ranks only, so it needs no judgments and this arm is final.

## Weighted fusion

Skipped. `data/qrels/standard.jsonl` has no judgments yet, and weighted fusion learns its weights from them. No weights, no tuning-set number, and no held-out number are reported, because there is nothing to compute them from. Judge with `scripts/judge.py` and re-run this command.
