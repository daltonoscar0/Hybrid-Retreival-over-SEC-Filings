"""The per-sentence novelty layer: language models, the contrast, the diff.

`kneser_ney` fits the models, `score` computes the firm-minus-background
contrast and holds the guardrails that keep a z-scored value out of any
cross-document number. Nothing is re-exported here on purpose: a caller that
writes `from ticker.novelty.score import novelty_raw` says which layer it is
reaching into, and the two layers are meant to stay separable.
"""
