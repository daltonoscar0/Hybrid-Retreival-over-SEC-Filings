"""Retrievers that satisfy `ticker.retrieval_types.Retriever`.

One module per retrieval system (`bm25`, later `dense`), each with a
`load_retriever()` factory taking no arguments so `scripts/pool.py`'s
dynamic `module.path:callable` import can wire it with no repo-specific
glue code.
"""
