# Phase 0: skeleton and data contract

Built: uv-managed Python 3.12 environment; DuckDB schema for filings, sections,
sentences, and chunks; frozen dataclass record types; the time-filtered data
layer; provisional downloader, sentence splitter, and chunker sufficient for one
end-to-end ingest.

Numbers:

- 1 filing ingested end to end: AAPL 10-K, accession 0000320193-25-000079,
  filed 2025-10-31
- rows: filings 1, sections 1, sentences 834, chunks 416
- tests: 11 passed, 0 failed (tests/test_time_filter.py)

Time discipline is enforced in src/ticker/db.py, not by convention. Every read
that feeds modeling takes an explicit as_of with no default and filters
sentences.filed_at with strict less-than. A filing at exactly as_of is excluded,
covered by a test. Naive datetimes are rejected at the module boundary on both
insert and read.

Open:

- The one stored section is item PROVISIONAL_FULL_TEXT. Real item boundary
  extraction is Phase 1.
- The sentence splitter is a first cut, not yet tuned against the financial
  prose cases (dollar amounts, Inc., U.S., numbered items, tabular fragments).
- background_sentences is corpus-wide minus the target firm. Sector scoping
  arrives with the sector column in Phase 1.

What a reader should doubt: the 834-sentence count came from the provisional
splitter over full document text and will change once item extraction and
splitter tuning land. Nothing downstream consumes these rows yet.
