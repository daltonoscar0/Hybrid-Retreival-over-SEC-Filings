#!/usr/bin/env python3
"""Score every sentence in a scope of filings with the novelty contrast.

Writes JSONL, one line per sentence:

    {"sentence_id", "novelty", "section_id", "accession", "item",
     "cik", "ticker", "sector", "form", "filed_at", "as_of"}

`novelty` is the RAW contrast in bits per token, firm minus sector
background, from `ticker.novelty.score.novelty_raw`. It is never the
within-document z-score. Invariant 3 says the z-score is for colouring one
document's sentences and nothing else, and a file on disk is exactly the
route by which a display value ends up averaged across documents in someone
else's script six weeks later. `scripts/spotcheck.py` and the Phase 5
validations read this file and all of them want the raw contrast.

Cost, and the as_of grid
-------------------------
The expensive term is the background model: every sector peer's sentences
filed before the cutoff, refit from scratch. Fitting one per filing means one
fit per distinct filing date, which for 966 filings is 966 fits over corpora
that reach six figures of sentences.

`--as-of` chooses the grid the cutoff is snapped to. The default is `filing`,
the exact cutoff.

  filing   cutoff is the filing's own `filed_at`. Exact, and the most history
           the time discipline permits.
  quarter  cutoff is the first instant of the quarter containing `filed_at`.
           Collapses the 966 distinct cutoffs to at most 24, so the two sector
           background models are fit at most 48 times instead of 966.

`quarter` does not leak. The training set it produces is a prefix of what
`filing` would have allowed, never a superset, and no sentence can be both
trained on and scored: `fit` rejects a sentence at or after `as_of`,
`novelty_raw` rejects one before it, and the two predicates are complementary.

It is still the wrong default, for a reason that took an audit to see. The bias
it introduces runs toward FALSE NOVELTY, not toward an understated contrast.
Take a firm that files its earnings 8-K on January 20 and the 10-K whose MD&A
recycles that language on February 10. Both snap to `as_of` January 1, so the
firm model scoring the 10-K has not seen the 8-K, even though the 8-K predates
it. The recycled text is unseen by the firm model, so firm surprisal is high;
the sector background has not seen it either, so subtracting does not cancel
it; the contrast reads new for text the firm published three weeks earlier.
Under the exact cutoff the firm model would have seen it and scored it strongly
negative.

That is not a rounding error on a minority of the corpus. 486 of 966 filings
are second or later within their own (firm, calendar quarter), covering 251,636
of 298,375 sentences, and the earnings-release-then-periodic-report pair is the
dominant shape. The failure lands precisely on the recycled boilerplate the
measure exists to detect.

`quarter` stays available because the background fit is the expensive term and
a corpus-wide pass over all three forms is hours of it. Anything feeding a
Phase 5 validation should use the default.

The firm model is cheap either way (one firm's own history) and is fit at the
same cutoff as its background, because `ticker.novelty.score` refuses a pair
whose two `as_of` values differ: a contrast across two time horizons measures
the horizon.

Usage:
  uv run python scripts/score_novelty.py --form 10-K
  uv run python scripts/score_novelty.py --as-of quarter
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from ticker.novelty.score import build_models, novelty_raw
from ticker.records import Sentence

DEFAULT_DB = Path("data/ticker.duckdb")
DEFAULT_OUT = Path("data/novelty/scores.jsonl")

# Ordered by firm first, then time, so every filing that shares a model pair
# is consecutive and exactly one pair has to be held at a time. Ordering by
# filed_at instead, the natural reading order, interleaves all 20 firms and
# forces either a rebuild per filing or a cache holding every pair built so
# far. The second is what the first version of this script did, and 51 filings
# in, holding 51 five-gram models over six-figure sentence counts, the OS
# killed it. Memory is the binding constraint here, not time.
_FILINGS_SQL = """
    SELECT accession, cik, ticker, sector, form, filed_at
    FROM filings
    WHERE (? IS NULL OR form = ?)
    ORDER BY cik, filed_at, accession
"""

_SENTENCES_SQL = """
    SELECT s.sentence_id, s.section_id, s.ordinal, s.text, s.filed_at, sec.item
    FROM sentences s
    JOIN sections sec ON s.section_id = sec.section_id
    WHERE sec.accession = ?
    ORDER BY sec.item, s.ordinal
"""


def quarter_floor(moment: datetime) -> datetime:
    """First instant of the quarter containing `moment`, in UTC.

    Converted to UTC first. DuckDB hands back a TIMESTAMPTZ in the session
    timezone, so reading `.year` and `.month` off it and then stamping the
    result `timezone.utc` mixes two calendars. West of UTC the result is still
    earlier than `moment` and nothing shows; east of UTC a filing late on the
    last day of a quarter renders as the first day of the next one, the floor
    lands after the filing, and the filing's own sentences fall into its own
    firm model. Which sentences get scored would then depend on the `TZ` of the
    process, which is not a property anyone would think to check.
    """
    if moment.tzinfo is None:
        # `.astimezone` on a naive datetime does not raise, it assumes system
        # local time, which would reintroduce the bug this function just
        # documented by a different route.
        raise ValueError(f"quarter_floor needs an aware datetime, got naive {moment!r}")
    moment = moment.astimezone(timezone.utc)
    quarter_start_month = 3 * ((moment.month - 1) // 3) + 1
    return datetime(moment.year, quarter_start_month, 1, tzinfo=timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--form", default=None, help="restrict to one form, e.g. 10-K")
    parser.add_argument("--as-of", choices=("filing", "quarter"), default="filing")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--order", type=int, default=5, help="n-gram order; PLAN fixes this at 5"
    )
    args = parser.parse_args()

    con = duckdb.connect(str(args.db), read_only=True)
    try:
        filings = con.execute(_FILINGS_SQL, [args.form, args.form]).fetchall()
        if args.limit is not None:
            filings = filings[: args.limit]
        if not filings:
            print(f"no filings match --form {args.form!r} in {args.db}")
            return

        args.out.parent.mkdir(parents=True, exist_ok=True)
        current_key: tuple[int, str, datetime] | None = None
        firm_lm = background_lm = None
        n_pairs = 0
        n_sentences = 0
        n_skipped = 0
        started = time.monotonic()

        with args.out.open("w") as out:
            for i, (accession, cik, ticker, sector, form, filed_at) in enumerate(filings, 1):
                as_of = quarter_floor(filed_at) if args.as_of == "quarter" else filed_at

                key = (cik, sector, as_of)
                if key != current_key:
                    # Rebind before building, so the previous pair is
                    # collectable while the next one allocates rather than
                    # after. Two of these alive at once is what the run cannot
                    # afford.
                    firm_lm = background_lm = None
                    firm_lm, background_lm = build_models(
                        con, cik=cik, sector=sector, as_of=as_of, order=args.order
                    )
                    current_key = key
                    n_pairs += 1

                rows = con.execute(_SENTENCES_SQL, [accession]).fetchall()
                for sentence_id, section_id, ordinal, text, sentence_filed_at, item in rows:
                    sentence = Sentence(
                        sentence_id=sentence_id,
                        section_id=section_id,
                        ordinal=ordinal,
                        text=text,
                        filed_at=sentence_filed_at,
                    )
                    # A sentence filed before the cutoff is in the models' own
                    # training data. Under --as-of quarter that is every
                    # sentence of a filing made earlier in the same quarter as
                    # another of the same firm's filings. Skipped and counted,
                    # never scored: the model recognising itself is not novelty.
                    if sentence.filed_at < as_of:
                        n_skipped += 1
                        continue
                    out.write(
                        json.dumps(
                            {
                                "sentence_id": sentence_id,
                                "novelty": novelty_raw(sentence, firm_lm, background_lm),
                                "section_id": section_id,
                                "accession": accession,
                                "item": item,
                                "cik": cik,
                                "ticker": ticker,
                                "sector": sector,
                                "form": form,
                                "filed_at": filed_at.isoformat(),
                                "as_of": as_of.isoformat(),
                            }
                        )
                        + "\n"
                    )
                    n_sentences += 1

                elapsed = time.monotonic() - started
                print(
                    f"[{i}/{len(filings)}] {ticker} {form} {accession} "
                    f"{len(rows)} sentences  {n_sentences} scored  "
                    f"{n_pairs} model pairs built  {elapsed:.0f}s",
                    flush=True,
                )
    finally:
        con.close()

    elapsed = time.monotonic() - started
    print(
        f"\nwrote {n_sentences} scores -> {args.out} in {elapsed:.0f}s "
        f"({n_pairs} model pairs built)"
    )
    if n_skipped:
        print(
            f"{n_skipped} sentences were filed before their own cutoff and were "
            f"skipped rather than scored (see --as-of {args.as_of} above)"
        )


if __name__ == "__main__":
    main()
