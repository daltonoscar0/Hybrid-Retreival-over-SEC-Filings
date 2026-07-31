#!/usr/bin/env python3
"""Run the Phase 1 download sweep: every 10-K, 10-Q, and EX-99.1-bearing 8-K
filed 2020-01-01 through 2025-12-31 for the 20 companies in ticker.universe.

On-disk cache keyed by accession under --cache-dir (default data/raw), so
killing this and re-running only pays for accessions not yet resolved. Rate
limiting is edgartools' own global limiter; see ticker.download's module
docstring.

Usage: .venv/bin/python3 scripts/download_corpus.py
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

from ticker.download import DEFAULT_CACHE_DIR, sweep_company
from ticker.universe import UNIVERSE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--company", action="append", default=None,
        help="restrict to this ticker (repeatable); default is the full universe",
    )
    args = parser.parse_args()

    companies = UNIVERSE
    if args.company:
        wanted = {t.upper() for t in args.company}
        companies = tuple(c for c in UNIVERSE if c.ticker in wanted)

    by_form_sector: Counter[tuple[str, str]] = Counter()
    status_counts: Counter[str] = Counter()
    failures: list[tuple[str, str, str]] = []

    start = time.monotonic()
    for company in companies:
        company_downloaded = 0
        for result in sweep_company(company, cache_dir=args.cache_dir):
            status_counts[result.status] += 1
            if result.status in ("cached", "downloaded"):
                by_form_sector[(result.form, company.sector)] += 1
                company_downloaded += 1
            if result.status == "failed":
                failures.append((company.ticker, result.accession, result.reason or ""))
        print(f"{company.ticker:6s} ({company.sector}): {company_downloaded} filings resolved")

    elapsed = time.monotonic() - start
    print(f"\ndone in {elapsed:.1f}s")
    print(f"status counts: {dict(status_counts)}")
    print("\nby form / sector:")
    for (form, sector), count in sorted(by_form_sector.items()):
        print(f"  {form:6s} {sector:16s} {count}")
    print(f"\ntotal filings cached: {sum(by_form_sector.values())}")

    if failures:
        print(f"\n{len(failures)} failures:")
        for ticker, accession, reason in failures:
            print(f"  {ticker} {accession}: {reason}")


if __name__ == "__main__":
    main()
