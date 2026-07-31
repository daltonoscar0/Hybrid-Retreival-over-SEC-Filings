"""EDGAR access via edgartools: identity, zero-padded CIKs, on-disk cache.

CIKs must be zero-padded to 10 digits or EDGAR's HTTP endpoints return a 500;
`zero_pad_cik` exists so nothing calls the API with a bare int. The 20
company / 6 year / 2 sector corpus build, rate limiting across hundreds of
filings, and `locationCodes` full-text-search filtering are Phase 1's job --
this module only pulls one filing at a time, cached by accession.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import edgar

DEFAULT_CACHE_DIR = Path("data/raw")


def zero_pad_cik(cik: int | str) -> str:
    return f"{int(cik):010d}"


def ensure_identity() -> None:
    identity = os.environ.get("EDGAR_IDENTITY")
    if not identity:
        raise RuntimeError(
            "EDGAR_IDENTITY is not set. EDGAR requires a real contact string "
            "in the User-Agent header and will reject requests without one."
        )
    edgar.set_identity(identity)


@dataclass(frozen=True, slots=True)
class DownloadedFiling:
    accession: str
    cik: int
    ticker: str
    form: str
    filed_at: datetime
    period_end: date | None
    url: str
    text: str


def _cache_paths(accession: str, cache_dir: Path) -> tuple[Path, Path]:
    safe = accession.replace("/", "-")
    return cache_dir / f"{safe}.json", cache_dir / f"{safe}.txt"


def _from_cache(meta_path: Path, text_path: Path) -> DownloadedFiling:
    meta = json.loads(meta_path.read_text())
    return DownloadedFiling(
        accession=meta["accession"],
        cik=meta["cik"],
        ticker=meta["ticker"],
        form=meta["form"],
        filed_at=datetime.fromisoformat(meta["filed_at"]),
        period_end=date.fromisoformat(meta["period_end"]) if meta["period_end"] else None,
        url=meta["url"],
        text=text_path.read_text(),
    )


def _to_cache(meta_path: Path, text_path: Path, filing: DownloadedFiling) -> None:
    meta_path.write_text(json.dumps({
        "accession": filing.accession,
        "cik": filing.cik,
        "ticker": filing.ticker,
        "form": filing.form,
        "filed_at": filing.filed_at.isoformat(),
        "period_end": filing.period_end.isoformat() if filing.period_end else None,
        "url": filing.url,
    }))
    text_path.write_text(filing.text)


def fetch_latest_filing(
    ticker: str, form: str, cache_dir: Path = DEFAULT_CACHE_DIR
) -> DownloadedFiling:
    """Fetch the most recent `form` filing for `ticker`, on-disk cached by accession."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    ensure_identity()

    company = edgar.Company(ticker)
    filing = company.get_filings(form=form).latest()

    meta_path, text_path = _cache_paths(filing.accession_no, cache_dir)
    if meta_path.exists() and text_path.exists():
        return _from_cache(meta_path, text_path)

    filed_at = filing.acceptance_datetime
    if filed_at.tzinfo is None:
        filed_at = filed_at.replace(tzinfo=timezone.utc)

    period_end = filing.period_of_report
    if isinstance(period_end, str) and period_end:
        period_end = date.fromisoformat(period_end)
    elif not period_end:
        period_end = None

    result = DownloadedFiling(
        accession=filing.accession_no,
        cik=int(filing.cik),
        ticker=ticker.upper(),
        form=filing.form,
        filed_at=filed_at,
        period_end=period_end,
        url=filing.filing_url,
        text=filing.text(),
    )
    _to_cache(meta_path, text_path, result)
    return result
