"""EDGAR access via edgartools: identity, zero-padded CIKs, on-disk cache.

CIKs must be zero-padded to 10 digits or EDGAR's HTTP endpoints return a 500.
edgartools zero-pads internally for the calls it makes on this module's
behalf, so nothing here relies on `zero_pad_cik` today; it is kept, tested,
and documented because PLAN.md calls this gotcha out explicitly and any
future code in this module that talks to a raw EDGAR endpoint (full-text
search, `data.sec.gov`) directly needs it. Rate limiting is edgartools' own
token-bucket limiter (`EDGAR_RATE_LIMIT_PER_SEC`, default 9 req/sec), applied
globally to every HTTP call this module makes; nothing here adds a second
limiter on top of it.

The on-disk cache is keyed by accession: `{accession}.json` (metadata) and
`{accession}.txt` (extracted text) under `DEFAULT_CACHE_DIR`. A 8-K accession
confirmed to carry no EX-99.1 exhibit is marked with a `{accession}.skip`
sentinel so a re-run does not re-fetch its attachment list over the network.
Both make the sweep resumable: killing it mid-run and re-running only pays
for accessions not yet resolved.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import edgar

from ticker.universe import Company, END_DATE, START_DATE

DEFAULT_CACHE_DIR = Path("data/raw")

FORM_10K = "10-K"
FORM_10Q = "10-Q"
FORM_8K = "8-K"


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
    sector: str
    form: str
    filed_at: datetime
    period_end: date | None
    url: str
    text: str


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Outcome of resolving one accession: cached, freshly downloaded,
    skipped (8-K with no EX-99.1), or failed. `filing` is set only for
    "cached" and "downloaded".
    """

    accession: str
    form: str
    status: str  # "cached" | "downloaded" | "skipped_no_exhibit" | "failed"
    filing: DownloadedFiling | None
    reason: str | None


_REQUIRED_META_KEYS = (
    "accession", "cik", "ticker", "sector", "form", "filed_at", "period_end", "url",
)


def _cache_paths(accession: str, cache_dir: Path) -> tuple[Path, Path]:
    safe = accession.replace("/", "-")
    return cache_dir / f"{safe}.json", cache_dir / f"{safe}.txt"


def _skip_path(accession: str, cache_dir: Path) -> Path:
    safe = accession.replace("/", "-")
    return cache_dir / f"{safe}.skip"


def _is_complete_cache_entry(meta_path: Path, text_path: Path) -> bool:
    """True only for a `{accession}.json`/`.txt` pair this module itself
    wrote. `data/raw` is a cache shared with other tools in this repo (e.g.
    the section-extraction fixture script) that key their own, differently
    shaped metadata by the same accession -- a same-named `.json` here is
    not necessarily one of this module's complete records, and treating any
    such file as a cache hit crashes `_from_cache` on the first missing key.
    """
    if not (meta_path.exists() and text_path.exists()):
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return False
    return all(key in meta for key in _REQUIRED_META_KEYS)


def _from_cache(meta_path: Path, text_path: Path) -> DownloadedFiling:
    meta = json.loads(meta_path.read_text())
    return DownloadedFiling(
        accession=meta["accession"],
        cik=meta["cik"],
        ticker=meta["ticker"],
        sector=meta["sector"],
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
        "sector": filing.sector,
        "form": filing.form,
        "filed_at": filing.filed_at.isoformat(),
        "period_end": filing.period_end.isoformat() if filing.period_end else None,
        "url": filing.url,
    }))
    text_path.write_text(filing.text)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _period_end(value) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return date.fromisoformat(value) if value else None
    return value or None


def fetch_latest_filing(
    ticker: str, form: str, sector: str, cache_dir: Path = DEFAULT_CACHE_DIR
) -> DownloadedFiling:
    """Fetch the most recent `form` filing for `ticker`, on-disk cached by accession."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    ensure_identity()

    company = edgar.Company(ticker)
    filing = company.get_filings(form=form).latest()

    meta_path, text_path = _cache_paths(filing.accession_no, cache_dir)
    if _is_complete_cache_entry(meta_path, text_path):
        return _from_cache(meta_path, text_path)

    result = DownloadedFiling(
        accession=filing.accession_no,
        cik=int(filing.cik),
        ticker=ticker.upper(),
        sector=sector,
        form=filing.form,
        filed_at=_aware(filing.acceptance_datetime),
        period_end=_period_end(filing.period_of_report),
        url=filing.filing_url,
        text=filing.text(),
    )
    _to_cache(meta_path, text_path, result)
    return result


def _is_earnings_release_8k(entity_filing) -> bool:
    """Item 2.02 (Results of Operations and Financial Condition) is the SEC
    item code for furnishing an earnings press release. Plenty of 8-Ks carry
    an EX-99.1 that is not an earnings release -- debt pricing, M&A closing,
    executive changes all commonly use EX-99.1 under Item 8.01/5.02/2.01 --
    so the item code, not exhibit presence alone, is what scopes this to
    "earnings release" per PLAN.md section 2.
    """
    items = getattr(entity_filing, "items", None) or ""
    return "2.02" in [code.strip() for code in items.split(",")]


def _find_ex99(filing) -> object | None:
    """The EX-99.x exhibit that is the earnings release.

    Most Item 2.02 8-Ks carry exactly one EX-99.x. A minority bundle a short
    EX-99.1 (e.g. a same-day dividend announcement) alongside the actual
    earnings release filed as EX-99.2 or higher -- `.size` metadata is not
    reliably populated by edgartools for these, so disambiguating requires
    comparing extracted text length, which only costs extra requests for
    that minority case.
    """
    candidates = filing.attachments.query("re.match('EX-99', document_type)", False).documents
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    scored = [(doc, doc.text() or "") for doc in candidates]
    return max(scored, key=lambda pair: len(pair[1]))[0]


def _download_primary(entity_filing, company: Company, cache_dir: Path) -> DownloadResult:
    accession = entity_filing.accession_no
    meta_path, text_path = _cache_paths(accession, cache_dir)
    if _is_complete_cache_entry(meta_path, text_path):
        return DownloadResult(accession, entity_filing.form, "cached", _from_cache(meta_path, text_path), None)
    try:
        text = entity_filing.text()
        if not text or not text.strip():
            return DownloadResult(accession, entity_filing.form, "failed", None, "empty primary document text")
        result = DownloadedFiling(
            accession=accession,
            cik=company.cik,
            ticker=company.ticker,
            sector=company.sector,
            form=entity_filing.form,
            filed_at=_aware(entity_filing.acceptance_datetime),
            period_end=_period_end(entity_filing.period_of_report),
            url=entity_filing.filing_url,
            text=text,
        )
        _to_cache(meta_path, text_path, result)
        return DownloadResult(accession, entity_filing.form, "downloaded", result, None)
    except Exception as exc:  # noqa: BLE001 -- one bad filing must not abort the sweep
        return DownloadResult(accession, entity_filing.form, "failed", None, repr(exc))


def _download_ex99(entity_filing, company: Company, cache_dir: Path) -> DownloadResult:
    accession = entity_filing.accession_no
    meta_path, text_path = _cache_paths(accession, cache_dir)
    if _is_complete_cache_entry(meta_path, text_path):
        return DownloadResult(accession, entity_filing.form, "cached", _from_cache(meta_path, text_path), None)
    skip_path = _skip_path(accession, cache_dir)
    if skip_path.exists():
        return DownloadResult(accession, entity_filing.form, "skipped_no_exhibit", None, skip_path.read_text())
    try:
        exhibit = _find_ex99(entity_filing)
        if exhibit is None:
            skip_path.write_text("no EX-99.1/EX-99.01/EX-99 exhibit")
            return DownloadResult(accession, entity_filing.form, "skipped_no_exhibit", None, "no EX-99.1 exhibit")
        text = exhibit.text()
        if not text or not text.strip():
            return DownloadResult(accession, entity_filing.form, "failed", None, "empty EX-99.1 text")
        result = DownloadedFiling(
            accession=accession,
            cik=company.cik,
            ticker=company.ticker,
            sector=company.sector,
            form=entity_filing.form,
            filed_at=_aware(entity_filing.acceptance_datetime),
            period_end=_period_end(entity_filing.period_of_report),
            url=entity_filing.filing_url,
            text=text,
        )
        _to_cache(meta_path, text_path, result)
        return DownloadResult(accession, entity_filing.form, "downloaded", result, None)
    except Exception as exc:  # noqa: BLE001 -- one bad filing must not abort the sweep
        return DownloadResult(accession, entity_filing.form, "failed", None, repr(exc))


def sweep_company(
    company: Company,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    start: str = START_DATE,
    end: str = END_DATE,
) -> Iterator[DownloadResult]:
    """Every 10-K, 10-Q, and EX-99.1-bearing 8-K for `company` filed in
    [start, end], yielded as they resolve. Safe to interrupt and re-run.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    ensure_identity()

    entity = edgar.Company(company.cik)
    date_range = f"{start}:{end}"

    for form in (FORM_10K, FORM_10Q):
        filings = entity.get_filings(form=form, date=date_range, amendments=False)
        for f in filings:
            yield _download_primary(f, company, cache_dir)

    eightk_filings = entity.get_filings(form=FORM_8K, date=date_range, amendments=False)
    for f in eightk_filings:
        if not _is_earnings_release_8k(f):
            continue
        yield _download_ex99(f, company, cache_dir)


def iter_cached_filings(cache_dir: Path = DEFAULT_CACHE_DIR) -> Iterator[DownloadedFiling]:
    """Every complete `DownloadedFiling` cache entry under `cache_dir`.

    `data/raw` is shared with other tools that key their own, differently
    shaped metadata by the same accession (see `_is_complete_cache_entry`);
    anything not written by this module -- including metadata from before
    the `sector` column existed, in Phase 0's one-off AAPL smoke filing --
    is skipped here rather than raising.
    """
    if not cache_dir.exists():
        return
    for meta_path in sorted(cache_dir.glob("*.json")):
        text_path = meta_path.with_suffix(".txt")
        if not _is_complete_cache_entry(meta_path, text_path):
            continue
        yield _from_cache(meta_path, text_path)
