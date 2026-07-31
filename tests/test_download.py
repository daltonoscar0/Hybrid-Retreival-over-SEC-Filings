"""Cache and helper contract for the download sweep.

No network calls: these exercise zero-padding, the resumable on-disk cache
round trip, and the universe filter on `iter_cached_filings`, all pure
enough to test without hitting EDGAR.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from ticker.download import (
    DownloadedFiling,
    _aware,
    _cache_paths,
    _from_cache,
    _is_complete_cache_entry,
    _period_end,
    _to_cache,
    iter_cached_filings,
    zero_pad_cik,
)


def test_zero_pad_cik_pads_int_to_ten_digits():
    assert zero_pad_cik(320193) == "0000320193"


def test_zero_pad_cik_pads_str_to_ten_digits():
    assert zero_pad_cik("320193") == "0000320193"


def test_zero_pad_cik_noop_when_already_ten_digits():
    assert zero_pad_cik(1234567890) == "1234567890"


def test_aware_adds_utc_to_naive_datetime():
    naive = datetime(2023, 6, 1, 12, 0, 0)
    result = _aware(naive)
    assert result.tzinfo is not None
    assert result.utcoffset().total_seconds() == 0


def test_aware_leaves_aware_datetime_unchanged():
    aware = datetime(2023, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert _aware(aware) is aware


def test_period_end_passes_through_date():
    d = date(2023, 6, 30)
    assert _period_end(d) == d


def test_period_end_converts_datetime_to_date():
    assert _period_end(datetime(2023, 6, 30, 8, 0)) == date(2023, 6, 30)


def test_period_end_parses_iso_string():
    assert _period_end("2023-06-30") == date(2023, 6, 30)


def test_period_end_none_for_empty_string():
    assert _period_end("") is None


def test_period_end_none_for_none():
    assert _period_end(None) is None


def _sample_filing(accession: str = "0000320193-24-000001") -> DownloadedFiling:
    return DownloadedFiling(
        accession=accession,
        cik=320193,
        ticker="AAPL",
        sector="semiconductors",
        form="10-K",
        filed_at=datetime(2024, 3, 1, tzinfo=timezone.utc),
        period_end=date(2023, 12, 31),
        url="https://example.com/" + accession,
        text="sample filing text",
    )


def test_cache_round_trip(tmp_path):
    filing = _sample_filing()
    meta_path = tmp_path / "acc.json"
    text_path = tmp_path / "acc.txt"
    _to_cache(meta_path, text_path, filing)
    result = _from_cache(meta_path, text_path)
    assert result == filing


def test_iter_cached_filings_yields_only_sector_tagged_entries(tmp_path):
    filing = _sample_filing("0000320193-24-000001")
    _to_cache(tmp_path / "0000320193-24-000001.json", tmp_path / "0000320193-24-000001.txt", filing)

    # a pre-Phase-1 cache entry with no "sector" key must be skipped, not raise
    legacy_meta = tmp_path / "0000320193-25-000079.json"
    legacy_meta.write_text(json.dumps({
        "accession": "0000320193-25-000079",
        "cik": 320193,
        "ticker": "AAPL",
        "form": "10-K",
        "filed_at": "2025-10-31T10:01:26+00:00",
        "period_end": "2025-09-27",
        "url": "https://example.com/legacy",
    }))
    (tmp_path / "0000320193-25-000079.txt").write_text("legacy text")

    result = list(iter_cached_filings(tmp_path))
    assert result == [filing]


def test_iter_cached_filings_on_missing_dir_returns_empty(tmp_path):
    missing = tmp_path / "does-not-exist"
    assert list(iter_cached_filings(missing)) == []


def test_foreign_partial_cache_entry_is_not_treated_as_complete(tmp_path):
    # data/raw is shared with other tools (e.g. the section-extraction
    # fixture script) that key their own, smaller metadata shape by the
    # same accession. A same-named .json/.txt pair from one of those must
    # not be mistaken for one of this module's complete cache entries --
    # that mistake is what crashed the sweep with a KeyError on "cik".
    accession = "0000109380-24-000134"
    meta_path, text_path = _cache_paths(accession, tmp_path)
    meta_path.write_text(json.dumps({
        "accession": accession, "ticker": "ZION", "form": "10-Q",
    }))
    text_path.write_text("raw filing text from a different tool's cache")
    assert _is_complete_cache_entry(meta_path, text_path) is False
    assert list(iter_cached_filings(tmp_path)) == []


def test_is_complete_cache_entry_true_for_own_writes(tmp_path):
    filing = _sample_filing()
    meta_path, text_path = _cache_paths(filing.accession, tmp_path)
    _to_cache(meta_path, text_path, filing)
    assert _is_complete_cache_entry(meta_path, text_path) is True


def test_is_complete_cache_entry_false_when_text_missing(tmp_path):
    filing = _sample_filing()
    meta_path, text_path = _cache_paths(filing.accession, tmp_path)
    meta_path.write_text(json.dumps({
        "accession": filing.accession, "cik": filing.cik, "ticker": filing.ticker,
        "sector": filing.sector, "form": filing.form,
        "filed_at": filing.filed_at.isoformat(), "period_end": None, "url": filing.url,
    }))
    assert _is_complete_cache_entry(meta_path, text_path) is False


def test_iter_cached_filings_skips_entry_missing_text_file(tmp_path):
    accession = "0000320193-24-000002"
    meta_path = tmp_path / f"{accession}.json"
    meta_path.write_text(json.dumps({
        "accession": accession,
        "cik": 320193,
        "ticker": "AAPL",
        "sector": "semiconductors",
        "form": "10-K",
        "filed_at": "2024-03-01T00:00:00+00:00",
        "period_end": None,
        "url": "https://example.com/" + accession,
    }))
    # no matching .txt written
    assert list(iter_cached_filings(tmp_path)) == []
