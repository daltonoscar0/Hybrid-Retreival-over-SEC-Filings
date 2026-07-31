#!/usr/bin/env python3
"""Produce section-extraction fixtures for human review.

For each accession: fetch the text `ticker.sections.extract_sections` would
see (the primary document for 10-K/10-Q, the EX-99.1 exhibit text for 8-K),
run the extractor, and write two files under `--raw-dir`
(default tests/fixtures/sections/raw/):

  {accession}.txt             the raw text, verbatim
  {accession}.candidate.json  the extractor's best guess

candidate.json's top-level "sections" map is `{item: [char_start, char_end]}`
-- exactly the shape a human reviewer copies into
tests/fixtures/sections/expected/{accession}.json after checking it (a hook
blocks this script from writing there itself). "preview" carries the first
and last 200 characters of each span plus its length so that check does not
require opening the raw dump. "failures" lists items the extractor could not
bound at all; those need a human to read the raw text directly, since there
is no candidate span to sanity-check.

Fetches go through a small on-disk cache under `--cache-dir` (default
data/raw/), keyed by accession in the same {accession}.json / {accession}.txt
shape ticker.download uses, so this and the corpus downloader can fill the
same cache concurrently without stepping on each other. For an 8-K, the
cached and returned text is the EX-99.1 exhibit body, not the full filing --
that is what extract_sections is contracted to receive for that form.

Usage:
  .venv/bin/python3 scripts/make_extraction_fixture.py NVDA 10-K 0001045810-24-000029
  .venv/bin/python3 scripts/make_extraction_fixture.py --all
  .venv/bin/python3 scripts/make_extraction_fixture.py --all --report reports/phase1_extraction.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import edgar  # noqa: E402

from ticker.download import DEFAULT_CACHE_DIR, ensure_identity, iter_cached_filings  # noqa: E402
from ticker.sections import (  # noqa: E402
    SectionExtraction,
    extract_sections,
    flag_length_outliers,
)

# Filings where the primary document's own filed HTML carries no recoverable
# narrative at all, confirmed by direct inspection: FITB's 2020-2023 10-Ks
# incorporate Items 1/1A/3/7/7A by reference to a PDF annual-report exhibit
# edgartools' `.text()` never touches, leaving only a front-matter table of
# contents and a wall of financial-statement tables (500K+ characters of raw
# text, well under 130K of it outside a table row, against 300K+ for a
# normal filing of similar size). No text-extraction fix reaches a PDF that
# was never fetched. This is a data-source limitation, not an extractor
# defect, and is counted separately from genuine misses in the corpus report
# rather than blended into either "success" or "failure."
NO_NARRATIVE_ACCESSIONS: frozenset[str] = frozenset({
    "0001193125-20-057751",  # FITB 10-K, filed 2020-03-02
    "0000035527-21-000100",  # FITB 10-K, filed 2021-02-26
    "0000035527-22-000119",  # FITB 10-K, filed 2022-02-25
    "0000035527-23-000122",  # FITB 10-K, filed 2023-02-24
})

# (accession, item) pairs where "start marker not found" is confirmed correct,
# not a miss: the filer omitted the heading entirely because it had nothing
# to disclose that period, verified by checking the same filer's other
# periods, where the heading appears normally and extracts cleanly. ADI
# alternates within the same 18-quarter span (8 quarters with "Item 1. Legal
# Proceedings" present and extracted, 10 without a trace of the heading
# anywhere in the body); TXN omitted it in every quarter from 2020-Q1 through
# 2024-Q2 and then started including and extracting it cleanly from
# 2024-Q3 onward; RF omitted "Item 1A. Risk Factors" in exactly 2 of its 18
# quarters, jumping straight from "Item 1. Legal Proceedings" to "Item 2." in
# both. None of these documents contain the omitted item's title anywhere,
# under any known heading variant, ruling out a rendering or regex miss.
LEGITIMATE_OMISSION_ACCESSIONS: frozenset[tuple[str, str]] = frozenset(
    (accession, "Part II Item 1") for accession in (
        "0000006281-20-000013", "0000006281-20-000087", "0000006281-20-000123",
        "0000006281-21-000022", "0000006281-21-000169", "0000006281-21-000197",
        "0000006281-22-000020", "0000006281-25-000023", "0000006281-25-000125",
        "0000006281-25-000144",  # ADI, 10 quarters
        "0000097476-20-000017", "0000097476-20-000026", "0000097476-21-000013",
        "0000097476-21-000020", "0000097476-21-000032", "0000097476-22-000028",
        "0000097476-22-000040", "0000097476-22-000048", "0000097476-23-000025",
        "0000097476-23-000035", "0000097476-23-000041", "0000097476-24-000021",
        "0001628280-20-014630",  # TXN, 13 quarters
    )
) | frozenset(
    (accession, "Part II Item 1A") for accession in (
        "0001281761-21-000043", "0001281761-21-000067",  # RF, 2 quarters
    )
)

RAW_DIR = Path("tests/fixtures/sections/raw")
EXPECTED_DIR = Path("tests/fixtures/sections/expected")
REPORT_PATH = Path("reports/phase1_extraction.md")

PREVIEW_CHARS = 200

# Fixture manifest: (ticker, form, accession). Spans both sectors in the
# locked universe (src/ticker/universe.py), forms 10-K/10-Q/8-K, and years
# 2020-2025. NVDA/WAL/ZION cover the common formatting; INTC is included
# deliberately even though its 10-K/10-Q push every "Item N." label into a
# cross-reference index at the end of the document instead of an inline
# heading -- see the module docstring in ticker.sections and the report this
# script writes. It is a real, filer-specific extraction gap that the full
# corpus build will hit for real, not a regex bug worth hiding by swapping
# in an easier company.
FIXTURE_MANIFEST: tuple[tuple[str, str, str], ...] = (
    ("NVDA", "10-K", "0001045810-20-000010"),
    ("NVDA", "10-K", "0001045810-23-000017"),
    ("NVDA", "10-K", "0001045810-25-000023"),
    ("NVDA", "10-Q", "0001045810-21-000064"),
    ("NVDA", "10-Q", "0001045810-24-000264"),
    ("INTC", "10-K", "0000050863-20-000011"),
    ("INTC", "10-K", "0000050863-24-000010"),
    ("INTC", "10-Q", "0000050863-21-000030"),
    ("INTC", "10-Q", "0000050863-25-000074"),
    ("WAL", "10-K", "0001212545-21-000085"),
    ("WAL", "10-K", "0001212545-24-000092"),
    ("WAL", "10-Q", "0001212545-20-000163"),
    ("WAL", "10-Q", "0001212545-23-000149"),
    ("ZION", "10-K", "0000109380-22-000072"),
    ("ZION", "10-K", "0000109380-25-000040"),
    ("ZION", "10-Q", "0000109380-21-000192"),
    ("ZION", "10-Q", "0000109380-24-000134"),
    ("NVDA", "8-K", "0001045810-25-000115"),
    ("INTC", "8-K", "0000050863-25-000169"),
    ("WAL", "8-K", "0001628280-25-045685"),
    ("ZION", "8-K", "0000109380-25-000124"),
)


def _cache_paths(accession: str, cache_dir: Path) -> tuple[Path, Path]:
    safe = accession.replace("/", "-")
    return cache_dir / f"{safe}.json", cache_dir / f"{safe}.txt"


def _read_cache(accession: str, cache_dir: Path) -> str | None:
    meta_path, text_path = _cache_paths(accession, cache_dir)
    if meta_path.exists() and text_path.exists():
        return text_path.read_text()
    return None


def _write_cache(accession: str, cache_dir: Path, ticker: str, form: str, text: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path, text_path = _cache_paths(accession, cache_dir)
    meta_path.write_text(json.dumps({"accession": accession, "ticker": ticker, "form": form}))
    text_path.write_text(text)


def _find_ex99_text(filing) -> str:
    """The EX-99.x exhibit text for an 8-K. Most carry exactly one; when a
    filing bundles more than one EX-99.x, the earnings release is reliably
    the longest one (short EX-99.1s alongside a same-day dividend notice or
    similar are a handful of paragraphs, not a press release).
    """
    candidates = [
        a for a in filing.attachments if "EX-99" in str(a.description or "").upper()
    ]
    if not candidates:
        raise RuntimeError(f"no EX-99.x exhibit found for {filing.accession_no}")
    texts = [(a, a.text() or "") for a in candidates]
    return max(texts, key=lambda pair: len(pair[1]))[1]


def fetch_text(ticker: str, form: str, accession: str, cache_dir: Path) -> str:
    cached = _read_cache(accession, cache_dir)
    if cached is not None:
        return cached

    ensure_identity()
    company = edgar.Company(ticker)
    filing = company.get_filings(form=form, accession_number=accession).latest()
    if filing is None:
        raise RuntimeError(f"accession {accession} not found for {ticker} {form}")

    text = _find_ex99_text(filing) if form.upper() == "8-K" else filing.text()
    if not text or not text.strip():
        raise RuntimeError(f"empty text fetched for {ticker} {form} {accession}")

    _write_cache(accession, cache_dir, ticker, form, text)
    return text


def _preview(text: str, start: int, end: int) -> dict:
    span = text[start:end]
    head = span[:PREVIEW_CHARS]
    tail = span[-PREVIEW_CHARS:] if len(span) > PREVIEW_CHARS else span
    return {"length": end - start, "head": head, "tail": tail}


def build_candidate(
    ticker: str, form: str, accession: str, text: str, result: SectionExtraction
) -> dict:
    return {
        "accession": accession,
        "ticker": ticker,
        "form": form,
        "text_length": len(text),
        "sections": {item: [start, end] for item, start, end in result.sections},
        "preview": {
            item: _preview(text, start, end) for item, start, end in result.sections
        },
        "failures": [{"item": item, "reason": reason} for item, reason in result.failures],
    }


def make_fixture(
    ticker: str, form: str, accession: str, cache_dir: Path, raw_dir: Path
) -> dict:
    text = fetch_text(ticker, form, accession, cache_dir)
    result = extract_sections(text, form)
    candidate = build_candidate(ticker, form, accession, text, result)

    raw_dir.mkdir(parents=True, exist_ok=True)
    safe = accession.replace("/", "-")
    (raw_dir / f"{safe}.txt").write_text(text)
    (raw_dir / f"{safe}.candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
    return candidate


def _write_fixture_section(lines: list[str], candidates: list[dict]) -> None:
    per_item_ok: dict[str, int] = {}
    per_item_total: dict[str, int] = {}
    failure_rows: list[tuple[str, str, str, str]] = []  # ticker, form, accession, "item: reason"

    outlier_entries: list[tuple[str, str, str, int]] = []
    for c in candidates:
        for item, (start, end) in c["sections"].items():
            outlier_entries.append((c["accession"], c["ticker"], item, end - start))
    outliers = flag_length_outliers(outlier_entries)
    outlier_by_accession_item: dict[tuple[str, str], str] = {
        (accession, item): reason for accession, item, reason in outliers
    }

    for c in candidates:
        seen_items = set(c["sections"]) | {f["item"] for f in c["failures"]}
        for item in seen_items:
            per_item_total[item] = per_item_total.get(item, 0) + 1
            key = (c["accession"], item)
            if item in c["sections"] and key not in outlier_by_accession_item:
                per_item_ok[item] = per_item_ok.get(item, 0) + 1
        for f in c["failures"]:
            failure_rows.append((c["ticker"], c["form"], c["accession"], f"{f['item']}: {f['reason']}"))
        for (accession, item), reason in outlier_by_accession_item.items():
            if accession == c["accession"]:
                failure_rows.append((c["ticker"], c["form"], accession, f"{item}: {reason}"))

    lines.append("## Fixture set (own candidates, human review pending)")
    lines.append("")
    lines.append(
        f"Candidate spans over {len(candidates)} fixture filings, extractor's own "
        "output only -- not yet checked against tests/fixtures/sections/expected/, "
        "which is empty until a human reviews the .candidate.json files below. Not "
        "the corpus-wide result above; this is 4% of the corpus and, per experience "
        "with INTC, not representative of it on its own."
    )
    lines.append("")
    lines.append("| item | ok | total | rate |")
    lines.append("|---|---|---|---|")
    for item in sorted(per_item_total):
        ok = per_item_ok.get(item, 0)
        total = per_item_total[item]
        lines.append(f"| {item} | {ok} | {total} | {ok / total:.0%} |")
    lines.append("")

    lines.append("### Fixture failures")
    lines.append("")
    if failure_rows:
        lines.append("| ticker | form | accession | item: reason |")
        lines.append("|---|---|---|---|")
        for ticker, form, accession, detail in failure_rows:
            lines.append(f"| {ticker} | {form} | {accession} | {detail} |")
    else:
        lines.append("None.")
    lines.append("")

    lines.append("### Accessions awaiting human review")
    lines.append("")
    lines.append(
        "tests/fixtures/sections/expected/ is empty. Every accession below has a "
        ".candidate.json under tests/fixtures/sections/raw/ ready to check against "
        "the paired .txt dump."
    )
    lines.append("")
    for c in candidates:
        lines.append(f"- {c['ticker']} {c['form']} {c['accession']}")
    lines.append("")


def write_report(
    report_path: Path,
    corpus_stats: dict | None = None,
    candidates: list[dict] | None = None,
) -> None:
    lines = ["# Phase 1: section extraction", ""]
    if corpus_stats is not None:
        _write_corpus_section(lines, corpus_stats)
    if candidates is not None:
        _write_fixture_section(lines, candidates)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))


def run_corpus_measurement(cache_dir: Path) -> dict:
    """Extraction success measured over every cached 10-K/10-Q/8-K filing --
    not the 21-filing fixture set, which is 4% of the corpus and was not
    representative of it.

    The per-item ok/total denominator excludes two things that are not
    misses, so neither is blended into the success rate: filings in
    `NO_NARRATIVE_ACCESSIONS` (excluded whole -- the primary document has no
    narrative to extract at all) and (accession, item) pairs in
    `LEGITIMATE_OMISSION_ACCESSIONS` (excluded per item -- the filer omitted
    that one heading because it had nothing to disclose that period,
    confirmed by checking the same filer's other periods). Both are returned
    separately, never folded into "ok" or "total" for any item.
    """
    per_item_ok: dict[str, int] = {}
    per_item_total: dict[str, int] = {}
    failures: list[tuple[str, str, str, str, str]] = []  # ticker, form, accession, item, reason
    legitimate_omissions: list[tuple[str, str, str, str]] = []  # ticker, form, accession, item
    excluded: list[tuple[str, str, str]] = []  # ticker, form, accession
    n_filings = 0

    for filing in iter_cached_filings(cache_dir):
        if filing.form not in ("10-K", "10-Q", "8-K"):
            continue
        if filing.accession in NO_NARRATIVE_ACCESSIONS:
            excluded.append((filing.ticker, filing.form, filing.accession))
            continue
        n_filings += 1
        result = extract_sections(filing.text, filing.form)
        seen_items = {item for item, _, _ in result.sections} | {item for item, _ in result.failures}
        for item in seen_items:
            if (filing.accession, item) in LEGITIMATE_OMISSION_ACCESSIONS:
                continue
            per_item_total[item] = per_item_total.get(item, 0) + 1
        for item, _, _ in result.sections:
            if (filing.accession, item) in LEGITIMATE_OMISSION_ACCESSIONS:
                continue
            per_item_ok[item] = per_item_ok.get(item, 0) + 1
        for item, reason in result.failures:
            if (filing.accession, item) in LEGITIMATE_OMISSION_ACCESSIONS:
                legitimate_omissions.append((filing.ticker, filing.form, filing.accession, item))
            else:
                failures.append((filing.ticker, filing.form, filing.accession, item, reason))

    return {
        "n_filings": n_filings,
        "per_item_ok": per_item_ok,
        "per_item_total": per_item_total,
        "failures": failures,
        "legitimate_omissions": legitimate_omissions,
        "excluded": excluded,
    }


def _write_corpus_section(lines: list[str], stats: dict) -> None:
    lines.append("## Corpus-wide extraction results")
    lines.append("")
    lines.append(
        f"Every cached 10-K/10-Q/8-K filing ({stats['n_filings']} filings), not the "
        "21-filing fixture set. Denominator excludes the filings listed under "
        "\"Excluded: no recoverable narrative\" and the (accession, item) pairs "
        "under \"Confirmed legitimate item omissions\" below -- neither is a miss, "
        "so neither counts against or for any item's rate."
    )
    lines.append("")
    lines.append("| item | ok | total | rate |")
    lines.append("|---|---|---|---|")
    for item in sorted(stats["per_item_total"]):
        ok = stats["per_item_ok"].get(item, 0)
        total = stats["per_item_total"][item]
        lines.append(f"| {item} | {ok} | {total} | {ok / total:.1%} |")
    lines.append("")

    lines.append("### Excluded: no recoverable narrative")
    lines.append("")
    if stats["excluded"]:
        lines.append(
            "Confirmed by direct inspection: the primary document's filed HTML "
            "carries no Item 1/1A/3/7/7A narrative at all (see NO_NARRATIVE_ACCESSIONS "
            "in scripts/make_extraction_fixture.py). Not counted as a miss above."
        )
        lines.append("")
        for ticker, form, accession in stats["excluded"]:
            lines.append(f"- {ticker} {form} {accession}")
    else:
        lines.append("None.")
    lines.append("")

    lines.append("### Confirmed legitimate item omissions (not misses, excluded from the rate)")
    lines.append("")
    lines.append(
        "\"Start marker not found\" here means the item's heading is absent from "
        "the filing, confirmed correct rather than a miss: see "
        "LEGITIMATE_OMISSION_ACCESSIONS in scripts/make_extraction_fixture.py for "
        "the per-filer evidence (alternating quarters with and without the "
        "heading present, extracting cleanly whenever it is). Not in the "
        "denominator or numerator of any item's rate above."
    )
    lines.append("")
    if stats["legitimate_omissions"]:
        lines.append("| ticker | form | accession | item |")
        lines.append("|---|---|---|---|")
        for ticker, form, accession, item in sorted(stats["legitimate_omissions"]):
            lines.append(f"| {ticker} | {form} | {accession} | {item} |")
    else:
        lines.append("None.")
    lines.append("")

    lines.append("### Genuine misses (unresolved)")
    lines.append("")
    lines.append(
        "Everything else: a real heading exists that this extractor did not "
        "find, or the document's structure defeated Part I/II disambiguation. "
        "Counted as a failure in the table above."
    )
    lines.append("")
    if stats["failures"]:
        lines.append("| ticker | form | accession | item | reason |")
        lines.append("|---|---|---|---|---|")
        for ticker, form, accession, item, reason in sorted(stats["failures"]):
            lines.append(f"| {ticker} | {form} | {accession} | {item} | {reason} |")
    else:
        lines.append("None.")
    lines.append("")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker", nargs="?")
    parser.add_argument("form", nargs="?")
    parser.add_argument("accession", nargs="?")
    parser.add_argument("--all", action="store_true", help="run the full FIXTURE_MANIFEST")
    parser.add_argument(
        "--corpus", action="store_true",
        help="measure extraction over every cached 10-K/10-Q/8-K filing, not just the fixture set",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    if not (args.all or args.corpus or (args.ticker and args.form and args.accession)):
        parser.error("one of --all, --corpus, or TICKER FORM ACCESSION is required")
        return

    candidates = None
    if args.all or (args.ticker and args.form and args.accession):
        manifest = FIXTURE_MANIFEST if args.all else ((args.ticker, args.form, args.accession),)
        candidates = []
        for ticker, form, accession in manifest:
            try:
                candidate = make_fixture(ticker, form, accession, args.cache_dir, args.raw_dir)
            except Exception as exc:  # noqa: BLE001 -- one bad accession must not abort the batch
                print(f"FETCH FAILED  {ticker} {form} {accession}: {exc!r}")
                continue
            n_ok = len(candidate["sections"])
            n_fail = len(candidate["failures"])
            print(f"{ticker:6s} {form:6s} {accession}  ok={n_ok} fail={n_fail}")
            for f in candidate["failures"]:
                print(f"    FAIL {f['item']}: {f['reason']}")
            candidates.append(candidate)

    corpus_stats = None
    if args.corpus:
        corpus_stats = run_corpus_measurement(args.cache_dir)
        print(f"\ncorpus: {corpus_stats['n_filings']} filings measured")
        for item in sorted(corpus_stats["per_item_total"]):
            ok = corpus_stats["per_item_ok"].get(item, 0)
            total = corpus_stats["per_item_total"][item]
            print(f"  {item:20s} {ok:4d}/{total:4d}  {ok / total:.1%}")

    if args.all or args.corpus:
        write_report(args.report, corpus_stats=corpus_stats, candidates=candidates)
        print(f"\nwrote {args.report}")

    if candidates:
        expected_present = {p.stem for p in EXPECTED_DIR.glob("*.json")} if EXPECTED_DIR.exists() else set()
        pending = [c for c in candidates if c["accession"].replace("/", "-") not in expected_present]
        if pending:
            print(f"\n{len(pending)} accessions awaiting human review (no expected/*.json yet):")
            for c in pending:
                print(f"  {c['ticker']} {c['form']} {c['accession']}")


if __name__ == "__main__":
    main()
