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
import re
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
from ticker.universe import UNIVERSE  # noqa: E402

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
# The set is not a judgment call. Counting non-table characters per 10-K
# across the universe, these nine occupy the entire low tail at 15,522 to
# 60,907 characters, the next filing above them has 155,320, and the corpus
# median is 383,652. No threshold anywhere inside that 94,413-character gap
# changes the membership. All nine are regional banks and six are 2020 filings
# covering fiscal 2019.
NO_NARRATIVE_ACCESSIONS: frozenset[str] = frozenset({
    "0000035527-23-000122",  # FITB 10-K, 15,522 non-table chars
    "0000035527-22-000119",  # FITB 10-K, 15,654
    "0000035527-21-000100",  # FITB 10-K, 15,732
    "0000109380-20-000092",  # ZION 10-K, 21,319
    "0000049196-20-000010",  # HBAN 10-K, 30,766
    "0001281761-20-000010",  # RF   10-K, 48,910
    "0001069157-20-000016",  # EWBC 10-K, 51,504
    "0000091576-20-000007",  # KEY  10-K, 57,330
    "0001193125-20-057751",  # FITB 10-K, 60,907
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

# Fixture manifest: (ticker, form, accession), 24 filings.
#
# The previous manifest drew 21 filings from 5 CIKs, 6 of them from one, and
# reported a 100% fixture extraction rate against a corpus rate of 92.5%. That
# gap is what an unrepresentative fixture set looks like: the five filers in it
# happened to be the ones the extractor already handled, so the set could not
# have caught the two failure modes that actually existed.
#
# Selection rule, applied in the fixed universe order in
# src/ticker/universe.py so it reproduces exactly:
#
#   Base, 20 filings. One per company, all 20 CIKs. Form alternates by
#   position, giving 10 10-Ks and 10 10-Qs and, because the universe
#   interleaves the sectors in blocks of ten, 5 of each form per sector. The
#   filing year is the company's position modulo the count of years it filed
#   that form, so the set spreads across 2020 through 2025 rather than
#   clustering. Filings in NO_NARRATIVE_ACCESSIONS are excluded from the draw:
#   there are no spans in them for a reviewer to approve.
#
#   Edge cases, 4 filings, named rather than drawn. CMA's 10-K satisfies Item 7
#   by pointing into the F-pages of the same document. RF's 2021 10-K presents
#   Items 7 and 7A jointly under back-to-back headings. Both are the filings
#   the short-Item-7 repairs in ticker.sections exist for, and an approved
#   fixture is the only thing that turns those repairs from plausible into
#   checked. Two 8-Ks, one per sector, cover EX-99.1; WAL's is 7,164 characters,
#   near the short end of the exhibit distribution.
#
# INTC needs no edge-case slot: its base draw is a 10-K, and every INTC 10-K
# pushes the "Item N." labels into a cross-reference index instead of using
# inline headings, so the integrated-report path is covered by the base set.
#
# One filing per company rather than one of each form per company. Two per
# company is 40 filings, and at the roughly two and a half minutes a careful
# span check takes that is close to two hours of review against the one hour
# STOP 1 budgets. Coverage of all 20 CIKs was kept and per-company coverage of
# both forms was given up, because the failure modes seen so far are
# filer-specific rather than form-specific.
FIXTURE_MANIFEST: tuple[tuple[str, str, str], ...] = (
    ("INTC", "10-K", "0000050863-20-000011"),
    ("NVDA", "10-Q", "0001045810-21-000131"),
    ("TXN", "10-K", "0000097476-22-000009"),
    ("QCOM", "10-Q", "0000804328-23-000023"),
    ("AVGO", "10-K", "0001730168-24-000139"),
    ("MU", "10-Q", "0000723125-25-000021"),
    ("ADI", "10-K", "0000006281-20-000156"),
    ("NXPI", "10-Q", "0001413447-21-000062"),
    ("MCHP", "10-K", "0000827054-22-000094"),
    ("ON", "10-Q", "0001628280-23-026196"),
    ("ZION", "10-K", "0000109380-21-000082"),
    ("RF", "10-Q", "0001281761-25-000063"),
    ("HBAN", "10-K", "0000049196-23-000020"),
    ("KEY", "10-Q", "0000091576-21-000114"),
    ("FITB", "10-K", "0000035527-24-000088"),
    ("CFG", "10-Q", "0000759944-23-000124"),
    ("WAL", "10-K", "0001212545-24-000092"),
    ("EWBC", "10-Q", "0001069157-25-000096"),
    ("CFR", "10-K", "0000039263-20-000010"),
    ("CMA", "10-Q", "0000028412-21-000140"),
    ("CMA", "10-K", "0000028412-23-000094"),
    ("RF", "10-K", "0001281761-21-000012"),
    ("NVDA", "8-K", "0001045810-22-000163"),
    ("WAL", "8-K", "0001212545-23-000109"),
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
        "which is empty until a human reviews the .candidate.json files below. This "
        "is not the corpus-wide result above and is not a substitute for it. The "
        "previous fixture set reported 100% against a corpus rate of 92.5%, because "
        "its 21 filings came from 5 CIKs that the extractor already handled. This "
        "set covers all 20."
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


# Every target item, and which form's filing count is its raw denominator.
# Hardcoded rather than derived from what the extractor happened to emit: an
# item that fails on every single filing would otherwise vanish from the table
# entirely instead of showing a 0% rate, which is the one case the table most
# needs to show.
ITEM_FORMS: tuple[tuple[str, str], ...] = (
    ("1", "10-K"),
    ("1A", "10-K"),
    ("3", "10-K"),
    ("7", "10-K"),
    ("7A", "10-K"),
    ("Part I Item 2", "10-Q"),
    ("Part II Item 1", "10-Q"),
    ("Part II Item 1A", "10-Q"),
    ("EX-99.1", "8-K"),
)

# A section that is a pointer rather than prose: short, and saying so. Counted
# and reported, never dropped. These are complete sections as filed, so
# excluding them would be deleting real corpus, but a per-firm language model
# fit partly on cross-reference sentences is learning the filer's boilerplate
# for pointing at a footnote, which is worth knowing when reading the Phase 5.2
# novelty-by-item table.
_CROSS_REFERENCE_RE = re.compile(
    r"incorporated (?:herein )?by reference|reference is made to|"
    r"see note \d|refer to note \d",
    re.IGNORECASE,
)
_CROSS_REFERENCE_MAX_CHARS = 1000


def run_corpus_measurement(cache_dir: Path) -> dict:
    """Extraction success over every cached filing from the locked universe.

    Three rates per item, all three reported, because a rate quoted only
    after removing cases from its own denominator is not checkable:

      raw         ok divided by every filing of that form
      extractable filings of that form, minus the ones under
                  NO_NARRATIVE_ACCESSIONS (the primary document carries no
                  item narrative at all) and minus the (accession, item)
                  pairs under LEGITIMATE_OMISSION_ACCESSIONS (the filer
                  omitted that one heading that period, confirmed against the
                  same filer's other periods)
      adjusted    ok divided by extractable

    Filings outside `ticker.universe.UNIVERSE` are skipped and counted
    separately. The cache is a working directory and can hold a filing pulled
    for a one-off check; letting one into the denominator would move every
    rate by a fraction of a percent for no reason anyone could reconstruct.
    """
    universe_ciks = {company.cik for company in UNIVERSE}

    per_item_ok: dict[str, int] = {}
    form_totals: dict[str, int] = {}
    failures: list[tuple[str, str, str, str, str]] = []
    legitimate_omissions: list[tuple[str, str, str, str]] = []
    excluded: list[tuple[str, str, str]] = []
    off_universe: list[tuple[str, str, str]] = []
    cross_references: list[tuple[str, str, str, str, int]] = []
    n_filings = 0

    for filing in iter_cached_filings(cache_dir):
        if filing.form not in ("10-K", "10-Q", "8-K"):
            continue
        if filing.cik not in universe_ciks:
            off_universe.append((filing.ticker, filing.form, filing.accession))
            continue
        form_totals[filing.form] = form_totals.get(filing.form, 0) + 1
        if filing.accession in NO_NARRATIVE_ACCESSIONS:
            excluded.append((filing.ticker, filing.form, filing.accession))
            continue
        n_filings += 1

        result = extract_sections(filing.text, filing.form)
        for item, start, end in result.sections:
            per_item_ok[item] = per_item_ok.get(item, 0) + 1
            span = filing.text[start:end]
            if len(span) <= _CROSS_REFERENCE_MAX_CHARS and _CROSS_REFERENCE_RE.search(span):
                cross_references.append(
                    (filing.ticker, filing.form, filing.accession, item, len(span))
                )
        for item, reason in result.failures:
            if (filing.accession, item) in LEGITIMATE_OMISSION_ACCESSIONS:
                legitimate_omissions.append(
                    (filing.ticker, filing.form, filing.accession, item)
                )
            else:
                failures.append(
                    (filing.ticker, filing.form, filing.accession, item, reason)
                )

    n_no_narrative_by_form: dict[str, int] = {}
    for _, form, _ in excluded:
        n_no_narrative_by_form[form] = n_no_narrative_by_form.get(form, 0) + 1

    rows = []
    for item, form in ITEM_FORMS:
        raw_total = form_totals.get(form, 0)
        omitted = sum(1 for _, _, _, i in legitimate_omissions if i == item)
        extractable = raw_total - n_no_narrative_by_form.get(form, 0) - omitted
        rows.append(
            {
                "item": item,
                "form": form,
                "ok": per_item_ok.get(item, 0),
                "raw_total": raw_total,
                "extractable": extractable,
            }
        )

    return {
        "n_filings": n_filings,
        "form_totals": form_totals,
        "rows": rows,
        "failures": failures,
        "legitimate_omissions": legitimate_omissions,
        "excluded": excluded,
        "off_universe": off_universe,
        "cross_references": cross_references,
    }


def _write_corpus_section(lines: list[str], stats: dict) -> None:
    form_totals = stats["form_totals"]
    lines.append("## Corpus-wide extraction results")
    lines.append("")
    lines.append(
        "Every cached filing from the locked 20-company universe: "
        + ", ".join(f"{n} {form}" for form, n in sorted(form_totals.items()))
        + ". Not the 24-filing fixture set, which is 2.5% of the corpus and, per "
        "the last fixture set's 100% against a corpus 92.5%, not representative "
        "of it on its own."
    )
    lines.append("")
    lines.append(
        "Three rates per item. The raw rate divides by every filing of that form. "
        "The extractable count removes the filings under \"Excluded: no "
        "recoverable narrative\" and the (accession, item) pairs under \"Confirmed "
        "legitimate item omissions\", neither of which is a section this extractor "
        "could have found. The adjusted rate divides by that. All three are shown "
        "because a rate quoted only after removing cases from its own denominator "
        "is not checkable."
    )
    lines.append("")
    lines.append("| item | form | ok | all filings | raw | extractable | adjusted |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in stats["rows"]:
        raw = row["ok"] / row["raw_total"] if row["raw_total"] else 0.0
        adjusted = row["ok"] / row["extractable"] if row["extractable"] else 0.0
        lines.append(
            f"| {row['item']} | {row['form']} | {row['ok']} | {row['raw_total']} | "
            f"{raw:.1%} | {row['extractable']} | {adjusted:.1%} |"
        )
    lines.append("")
    lines.append(
        "These are census counts over the whole corpus, not estimates from a "
        "sample, so no confidence interval applies."
    )
    lines.append("")

    if stats["off_universe"]:
        lines.append("### Skipped: outside the locked universe")
        lines.append("")
        lines.append(
            "Present in the cache but not in `ticker.universe.UNIVERSE`, so not in "
            "any denominator above. The cache is a working directory and can hold "
            "a filing pulled for a one-off check."
        )
        lines.append("")
        for ticker, form, accession in sorted(stats["off_universe"]):
            lines.append(f"- {ticker} {form} {accession}")
        lines.append("")

    lines.append("### Excluded: no recoverable narrative")
    lines.append("")
    if stats["excluded"]:
        lines.append(
            "The filed HTML primary document carries no item narrative at all. "
            "These filers incorporate the business description, risk factors, and "
            "MD&A by reference to an annual report filed as a separate exhibit, "
            "leaving the primary document as financial statement tables and XBRL. "
            "Counting non-table characters per 10-K across the universe, these "
            "occupy the entire low tail at 15,522 to 60,907 characters, the next "
            "filing above them has 155,320, and the corpus median is 383,652. No "
            "threshold anywhere inside that gap changes the membership. See "
            "NO_NARRATIVE_ACCESSIONS in scripts/make_extraction_fixture.py."
        )
        lines.append("")
        for ticker, form, accession in sorted(stats["excluded"]):
            lines.append(f"- {ticker} {form} {accession}")
    else:
        lines.append("None.")
    lines.append("")

    lines.append("### Cross-reference sections (extracted, and counted separately)")
    lines.append("")
    lines.append(
        f"{len(stats['cross_references'])} extracted sections are under "
        f"{_CROSS_REFERENCE_MAX_CHARS} characters and consist of a pointer to a "
        "financial statement note or another part of the document rather than "
        "narrative prose. These are complete sections as filed, not truncations, "
        "and they are kept: dropping them would be deleting real corpus. They are "
        "counted here because a per-firm language model fit partly on "
        "cross-reference sentences is learning how that filer words a pointer, "
        "which is worth knowing when reading the Phase 5.2 novelty-by-item table."
    )
    lines.append("")
    if stats["cross_references"]:
        by_item: dict[str, list[int]] = {}
        for _, _, _, item, length in stats["cross_references"]:
            by_item.setdefault(item, []).append(length)
        lines.append("| item | sections | median chars |")
        lines.append("|---|---|---|")
        for item in sorted(by_item):
            lengths = sorted(by_item[item])
            lines.append(
                f"| {item} | {len(lengths)} | {lengths[len(lengths) // 2]} |"
            )
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

    joint = [row for row in stats["failures"] if row[4].startswith("presented jointly")]
    unresolved = [row for row in stats["failures"] if not row[4].startswith("presented jointly")]

    lines.append("### Items folded into a jointly presented section (not misses)")
    lines.append("")
    lines.append(
        "The filer put two item headings back to back and ran one narrative "
        "under both. The combined span is emitted under the earlier item and "
        "the later one is not emitted at all, because two sections over the "
        "same offsets would double every sentence in them through the chunker "
        "and into both language models. No text is lost. These count against "
        "the later item's rate above, which is the conservative reading: the "
        "item has no span of its own."
    )
    lines.append("")
    if joint:
        lines.append("| ticker | form | accession | item |")
        lines.append("|---|---|---|---|")
        for ticker, form, accession, item, _ in sorted(joint):
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
    if unresolved:
        lines.append("| ticker | form | accession | item | reason |")
        lines.append("|---|---|---|---|---|")
        for ticker, form, accession, item, reason in sorted(unresolved):
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
        for row in corpus_stats["rows"]:
            raw = row["ok"] / row["raw_total"] if row["raw_total"] else 0.0
            adjusted = row["ok"] / row["extractable"] if row["extractable"] else 0.0
            print(
                f"  {row['item']:20s} {row['ok']:4d}/{row['raw_total']:4d} raw {raw:6.1%}"
                f"   {row['ok']:4d}/{row['extractable']:4d} adjusted {adjusted:6.1%}"
            )

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
