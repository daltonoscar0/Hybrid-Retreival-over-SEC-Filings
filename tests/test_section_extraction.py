"""Section-boundary regression tests for ticker.sections.extract_sections.

Two layers. The synthetic tests below construct minimal filing-shaped text
by hand and exercise the extractor's actual decision rules directly: real
heading vs table-of-contents row vs inline cross-reference, Part I/Part II
disambiguation, the split-letter rendering quirk, and the "never truncate,
fail loud instead" contract for a missing start or unbounded end. These run
unconditionally and do not depend on any fixture.

The fixture-driven test below that reads every tests/fixtures/sections/
expected/*.json a human has approved (produced by reviewing the
.candidate.json files scripts/make_extraction_fixture.py writes to
tests/fixtures/sections/raw/) and asserts the extractor reproduces those
offsets on the real filing text. expected/ starts empty -- a hook blocks
this test file, the fixture script, and everything else from writing there
-- so this collects zero cases and reports a skip rather than failing on an
empty corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ticker.sections import extract_sections, flag_length_outliers

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sections"
RAW_DIR = FIXTURES_DIR / "raw"
EXPECTED_DIR = FIXTURES_DIR / "expected"

# A human reviewer reads offsets off the same raw/{accession}.txt dump this
# test reads, so a real match should be exact. The tolerance exists for one
# legitimate source of disagreement -- whether a reviewer's copied span
# includes the blank line immediately around a heading -- not as slack for a
# boundary miss. Misses in this extractor land on the wrong header entirely:
# hundreds to tens of thousands of characters off, never a handful. 5
# characters covers "\n\n" plus a little padding and nothing more.
TOLERANCE_CHARS = 5


# ---------------------------------------------------------------------------
# synthetic fixtures: exercise the extractor's rules without network access
# ---------------------------------------------------------------------------

SYNTHETIC_10K = (
    "UNITED STATES SECURITIES AND EXCHANGE COMMISSION\n\n"
    # Single newline, not a blank line, between the TOC caption and its
    # first row -- matching real filings, where the TOC's first row is
    # never itself preceded by a paragraph break. A blank line here would
    # make this test pass for the wrong reason: it would exercise "first
    # line of a block is preceded by blank line" rather than the TOC
    # look-alike this test is actually named for.
    "Table of Contents\n"
    "  Item 1.         Business                                    4  \n"
    "  Item 1A.        Risk Factors                                13  \n"
    "  Item 3.         Legal Proceedings                           30  \n"
    "  Item 7.         Management's Discussion and Analysis        40  \n"
    "  Item 7A.        Quantitative and Qualitative Disclosures    55  \n"
    "  Item 8.         Financial Statements                        60  \n\n"
    "Part I\n\n"
    "Item 1. Business\n\n"
    "We make widgets. See “Item 1A. Risk Factors” for more.\n\n"
    "Item 1A. Risk Factors\n\n"
    "Widgets may break.\n\n"
    "Item 2. Properties\n\n"
    "One factory.\n\n"
    "Item 3. Legal Proceedings\n\n"
    "None pending.\n\n"
    "Item 6. [Reserved]\n\n"
    "Item 7. Management's Discussion and Analysis\n\n"
    "Revenue grew, as discussed in “Item 1A. Risk Factors” above.\n\n"
    "Item 7A. Quantitative and Qualitative Disclosures\n\n"
    "Rates matter.\n\n"
    "Item 8. Financial Statements\n\n"
    "See attached.\n"
)


def test_10k_synthetic_resolves_all_targets_to_real_headers_not_toc():
    result = extract_sections(SYNTHETIC_10K, "10-K")
    assert result.failures == []
    got = {item: (start, end) for item, start, end in result.sections}
    assert set(got) == {"1", "1A", "3", "7", "7A"}

    real_item_1 = SYNTHETIC_10K.index("Item 1. Business\n\nWe make widgets")
    real_item_1a = SYNTHETIC_10K.index("Item 1A. Risk Factors\n\nWidgets may break")
    assert got["1"] == (real_item_1, real_item_1a)


def test_10k_end_boundary_is_next_header_regardless_of_item_identity():
    result = extract_sections(SYNTHETIC_10K, "10-K")
    got = {item: (start, end) for item, start, end in result.sections}
    real_item_3 = SYNTHETIC_10K.index("Item 3. Legal Proceedings\n\nNone pending")
    real_item_7 = SYNTHETIC_10K.index("Item 7. Management's Discussion")
    # Item 4/5/6 sit between 3 and 7 in this synthetic doc and are not
    # targets; Item 3's span must still stop at the very next header (6),
    # not swallow everything up to the next *target* header (7).
    real_item_6 = SYNTHETIC_10K.index("Item 6. [Reserved]")
    assert got["3"] == (real_item_3, real_item_6)
    assert real_item_6 < real_item_7


def test_10k_cross_references_do_not_shift_the_real_boundary():
    # Both "Item 1A. Risk Factors" cross-references sit inside other items'
    # bodies, preceded by a smart quote rather than a blank line. If they
    # were mistaken for headers, Item 1's span would truncate early or
    # Item 7's body would fracture.
    result = extract_sections(SYNTHETIC_10K, "10-K")
    got = {item: (start, end) for item, start, end in result.sections}
    start, end = got["7"]
    body = SYNTHETIC_10K[start:end]
    assert "Revenue grew" in body
    assert body.count("Item 7A.") == 0  # 7A itself must not be inside 7's span


def test_10k_missing_item_is_a_loud_failure_not_a_truncated_section():
    text = SYNTHETIC_10K.replace(
        "Item 7A. Quantitative and Qualitative Disclosures\n\nRates matter.\n\n", ""
    )
    result = extract_sections(text, "10-K")
    got = {item for item, _, _ in result.sections}
    assert "7A" not in got
    failed = dict(result.failures)
    assert failed["7A"] == "start marker not found"
    # Item 7 must not silently absorb 7A's would-be text or run to EOF.
    item7_span = next((s, e) for item, s, e in result.sections if item == "7")
    assert text[item7_span[1]:item7_span[1] + len("Item 8.")] == "Item 8."


def test_10k_no_header_after_start_marker_fails_rather_than_taking_rest_of_doc():
    # Item 7A is the very last heading in the document -- there is nothing
    # after it to bound the section, so it must fail, not run to len(text).
    text = SYNTHETIC_10K[: SYNTHETIC_10K.index("Item 8. Financial Statements")]
    result = extract_sections(text, "10-K")
    got = {item for item, _, _ in result.sections}
    assert "7A" not in got
    failed = dict(result.failures)
    assert "no closing boundary" in failed["7A"]


def test_10k_split_letter_rendering_is_recovered_for_known_titles():
    # Some filers' HTML-to-text conversion splits "Item 1A." into
    # "Item 1. A" -- see ZION 0000109380-21-000192 in the real fixture set.
    text = SYNTHETIC_10K.replace(
        "Item 1A. Risk Factors\n\nWidgets may break.",
        "Item 1. A Risk Factors\n\nWidgets may break.",
    )
    result = extract_sections(text, "10-K")
    got = {item: (start, end) for item, start, end in result.sections}
    assert "1A" in got
    start, _ = got["1A"]
    assert text[start:].startswith("Item 1. A Risk Factors")


def test_10k_indefinite_article_after_item_number_is_not_treated_as_a_letter():
    # "Item 3. A" must not be mistaken for the split-letter quirk just
    # because "A" follows the period -- it is only recognized for the exact
    # canonical 1A/7A titles, precisely to avoid this.
    text = SYNTHETIC_10K.replace(
        "Item 3. Legal Proceedings\n\nNone pending.",
        "Item 3. A previously disclosed matter was settled.",
    )
    result = extract_sections(text, "10-K")
    got = {item for item, _, _ in result.sections}
    assert "3" in got  # still resolves via the plain "Item 3." heading form
    assert "3A" not in got


def test_unsupported_form_raises():
    with pytest.raises(ValueError):
        extract_sections("whatever", "S-1")


SYNTHETIC_10Q = (
    # See the comment on SYNTHETIC_10K's TOC block: single newline, not a
    # blank line, above the first TOC row.
    "Table of Contents\n"
    "                 Part I : Financial Information            2  \n"
    "                 Part II : Other Information               9  \n\n"
    "Part I. Financial Information\n\n"
    "Item 1. Financial Statements\n\n"
    "Balance sheet here.\n\n"
    "Item 2. Management's Discussion and Analysis\n\n"
    "Results improved.\n\n"
    "Item 3. Quantitative and Qualitative Disclosures\n\n"
    "Rates matter.\n\n"
    "Item 4. Controls and Procedures\n\n"
    "Effective.\n\n"
    "Part II. Other Information\n\n"
    "Item 1. Legal Proceedings\n\n"
    "None pending.\n\n"
    "Item 1A. Risk Factors\n\n"
    "No material changes.\n\n"
    "Item 6. Exhibits\n\n"
    "See index.\n"
)


def test_10q_disambiguates_item_1_by_part():
    result = extract_sections(SYNTHETIC_10Q, "10-Q")
    assert result.failures == []
    got = {item: (start, end) for item, start, end in result.sections}
    assert set(got) == {"Part I Item 2", "Part II Item 1", "Part II Item 1A"}

    part2_item1_start, _ = got["Part II Item 1"]
    assert SYNTHETIC_10Q[part2_item1_start:part2_item1_start + 40].startswith(
        "Item 1. Legal Proceedings"
    )

    part1_item2_start, _ = got["Part I Item 2"]
    assert SYNTHETIC_10Q[part1_item2_start:part1_item2_start + 45].startswith(
        "Item 2. Management's Discussion"
    )


def test_10q_toc_rows_are_not_mistaken_for_part_headers():
    result = extract_sections(SYNTHETIC_10Q, "10-Q")
    got = {item: (start, end) for item, start, end in result.sections}
    real_part1 = SYNTHETIC_10Q.index("Part I. Financial Information\n\n")
    start, _ = got["Part I Item 2"]
    assert start > real_part1  # not the "Part I :" TOC row near byte 0


def test_10q_missing_part_structure_fails_all_targets_loudly():
    text = SYNTHETIC_10Q.replace("Part I. Financial Information\n\n", "")
    result = extract_sections(text, "10-Q")
    assert result.sections == []
    failed_items = {item for item, _ in result.failures}
    assert failed_items == {"Part I Item 2", "Part II Item 1", "Part II Item 1A"}
    assert all("Part I / Part II split" in reason for _, reason in result.failures)


# ---------------------------------------------------------------------------
# integrated-report fallback: filers (observed: Intel, 2020-2025) whose body
# never uses a literal "Item N." heading at all -- every item number appears
# exactly once, in a trailing cross-reference index that maps items to page
# numbers -- and page numbers do not survive text extraction. The
# item-label path above finds nothing there and, unmodified, would report
# all five target items missing.
# ---------------------------------------------------------------------------

SYNTHETIC_INTEGRATED_10K = (
    "UNITED STATES SECURITIES AND EXCHANGE COMMISSION\n\n"
    "Table of Contents\n"
    "  Fundamentals of Our Business                    Page  \n"
    "  Introduction to Our Business                       3  \n"
    "  A Year in Review                                   5  \n"
    "  Our Strategy                                       7  \n"
    "  Management's Discussion and Analysis                  \n"
    "  Our Products                                      20  \n"
    "  Segment Trends and Results                        21  \n"
    "  Risk Factors and Other Key Information                \n"
    "  Risk Factors                                      48  \n"
    "  Quantitative and Qualitative Disclosures about Market Risk  64  \n"
    "  Financial Statements and Supplemental Details          \n"
    "  Notes to Consolidated Financial Statements         79  \n"
    "  Key Terms                                         112  \n"
    "  Form 10-K Cross-Reference Index                    121  \n"
    "\n"
    "Table of Contents\n\n"
    "Forward-Looking Statements\n\n"
    "This report contains forward-looking statements.\n\n"
    # "Introduction to Our Business" never renders as a text heading -- an
    # image-only spread in the real filings this pattern is modeled on --
    # so the chapter's *second* listed subsection is the real anchor.
    "Introduction to Our Business Spread.jpg\n\n"
    "A Year in Review\n\n"
    "Widget demand grew this year.\n\n"
    "Our Strategy\n\n"
    "We aim to lead the market.\n\n"
    "Management's Discussion and Analysis\n\n"
    "Our Products\n\n"
    "We sell widgets and gadgets.\n\n"
    "Segment Trends and Results\n\n"
    "Widget segment revenue increased.\n\n"
    "Risk Factors and Other Key Information\n\n"
    "Risk Factors\n\n"
    "Our business faces various risks.\n\n"
    "Quantitative and Qualitative Disclosures about Market Risk\n\n"
    "We are exposed to interest rate risk.\n\n"
    "Financial Statements and Supplemental Details\n\n"
    "Notes to Consolidated Financial Statements\n\n"
    "Legal Proceedings\n\n"
    "We are party to various legal proceedings.\n\n"
    "Key Terms\n\n"
    "Definitions used throughout this report.\n\n"
    "Form 10-K Cross-Reference Index\n\n"
    "Item Number      Item\n"
    "Item 1.          Business:                 Pages 3-9\n"
    "Item 1A.         Risk Factors               Pages 48-62\n"
    "Item 3.          Legal Proceedings          Pages 79-80\n"
    "Item 7.          Management's Discussion and Analysis   Pages 20-45\n"
    "Item 7A.         Quantitative and Qualitative Disclosures About Market Risk   Pages 64-65\n"
)


def test_10k_integrated_report_recovers_all_five_items():
    result = extract_sections(SYNTHETIC_INTEGRATED_10K, "10-K")
    assert result.failures == []
    got = {item: (s, e) for item, s, e in result.sections}
    assert set(got) == {"1", "1A", "3", "7", "7A"}

    start, _ = got["1"]
    assert SYNTHETIC_INTEGRATED_10K[start:].startswith("A Year in Review")
    start, _ = got["7"]
    assert SYNTHETIC_INTEGRATED_10K[start:].startswith("Our Products")


def test_10k_integrated_report_chapter_span_includes_interior_subsections():
    # Item 1's real content runs through every subsection of its chapter
    # ("A Year in Review" *and* "Our Strategy"), not just the first one
    # found -- bounding it against the next TOC-listed heading of any kind
    # would truncate at "Our Strategy" and lose almost everything.
    result = extract_sections(SYNTHETIC_INTEGRATED_10K, "10-K")
    got = {item: (s, e) for item, s, e in result.sections}
    start, end = got["1"]
    body = SYNTHETIC_INTEGRATED_10K[start:end]
    assert "Our Strategy" in body
    assert "We aim to lead the market" in body
    start, end = got["7"]
    body = SYNTHETIC_INTEGRATED_10K[start:end]
    assert "Segment Trends and Results" in body


def test_10k_integrated_report_single_subsection_items_stop_at_next_heading():
    # 1A has no children of its own, so it should stop at the very next
    # real heading (7A here), not run on into unrelated chapters.
    result = extract_sections(SYNTHETIC_INTEGRATED_10K, "10-K")
    got = {item: (s, e) for item, s, e in result.sections}
    start, end = got["1A"]
    body = SYNTHETIC_INTEGRATED_10K[start:end]
    assert "various risks" in body
    assert "interest rate risk" not in body  # that is 7A's content, not 1A's


def test_10k_integrated_report_not_detected_for_a_normal_partial_failure():
    # A document using the standard "Item N." heading style that happens to
    # be missing one item must not be reinterpreted as an integrated
    # report -- detection requires zero item-label headings anywhere, not
    # "the item-label path failed at something."
    text = SYNTHETIC_10K.replace(
        "Item 7A. Quantitative and Qualitative Disclosures\n\nRates matter.\n\n", ""
    )
    result = extract_sections(text, "10-K")
    failed_items = {item for item, _ in result.failures}
    assert "7A" in failed_items
    assert dict(result.failures)["7A"] == "start marker not found"


SYNTHETIC_INTEGRATED_10Q = (
    "UNITED STATES SECURITIES AND EXCHANGE COMMISSION\n\n"
    "Table of Contents\n"
    "  Consolidated Condensed Financial Statements and Supplemental Details  \n"
    "  Notes to Consolidated Condensed Financial Statements               8  \n"
    "  Management's Discussion and Analysis (MD&A)                           \n"
    "  Operating Segments Trends and Results                              25  \n"
    "  Liquidity and Capital Resources                                    33  \n"
    "  Risk Factors and Other Key Information                                \n"
    "  Risk Factors                                                       35  \n"
    "  Form 10-Q Cross-Reference Index                                    38  \n"
    "\n"
    "Table of Contents\n\n"
    "Forward-Looking Statements\n\n"
    "This report contains forward-looking statements.\n\n"
    "Consolidated Condensed Financial Statements and Supplemental Details\n\n"
    "Balance sheet data follows.\n\n"
    "Notes to Consolidated Condensed Financial Statements\n\n"
    "Legal Proceedings\n\n"
    "We are party to various legal proceedings this quarter.\n\n"
    "Management's Discussion and Analysis (MD&A)\n\n"
    "Operating Segments Trends and Results\n\n"
    "Segment revenue increased this quarter.\n\n"
    "Liquidity and Capital Resources\n\n"
    "Cash flow remained strong.\n\n"
    "Risk Factors and Other Key Information\n\n"
    "Risk Factors\n\n"
    "The risks described in our most recent Form 10-K remain applicable.\n\n"
    "Form 10-Q Cross-Reference Index\n\n"
    "Item Number      Item\n"
    "Item 2.          Management's Discussion and Analysis   Page 25\n"
    "Item 1.          Legal Proceedings                      Page 8\n"
    "Item 1A.         Risk Factors                            Page 35\n"
)


def test_10q_integrated_report_recovers_all_three_targets():
    result = extract_sections(SYNTHETIC_INTEGRATED_10Q, "10-Q")
    assert result.failures == []
    got = {item: (s, e) for item, s, e in result.sections}
    assert set(got) == {"Part I Item 2", "Part II Item 1", "Part II Item 1A"}

    start, end = got["Part I Item 2"]
    body = SYNTHETIC_INTEGRATED_10Q[start:end]
    assert body.startswith("Operating Segments Trends and Results")
    assert "Cash flow remained strong" in body  # Liquidity, same chapter

    start, _ = got["Part II Item 1"]
    assert SYNTHETIC_INTEGRATED_10Q[start:].startswith("Legal Proceedings")


def test_8k_returns_whole_text_as_one_section():
    text = "NVIDIA Announces Financial Results\n\nRevenue of $44.1 billion.\n"
    result = extract_sections(text, "8-K")
    assert result.failures == []
    assert result.sections == [("EX-99.1", 0, len(text))]


def test_8k_empty_exhibit_text_is_a_failure_not_an_empty_section():
    result = extract_sections("   \n\n  ", "8-K")
    assert result.sections == []
    assert result.failures == [("EX-99.1", "exhibit text is empty")]


def test_flag_length_outliers_needs_enough_history_before_scoring():
    entries = [("acc-1", "NVDA", "1A", 1000), ("acc-2", "NVDA", "1A", 50000)]
    assert flag_length_outliers(entries) == []


def test_flag_length_outliers_catches_a_short_span_among_consistent_ones():
    entries = [
        ("acc-1", "NVDA", "1A", 48000),
        ("acc-2", "NVDA", "1A", 51000),
        ("acc-3", "NVDA", "1A", 49500),
        ("acc-4", "NVDA", "1A", 50500),
        ("acc-5", "NVDA", "1A", 300),  # boundary miss: only picked up a stub
    ]
    flagged = flag_length_outliers(entries)
    assert [accession for accession, _, _ in flagged] == ["acc-5"]


def test_flag_length_outliers_is_scoped_per_ticker_and_item():
    entries = [
        ("acc-1", "NVDA", "1A", 50000),
        ("acc-2", "NVDA", "1A", 50100),
        ("acc-3", "NVDA", "1A", 49900),
        ("acc-4", "NVDA", "1A", 50050),
        ("acc-5", "NVDA", "1A", 49950),
        # A bank's short Item 3 is normal for that item, not an outlier of
        # NVDA's unrelated 1A distribution.
        ("acc-6", "ZION", "3", 150),
        ("acc-7", "ZION", "3", 160),
        ("acc-8", "ZION", "3", 155),
        ("acc-9", "ZION", "3", 152),
        ("acc-10", "ZION", "3", 158),
    ]
    assert flag_length_outliers(entries) == []


# ---------------------------------------------------------------------------
# fixture-driven: real filing text, human-approved offsets
# ---------------------------------------------------------------------------


def _expected_cases() -> list[tuple[str, Path]]:
    if not EXPECTED_DIR.exists():
        return []
    return [(p.stem, p) for p in sorted(EXPECTED_DIR.glob("*.json"))]


CASES = _expected_cases()


def test_fixtures_pending_or_present():
    if not CASES:
        pytest.skip(
            "tests/fixtures/sections/expected/ is empty -- run "
            "scripts/make_extraction_fixture.py --all and get the "
            ".candidate.json files reviewed before this test has any cases"
        )


@pytest.mark.parametrize("accession,expected_path", CASES, ids=[c[0] for c in CASES])
def test_extraction_matches_approved_fixture(accession, expected_path):
    expected: dict[str, list[int]] = json.loads(expected_path.read_text())

    candidate_path = RAW_DIR / f"{accession}.candidate.json"
    text_path = RAW_DIR / f"{accession}.txt"
    assert candidate_path.exists(), (
        f"{accession}: expected/{accession}.json exists but "
        f"raw/{accession}.candidate.json is missing -- regenerate with "
        "scripts/make_extraction_fixture.py"
    )
    assert text_path.exists(), f"{accession}: raw/{accession}.txt is missing"

    form = json.loads(candidate_path.read_text())["form"]
    text = text_path.read_text()

    result = extract_sections(text, form)
    got = {item: (start, end) for item, start, end in result.sections}
    failure_reasons = dict(result.failures)

    for item, (exp_start, exp_end) in expected.items():
        assert item in got, (
            f"{accession} {item}: expected a span but extractor produced no "
            f"section (failure reason: {failure_reasons.get(item, 'item not attempted')})"
        )
        got_start, got_end = got[item]
        assert abs(got_start - exp_start) <= TOLERANCE_CHARS, (
            f"{accession} {item}: start {got_start} vs expected {exp_start}"
        )
        assert abs(got_end - exp_end) <= TOLERANCE_CHARS, (
            f"{accession} {item}: end {got_end} vs expected {exp_end}"
        )
