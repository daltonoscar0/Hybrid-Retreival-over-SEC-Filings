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

import functools
import json
import re
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

_FILLER_SENTENCE = "This sentence is filler so the section clears its minimum length."


def _pad(lead: str, target: int) -> str:
    """`lead`, then plain filler on the following line, to `target` characters.

    These documents exist to exercise boundary rules, not lengths, but
    `extract_sections` refuses to emit a span under `MIN_SECTION_CHARS[item]`.
    Padding keeps them testing the shipped contract rather than a variant of
    it with the length gate switched off, which is the version that let a
    95-character Item 7 through in the first place.

    The filler goes on the line after `lead`, separated by a single newline
    and never a blank one, so it cannot read as a heading: every heading rule
    in `ticker.sections` requires a blank line above. It also carries no item
    number, no parenthetical, no F-page, and no double space, so it cannot
    trip the item-suffix, F-page-index, or table-of-contents patterns either.
    """
    if len(lead) >= target:
        return lead
    repeats = -(-(target - len(lead)) // (len(_FILLER_SENTENCE) + 1))
    return lead + "\n" + " ".join([_FILLER_SENTENCE] * repeats)


ITEM_1_BODY = _pad("We make widgets. See “Item 1A. Risk Factors” for more.", 6000)
ITEM_1A_BODY = _pad("Widgets may break.", 6000)
ITEM_3_BODY = _pad("None pending.", 200)
ITEM_7_BODY = _pad("Revenue grew, as discussed in “Item 1A. Risk Factors” above.", 3000)
ITEM_7A_BODY = _pad("Rates matter.", 300)

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
    f"{ITEM_1_BODY}\n\n"
    "Item 1A. Risk Factors\n\n"
    f"{ITEM_1A_BODY}\n\n"
    "Item 2. Properties\n\n"
    "One factory.\n\n"
    "Item 3. Legal Proceedings\n\n"
    f"{ITEM_3_BODY}\n\n"
    "Item 6. [Reserved]\n\n"
    "Item 7. Management's Discussion and Analysis\n\n"
    f"{ITEM_7_BODY}\n\n"
    "Item 7A. Quantitative and Qualitative Disclosures\n\n"
    f"{ITEM_7A_BODY}\n\n"
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
        f"Item 7A. Quantitative and Qualitative Disclosures\n\n{ITEM_7A_BODY}\n\n", ""
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
        f"Item 1A. Risk Factors\n\n{ITEM_1A_BODY}",
        f"Item 1. A Risk Factors\n\n{ITEM_1A_BODY}",
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
        f"Item 3. Legal Proceedings\n\n{ITEM_3_BODY}",
        _pad("Item 3. A previously disclosed matter was settled.", 200),
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
    f"{_pad('Results improved.', 3000)}\n\n"
    "Item 3. Quantitative and Qualitative Disclosures\n\n"
    "Rates matter.\n\n"
    "Item 4. Controls and Procedures\n\n"
    "Effective.\n\n"
    "Part II. Other Information\n\n"
    "Item 1. Legal Proceedings\n\n"
    f"{_pad('None pending.', 120)}\n\n"
    "Item 1A. Risk Factors\n\n"
    f"{_pad('No material changes.', 200)}\n\n"
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
    f"{_pad('Widget demand grew this year.', 3000)}\n\n"
    "Our Strategy\n\n"
    f"{_pad('We aim to lead the market.', 3000)}\n\n"
    "Management's Discussion and Analysis\n\n"
    "Our Products\n\n"
    f"{_pad('We sell widgets and gadgets.', 1500)}\n\n"
    "Segment Trends and Results\n\n"
    f"{_pad('Widget segment revenue increased.', 1500)}\n\n"
    "Risk Factors and Other Key Information\n\n"
    "Risk Factors\n\n"
    f"{_pad('Our business faces various risks.', 6000)}\n\n"
    "Quantitative and Qualitative Disclosures about Market Risk\n\n"
    f"{_pad('We are exposed to interest rate risk.', 300)}\n\n"
    "Financial Statements and Supplemental Details\n\n"
    "Notes to Consolidated Financial Statements\n\n"
    "Legal Proceedings\n\n"
    f"{_pad('We are party to various legal proceedings.', 200)}\n\n"
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
        f"Item 7A. Quantitative and Qualitative Disclosures\n\n{ITEM_7A_BODY}\n\n", ""
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
    f"{_pad('We are party to various legal proceedings this quarter.', 120)}\n\n"
    "Management's Discussion and Analysis (MD&A)\n\n"
    "Operating Segments Trends and Results\n\n"
    f"{_pad('Segment revenue increased this quarter.', 1500)}\n\n"
    "Liquidity and Capital Resources\n\n"
    f"{_pad('Cash flow remained strong.', 1500)}\n\n"
    "Risk Factors and Other Key Information\n\n"
    "Risk Factors\n\n"
    f"{_pad('The risks described in our most recent Form 10-K remain applicable.', 300)}\n\n"
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
    text = (
        "NVIDIA Announces Financial Results\n\n"
        f"{_pad('Revenue of $44.1 billion.', 900)}\n"
    )
    result = extract_sections(text, "8-K")
    assert result.failures == []
    assert result.sections == [("EX-99.1", 0, len(text))]


def test_8k_empty_exhibit_text_is_a_failure_not_an_empty_section():
    result = extract_sections("   \n\n  ", "8-K")
    assert result.sections == []
    assert result.failures == [("EX-99.1", "exhibit text is empty")]


# ---------------------------------------------------------------------------
# minimum section length: a short span is a failure, never a valid section
# ---------------------------------------------------------------------------


def test_short_narrative_item_is_a_failure_not_a_valid_section():
    text = SYNTHETIC_10K.replace(ITEM_1_BODY, "We make widgets.")
    result = extract_sections(text, "10-K")
    assert "1" not in {item for item, _, _ in result.sections}
    reason = dict(result.failures)["1"]
    assert "below the 5000-character minimum" in reason
    # The measured length belongs in the reason, so the failure table says how
    # short rather than only that it was short.
    assert re.search(r"span is \d+ characters", reason)


def test_cross_reference_item_at_its_legitimate_floor_is_still_valid():
    # Item 3's shortest real span in the corpus is 128 characters: a single
    # sentence pointing at a financial statement note. That is a complete
    # section as filed, not a truncation, and a floor that rejected it would
    # be deleting real corpus rather than catching a miss.
    body = (
        "Reference is made to Note 20, Contingent Liabilities, of the "
        "Consolidated Financial Statements in this report."
    )
    text = SYNTHETIC_10K.replace(ITEM_3_BODY, body)
    result = extract_sections(text, "10-K")
    got = {item: (start, end) for item, start, end in result.sections}
    assert "3" in got
    assert got["3"][1] - got["3"][0] < 200


def test_minimum_is_per_item_not_one_global_threshold():
    # The same length is fine for Item 3 and a boundary miss for Item 7.
    short = "Reference is made to the Financial Section of this report."
    text = SYNTHETIC_10K.replace(ITEM_3_BODY, _pad(short, 150))
    text = text.replace(ITEM_7_BODY, _pad(short, 150))
    result = extract_sections(text, "10-K")
    got = {item for item, _, _ in result.sections}
    assert "3" in got
    assert "7" not in got


def test_below_minimum_span_is_never_silently_dropped():
    # Rejecting the span is only half the contract; PLAN section 2 asks for
    # the failure to be loud. Every rejected item must appear in `failures`.
    text = SYNTHETIC_10K.replace(ITEM_1A_BODY, "Widgets may break.")
    result = extract_sections(text, "10-K")
    assert "1A" in {item for item, _ in result.failures}


# ---------------------------------------------------------------------------
# short-Item-7 repairs: joint presentation, and incorporation by reference
# into the F-pages of the same document
# ---------------------------------------------------------------------------


def _joint_presentation_10k(mdna_body: str) -> str:
    """A 10-K whose Item 7 and Item 7A headings sit back to back with one
    combined narrative under both, the shape of RF's 2021 10-K."""
    return SYNTHETIC_10K.replace(
        f"Item 7. Management's Discussion and Analysis\n\n"
        f"{ITEM_7_BODY}\n\n"
        f"Item 7A. Quantitative and Qualitative Disclosures\n\n"
        f"{ITEM_7A_BODY}\n\n",
        f"Item 7. Management's Discussion and Analysis\n\n"
        f"Item 7A. Quantitative and Qualitative Disclosures\n\n"
        f"{mdna_body}\n\n",
    )


def test_jointly_presented_item_7_and_7a_put_the_narrative_under_item_7():
    body = _pad("EXECUTIVE OVERVIEW. Revenue grew this year.", 40000)
    result = extract_sections(_joint_presentation_10k(body), "10-K")
    got = {item: (start, end) for item, start, end in result.sections}
    assert "7" in got
    start, end = got["7"]
    assert end - start > 40000
    assert "Revenue grew this year" in _joint_presentation_10k(body)[start:end]


def test_joint_presentation_does_not_emit_7a_over_the_same_span():
    # Two sections sharing offsets would double every sentence in the
    # combined narrative through the chunker and into both language models.
    body = _pad("EXECUTIVE OVERVIEW. Revenue grew this year.", 40000)
    result = extract_sections(_joint_presentation_10k(body), "10-K")
    assert "7A" not in {item for item, _, _ in result.sections}
    assert "presented jointly with Item 7" in dict(result.failures)["7A"]


def _incorporated_10k(year: int, *, index_rows: int = 8) -> str:
    """A 10-K that satisfies Item 7 with a pointer into the F-pages of the
    same document, the shape of all six of Comerica's 10-Ks.

    `index_rows` controls how many F-page rows the index block carries, so a
    test can starve the block below the minimum run length.
    """
    rows = [
        ("Performance Graph", 2),
        ("Selected Financial Data", 3),
        (f"{year} Overview", 4),
        ("Results of Operations", 6),
        ("Risk Management", 20),
        ("Critical Accounting Policies", 34),
        ("Forward-Looking Statements", 38),
        ("Consolidated Balance Sheets", 40),
    ][:index_rows]
    index_block = "".join(f"  {title:<58}F-{page}   \n" for title, page in rows)

    return (
        SYNTHETIC_10K.replace(
            f"Item 7. Management's Discussion and Analysis\n\n{ITEM_7_BODY}\n\n",
            "Item 7. Management's Discussion and Analysis\n\n"
            "Reference is made to the sections entitled "
            f"“{year} Overview,” “Results of Operations,” “Risk Management,” "
            "“Critical Accounting Policies” and “Forward-Looking Statements” "
            f"on pages F-4 through F-39 of the Financial Section of this report.\n\n",
        )
        + "\nFINANCIAL REVIEW AND REPORTS\n\n"
        + "Widget Bancorp and Subsidiaries\n\n"
        + index_block
        + "\nF-1\n\n"
        + "PERFORMANCE GRAPH\n\n"
        + _pad("The graph compares total returns against two indices.", 4000)
        + "\n\nSELECTED FINANCIAL DATA\n\n"
        + _pad("Five year summary of selected financial data.", 4000)
        + f"\n\n{year} OVERVIEW\n\n"
        + _pad(
            "Net income rose. As shown in the Consolidated Balance Sheets "
            "above, total assets grew.",
            40000,
        )
        + "\n\nRISK MANAGEMENT\n\n"
        + _pad("Credit risk is managed through underwriting standards.", 20000)
        + "\n\nCONSOLIDATED BALANCE SHEETS\n\n"
        + _pad("Total assets 90,000. Total liabilities 80,000.", 5000)
        + "\n\nREPORT OF MANAGEMENT\n\n"
        + _pad("Management is responsible for the financial statements.", 2000)
        + "\n"
    )


def test_incorporation_by_reference_recovers_item_7_from_the_f_pages():
    text = _incorporated_10k(2019)
    result = extract_sections(text, "10-K")
    got = {item: (start, end) for item, start, end in result.sections}
    assert "7" in got
    start, end = got["7"]
    assert end - start > 50000
    body = text[start:end]
    assert "Net income rose" in body
    assert "Credit risk is managed" in body


def test_incorporation_by_reference_survives_the_year_rolling_forward():
    # The pointer's own section names change every year: "2019 Overview and
    # 2020 Outlook" becomes "2024 Overview". Anchoring on a literal year
    # would pass on one filing and fail on the next five.
    for year in (2019, 2021, 2024):
        text = _incorporated_10k(year)
        got = {item: (s, e) for item, s, e in extract_sections(text, "10-K").sections}
        assert "7" in got, year
        assert got["7"][1] - got["7"][0] > 50000, year


def test_incorporation_by_reference_skips_the_headings_item_6_incorporates():
    # Performance Graph and Selected Financial Data are what Item 6's own
    # pointer incorporates. Starting at the first body heading after the
    # index block would hand Item 6's content to Item 7.
    text = _incorporated_10k(2019)
    start, end = next((s, e) for item, s, e in extract_sections(text, "10-K").sections if item == "7")
    body = text[start:end]
    assert "The graph compares total returns" not in body
    assert "Five year summary" not in body


def test_incorporation_end_boundary_ignores_the_phrase_inside_prose():
    # "Consolidated Balance Sheets" appears 43 times inside Comerica's own
    # MD&A prose. Only the standalone heading may end the section.
    text = _incorporated_10k(2019)
    start, end = next((s, e) for item, s, e in extract_sections(text, "10-K").sections if item == "7")
    body = text[start:end]
    assert "As shown in the Consolidated Balance Sheets" in body
    assert "Total assets 90,000" not in body


def test_incorporation_by_reference_requires_a_real_index_block():
    # One stray F-page row in a table of contents is not an index block. With
    # nothing to anchor on, Item 7 is reported as a failure rather than
    # guessed at.
    text = _incorporated_10k(2019, index_rows=2)
    result = extract_sections(text, "10-K")
    assert "7" not in {item for item, _, _ in result.sections}
    assert "7" in {item for item, _ in result.failures}


def test_repairs_do_not_fire_on_an_item_7_that_is_already_long_enough():
    result = extract_sections(SYNTHETIC_10K, "10-K")
    got = {item: (start, end) for item, start, end in result.sections}
    real_item_7 = SYNTHETIC_10K.index("Item 7. Management's Discussion")
    real_item_7a = SYNTHETIC_10K.index("Item 7A. Quantitative")
    assert got["7"] == (real_item_7, real_item_7a)
    assert "7A" in got


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
# real-filing regression for the two short-Item-7 repairs
#
# These read the raw text dumps, not human-approved offsets, so they assert a
# magnitude rather than an exact span: an Item 7 that recovers the F-page
# narrative is six figures of characters, and one that did not is three. No
# reviewer judgment is needed to tell those apart, which is why this can be a
# test today while the offset fixtures are still pending review.
#
# The corpus cache under data/raw/ is not committed (278 MB), so the full
# six-year Comerica sweep only runs where it exists and skips elsewhere. The
# committed fixture dumps under tests/fixtures/sections/raw/ carry whichever
# Comerica and Regions years the fixture manifest pins, and those run
# everywhere.
# ---------------------------------------------------------------------------

CACHE_DIR = Path("data/raw")


@functools.lru_cache(maxsize=1)
def _cache_index() -> tuple[tuple[str, str, str, Path], ...]:
    """(ticker, form, accession, text_path) for every complete cache entry.

    Cached because the three tests below would otherwise each walk all 967
    metadata files, and the walk costs more than the extraction they exist to
    check.
    """
    if not CACHE_DIR.exists():
        return ()
    rows = []
    for meta_path in sorted(CACHE_DIR.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        text_path = meta_path.with_suffix(".txt")
        if "ticker" in meta and "form" in meta and text_path.exists():
            rows.append((meta["ticker"], meta["form"], meta["accession"], text_path))
    return tuple(rows)


def _cached_10k_texts(ticker: str) -> list[tuple[str, str]]:
    """(accession, text) for every cached 10-K of `ticker`, oldest first."""
    return [
        (accession, path.read_text())
        for tk, form, accession, path in _cache_index()
        if tk == ticker and form == "10-K"
    ]


def _item_length(text: str, item: str) -> int | None:
    for name, start, end in extract_sections(text, "10-K").sections:
        if name == item:
            return end - start
    return None


def test_every_cached_comerica_10k_recovers_its_f_page_mdna():
    filings = _cached_10k_texts("CMA")
    if not filings:
        pytest.skip("data/raw/ is not populated here; run scripts/download_corpus.py")
    assert len(filings) == 6, f"expected 6 Comerica 10-Ks in the cache, found {len(filings)}"
    for accession, text in filings:
        length = _item_length(text, "7")
        assert length is not None, f"{accession}: Item 7 was not extracted at all"
        assert length > 50_000, f"{accession}: Item 7 is only {length} characters"


def test_regions_2021_mdna_is_filed_under_item_7_not_market_risk():
    filings = dict(_cached_10k_texts("RF"))
    text = filings.get("0001281761-21-000012")
    if text is None:
        pytest.skip("data/raw/ is not populated here; run scripts/download_corpus.py")
    got = {item: (s, e) for item, s, e in extract_sections(text, "10-K").sections}
    assert got["7"][1] - got["7"][0] > 300_000
    assert "7A" not in got


def test_the_repairs_change_nothing_for_a_filer_that_never_needed_them():
    # NVDA's 10-Ks extract cleanly on the item-label path. If a repair fired
    # on one of them it would be stealing a correct extraction.
    filings = _cached_10k_texts("NVDA")
    if not filings:
        pytest.skip("data/raw/ is not populated here; run scripts/download_corpus.py")
    for accession, text in filings:
        got = {item for item, _, _ in extract_sections(text, "10-K").sections}
        assert got == {"1", "1A", "3", "7", "7A"}, accession


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
