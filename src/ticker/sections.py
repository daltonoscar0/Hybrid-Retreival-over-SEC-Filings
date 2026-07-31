"""Item-boundary extraction for 10-K, 10-Q, and 8-K filing text.

The whole problem is telling a real section heading apart from three
look-alikes that show up in edgartools' plain-text rendering of a filing:

1. The table of contents, which lists every item name once near the top of
   the document, right-padded with spaces and a page number, all on a single
   line with no blank line before it.
2. Cross-references in prose ("Refer to “Item 1A. Risk Factors” for a
   discussion of ...") which sit mid-paragraph, immediately after a quote
   character, not at a paragraph break.
3. For 10-Qs, the same item number reused across Part I and Part II
   ("Item 1" is Financial Statements in Part I and Legal Proceedings in
   Part II).

A real heading in this text is reliably preceded by a blank line -- a
paragraph break -- and holds the item number and title alone on their own
line. TOC rows and inline references are not: TOC rows are separated from
each other by a single newline, and inline references are embedded in
running prose. Requiring the blank line before "Item N." and "Part N."
removes both look-alikes without a TOC-specific carve-out anywhere else in
this module.

The blank-line check is done as a separate, zero-width lookback after
matching a candidate line, not folded into the regex as a literal leading
"\n\n". Two headers can sit back to back with exactly one blank line between
them ("Item 6. [Reserved].\n\nItem 7. Management's Discussion...") and a
regex that consumes the blank line as part of matching the first heading
leaves nothing for the second heading's own leading-blank-line requirement
to match, silently dropping it.

Section end boundaries are always "wherever the next heading starts," found
in document order. This is deliberately naive about which item is supposed
to follow which: some 10-Qs, especially bank 10-Qs, present Item 2 (MD&A)
before Item 1 (Financial Statements) in document order even though the
numbering runs the other way, and next-heading-in-document-order handles
that correctly without a lookup table of "expected" item sequences.

Failures are returned, never raised, from `extract_sections` for anything
that is a property of one filing's text (a missing item, an unresolvable end
boundary, an unparseable Part structure for a 10-Q). A missing or duplicated
heading never falls back to "take the rest of the document" -- that produces
a section that looks complete and is silently wrong, which is exactly the
failure mode PLAN.md calls out. `extract_sections` only raises for a
programming error at its own boundary: an unrecognized `form` value.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field

# A candidate heading line: "Item N[Letter]. Title" alone on its line. The
# title is capped at 180 characters -- generously long enough for every
# observed 10-K/10-Q item title, including ADI's habit of appending a units
# parenthetical directly onto the Item 7 heading line ("Item 7. Management's
# Discussion and Analysis of Financial Condition and Results of Operations
# (all tabular amounts in thousands except per share amounts)", 145
# characters of title alone) -- and still short enough to reject a
# heading-shaped false match that runs on into a paragraph because the
# line-end regex failed to anchor. Whether it is a real heading (blank line
# above) is checked separately by `_preceded_by_blank_line`.
#
# `[.:]` rather than a literal period: HBAN's 10-Qs punctuate every Part I
# item and the first two Part II items with a colon ("Item 1: Financial
# Statements", "Item 1A: Risk Factors") while using a period for the rest of
# the same document ("Item 2. Unregistered Sales", "Item 6. Exhibits") --
# an inconsistency within one filer's own template, not two filers to
# distinguish between.
_ITEM_LINE_RE = re.compile(
    r"^[ \t]*(Item[ \t]+(\d{1,2}[A-C]?)[.:][ \t]+\S.{0,180})$",
    re.IGNORECASE | re.MULTILINE,
)

# The mirror image: "Title (Item N)" rather than "Item N. Title" -- FITB's
# own convention, e.g. "Legal Proceedings (Item 1)", "Risk Factors
# (Item 1A)". The title is captured non-greedily so the parenthetical stays
# anchored to the true end of the line rather than swallowing a later
# unrelated "(Item N)" mention on the same line.
_ITEM_LINE_SUFFIX_RE = re.compile(
    r"^[ \t]*(\S.{0,180}?[ \t]*\(Item[ \t]+(\d{1,2}[A-C]?)\))[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

# Some filings' HTML-to-text conversion splits the item letter from the
# number -- "Item 1A." renders as "Item 1. A" -- apparently a table-cell or
# superscript artifact in the source HTML. Matching this generally would be
# dangerous: "Item 3. A description of ..." is ordinary sentence-case prose,
# not a heading, and "A" is the most common indefinite article in English.
# Anchoring to the exact canonical Item 1A / 7A titles keeps this safe: those
# two phrases do not occur as the start of ordinary prose right after
# "Item N.", so there is no plausible false positive to guard against.
_SPLIT_LETTER_TITLES = {
    "1A": r"RISK\s+FACTORS",
    "7A": r"QUANTITATIVE\s+AND\s+QUALITATIVE",
}
_ITEM_LINE_SPLIT_RE = re.compile(
    r"^[ \t]*(Item[ \t]+(\d{1,2})\.[ \t]+([A-C])[ \t]+(?:"
    + "|".join(_SPLIT_LETTER_TITLES.values())
    + r").*)$",
    re.IGNORECASE | re.MULTILINE,
)

# Same idea for the "Part I" / "Part II" markers that disambiguate 10-Q item
# numbers reused across parts. Unlike item headers, the title is not
# required to be on the same line -- some filers put a glossary or other
# block directly under a bare "Part I" heading -- so this only anchors on
# the roman numeral itself. `I{1,2}\b` stops "Part III" (Directors /
# Compensation, in some 10-Ks) from matching as "Part I" or "Part II".
_PART_LINE_RE = re.compile(
    r"^[ \t]*(Part[ \t]+(I{1,2})\b\.?).*$",
    re.IGNORECASE | re.MULTILINE,
)


def _preceded_by_blank_line(text: str, line_start: int) -> bool:
    """True if the line immediately above `line_start` is blank or absent.

    `line_start` must be the position where a MULTILINE `^` matched, so it is
    either 0 or immediately after a newline. The line above runs from the
    newline before that back to the previous newline (or the start of text).
    """
    if line_start == 0:
        return False
    prev_end = line_start - 1
    prev_start = text.rfind("\n", 0, prev_end) + 1
    return text[prev_start:prev_end].strip() == ""


TEN_K_TARGET_ITEMS: tuple[str, ...] = ("1", "1A", "3", "7", "7A")
TEN_Q_PART1_TARGET_ITEMS: tuple[str, ...] = ("2",)
TEN_Q_PART2_TARGET_ITEMS: tuple[str, ...] = ("1", "1A")

EX99_ITEM = "EX-99.1"

# Per-item floor, below which a span is reported as a failure instead of
# emitted. PLAN.md section 2: "log extraction failures loudly rather than
# silently emitting a truncated section." Before this existed the extractor
# emitted a 95-character Item 7 as a valid MD&A, which is precisely that
# failure mode.
#
# Every number is read off the corpus's own length distribution, not chosen
# for roundness, and each sits below the shortest span the item legitimately
# has while still catching a heading with no body under it. The two kinds of
# item need very different floors and averaging them would defeat the check:
#
#   Narrative items -- 1, 1A, 7, Part I Item 2 -- have observed minima of
#   25,114 / 24,154 / 16,752 / 14,081 characters. A floor well under those is
#   still far above any boundary miss, which lands in the hundreds.
#
#   Cross-reference items -- 3, 7A, Part II Item 1, Part II Item 1A -- are
#   routinely satisfied by a single sentence pointing at a financial statement
#   note ("Reference is made to Note 20"), or by the word "None." Their
#   observed minima are 128 / 201 / 44 / 124 characters and those are complete
#   sections as filed, not truncations. The floor here only rules out a span
#   too short to hold a sentence at all. Filings whose Item 3 or 7A is a
#   pointer rather than prose are counted and reported separately in
#   reports/phase1_extraction.md, since a corpus of pointers is a real quality
#   limitation even though it is not an extraction defect.
MIN_SECTION_CHARS: dict[str, int] = {
    "1": 5000,
    "1A": 5000,
    "3": 100,
    "7": 2000,
    "7A": 150,
    "Part I Item 2": 2000,
    "Part II Item 1": 30,
    "Part II Item 1A": 100,
    EX99_ITEM: 500,
}


def _below_minimum(item: str, start: int, end: int) -> str | None:
    minimum = MIN_SECTION_CHARS.get(item)
    if minimum is None or (end - start) >= minimum:
        return None
    return (
        f"span is {end - start} characters, below the {minimum}-character "
        f"minimum for {item}; recorded as a failure rather than emitted, "
        "because a span this short for this item is a boundary miss and a "
        "short section that looks complete poisons everything downstream"
    )


def enforce_minimums(extraction: SectionExtraction) -> SectionExtraction:
    """Move every below-minimum span out of `sections` and into `failures`.

    Applied once at the end of `extract_sections`, after the repairs in
    `_repair_short_item7` have had their chance, so a span that a repair can
    rescue is rescued rather than reported.
    """
    kept: list[tuple[str, int, int]] = []
    failures = list(extraction.failures)
    for item, start, end in extraction.sections:
        reason = _below_minimum(item, start, end)
        if reason is None:
            kept.append((item, start, end))
        else:
            failures.append((item, reason))
    return SectionExtraction(sections=kept, failures=failures)


@dataclass(frozen=True, slots=True)
class SectionExtraction:
    sections: list[tuple[str, int, int]] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _Header:
    item: str
    start: int  # position of "Item"/"Part", start of the text a human would read as the heading
    end: int  # position right after the heading's trailing newline


def _find_headers(
    text: str, pattern: re.Pattern[str], item_of: Callable[[re.Match[str]], str]
) -> list[_Header]:
    headers = []
    for m in pattern.finditer(text):
        if not _preceded_by_blank_line(text, m.start()):
            continue
        headers.append(_Header(item=item_of(m), start=m.start(1), end=m.end()))
    return headers


def _find_item_headers(text: str) -> list[_Header]:
    plain = _find_headers(text, _ITEM_LINE_RE, lambda m: m.group(2).upper())
    split = _find_headers(
        text, _ITEM_LINE_SPLIT_RE, lambda m: f"{m.group(2)}{m.group(3)}".upper()
    )
    suffix = _find_headers(text, _ITEM_LINE_SUFFIX_RE, lambda m: m.group(2).upper())
    return sorted(plain + split + suffix, key=lambda h: h.start)


def _next_start(headers: list[_Header], after: _Header) -> int | None:
    """Position of the next heading in document order after `after`, or None."""
    for h in headers:
        if h.start > after.start:
            return h.start
    return None


def _resolve_in_order(
    headers: list[_Header], target_order: tuple[str, ...]
) -> tuple[dict[str, _Header], list[str]]:
    """Assign each item in `target_order` to a header, enforcing document order.

    `headers` must already be filtered to the range being searched (e.g. one
    Part of a 10-Q). For each target item in turn, take the first header with
    that item code whose position is after the previously resolved target --
    never before it. That makes a stray duplicate of an already-resolved
    item earlier in the text impossible to pick up by construction, while
    still finding a target that comes later even if an unrelated item
    (outside `target_order`) sits between them. An item with no header after
    the running position is reported missing rather than guessed at.
    Returns (resolved, missing_items).
    """
    by_item: dict[str, list[_Header]] = {}
    for h in headers:
        by_item.setdefault(h.item, []).append(h)

    resolved: dict[str, _Header] = {}
    missing: list[str] = []
    floor = -1
    for item in target_order:
        candidate = next((h for h in by_item.get(item, []) if h.start > floor), None)
        if candidate is None:
            missing.append(item)
            continue
        resolved[item] = candidate
        floor = candidate.start
    return resolved, missing


def _build_sections(
    text: str,
    all_headers: list[_Header],
    resolved: dict[str, _Header],
    missing: list[str],
    label: dict[str, str] | None = None,
) -> SectionExtraction:
    sections: list[tuple[str, int, int]] = []
    failures: list[tuple[str, str]] = []

    for item in missing:
        name = label.get(item, item) if label else item
        failures.append((name, "start marker not found"))

    for item, header in resolved.items():
        name = label.get(item, item) if label else item
        end = _next_start(all_headers, header)
        if end is None:
            failures.append(
                (name, "no closing boundary found after start marker; "
                       "refusing to take the rest of the document")
            )
            continue
        if end <= header.start:
            failures.append((name, "closing boundary precedes start marker"))
            continue
        sections.append((name, header.start, end))

    return SectionExtraction(sections=sections, failures=failures)


# ---------------------------------------------------------------------------
# Integrated-report fallback
#
# A minority of filers (Intel, and -- measured against the full 481-filing
# corpus, not just the fixture set -- HBAN, FITB, KEY, ZION among others)
# file a 10-K/10-Q whose body never uses a real "Item N." heading at all.
# Every canonical item number instead appears only where a human reader
# would look it up: Intel's trailing "Form 10-K/10-Q Cross-Reference Index",
# or the others' front-matter table of contents. Either way that index maps
# items to page numbers, and page numbers do not survive text extraction,
# so it gives no character offsets to work with. The item-label path above
# correctly finds nothing there and, unmodified, would report every target
# item as a plain "start marker not found" miss.
#
# The body is not unheaded, though. It uses the filer's own narrative
# chapter/subsection titles ("Risk Factors", "Legal Proceedings", "Our
# Products"...) as clean headings with exactly the structural signature the
# item-label path already keys on: unindented or lightly indented, a blank
# line (or a horizontal rule) above and below, no trailing page number.
# Those titles are listed, in document order, in the filing's own
# front-matter table of contents, which this fallback parses directly
# rather than hardcoding a per-year list of section names -- Intel's own
# fixtures show that list changing release to release (2020 and 2024 group
# the same items under different chapter names and different orderings).
#
# This path only ever engages when the item-label path -- covering every
# rendering variant it knows about, not just the plain "Item N." case --
# found zero real headings anywhere in the whole document. That is a narrow,
# explicit detector, not "try this whenever something failed": a filing
# that is merely missing one or two items still has other real item-label
# headings, so this never overrides a normal filing's own genuine,
# reportable miss.
#
# Item 3 (Legal Proceedings) is deliberately not attempted here beyond its
# start marker in the 10-K/10-Q paths below, for a reason worth recording:
# in both INTC fixture years, "Legal Proceedings" sits inside "Notes to
# Consolidated Financial Statements" as one footnote among many, and its own
# named sub-matters ("VLSI Technology LLC v. Intel", "R2 Semiconductor
# Patent Litigation", ...) use the exact same structural signature as a real
# section boundary. There is no textual signal in this document that
# distinguishes "the next litigation matter, still part of this note" from
# "the next unrelated footnote" -- so any automatic end boundary found this
# way would be a guess dressed up as a rule. The item-label path's own end
# boundary is reused here (next entry in the whitelist below), which is
# honest about being an approximation: it can only be as precise as the
# nearest TOC-listed heading that happens to follow, and is documented as
# such rather than presented as exact.
# ---------------------------------------------------------------------------

_TOC_ROW_RE = re.compile(r"^[ \t]*(\S.{0,100}?)[ \t]{2,}(?:\d{1,4}|[Pp]age)?[ \t]*$")


def _parse_front_toc_titles(text: str) -> list[str]:
    """Titles listed in the filing's own front-matter table of contents, in
    document order. The TOC lives between the first two occurrences of
    "Table of Contents": the filer repeats that heading once as the TOC's
    own caption and again immediately before the narrative body begins, so
    the text between them is exactly the TOC's row list.
    """
    first = re.search(r"Table of Contents", text, re.IGNORECASE)
    if first is None:
        return []
    second = re.search(r"Table of Contents", text[first.end():], re.IGNORECASE)
    if second is None:
        return []
    block = text[first.end():first.end() + second.start()]

    titles = []
    for line in block.split("\n"):
        m = _TOC_ROW_RE.match(line)
        if not m:
            continue
        title = m.group(1).strip()
        if title and title.lower() != "page":
            titles.append(title)
    return titles


def _is_blank_or_rule(line: str) -> bool:
    """Blank, or a horizontal-rule line of dashes/underscores only -- some
    headings in this format are followed by a rule before the body text
    starts, not directly by a blank line.
    """
    stripped = line.strip()
    if stripped == "":
        return True
    return all(c in "-─━_" for c in stripped)


def _find_title_heading(text: str, title: str) -> int | None:
    """First position of `title` as a real, standalone body heading:
    case-insensitive exact line match, blank line (or rule) above and
    below. The exact-line match alone already excludes a trailing page
    number, which every TOC row and running-footer decoy carries and a real
    heading never does.
    """
    pattern = re.compile(r"^[ \t]*" + re.escape(title) + r"[ \t]*$", re.IGNORECASE | re.MULTILINE)
    for m in pattern.finditer(text):
        if not _preceded_by_blank_line(text, m.start()):
            continue
        next_nl = text.find("\n", m.end())
        if next_nl == -1:
            continue
        after = text[next_nl + 1:]
        next_line_end = after.find("\n")
        next_line = after[:next_line_end] if next_line_end != -1 else after
        if not _is_blank_or_rule(next_line):
            continue
        return m.start()
    return None


def _toc_whitelist_headers(text: str, toc_titles: list[str]) -> list[_Header]:
    """Every TOC-listed title that actually renders as a real body heading,
    as `_Header`s sorted by position. This is the boundary set for the
    fallback: real headings the filer's own TOC vouches for, none of which
    a sub-heading inside one item's own narrative (a named risk factor, a
    named litigation matter, an ESG sub-topic) could be mistaken for, since
    those are not TOC-listed.
    """
    headers = []
    seen: set[int] = set()
    for title in toc_titles:
        pos = _find_title_heading(text, title)
        if pos is not None and pos not in seen:
            headers.append(_Header(item=title, start=pos, end=pos + len(title)))
            seen.add(pos)
    return sorted(headers, key=lambda h: h.start)


def _is_integrated_report(item_label_headers: list[_Header]) -> bool:
    """True when the item-label path (covering "Item N.", "Item N:", and
    "Title (Item N)", every rendering variant seen across the corpus) found
    zero real headings anywhere in the whole document.

    This used to also require a "cross-reference index" heading nearby, on
    the theory that item numbers would be clustered at the end of the
    document the way Intel's are. Measured against the full corpus, that
    was overfit to Intel specifically: HBAN, FITB, KEY, ZION and others
    share the same "no item-label heading anywhere in the body" shape but
    list their item numbers only in the front-matter table of contents, not
    a trailing index. "Zero item-label headings anywhere" is itself already
    a narrow, rare signal on a regex proven to find real headings reliably
    everywhere else in the corpus -- it does not need a second, more
    specific corroborating signal to be trustworthy.
    """
    return not item_label_headers


def _resolve_chapter_first_rendered_subsection(
    text: str, toc_titles: list[str], chapter_pattern: re.Pattern[str]
) -> tuple[str, int] | None:
    """Find `chapter_pattern`'s chapter in the TOC, then walk its listed
    subsections in order and return the first one that actually renders as
    a body heading. Some listed subsections (an image-only "Introduction to
    Our Business" spread, observed in both INTC fixture years) never render
    as extractable text at all, so treating "the chapter's first listed
    subsection" as a fixed title is not safe -- this tries each in turn.
    """
    chapter_idx = next(
        (i for i, t in enumerate(toc_titles) if chapter_pattern.search(t)), None
    )
    if chapter_idx is None:
        return None
    for title in toc_titles[chapter_idx + 1:]:
        pos = _find_title_heading(text, title)
        if pos is not None:
            return title, pos
    return None


_BUSINESS_CHAPTER_RE = re.compile(r"Fundamentals of Our Business", re.IGNORECASE)
_MDNA_CHAPTER_RE = re.compile(r"Management.s Discussion and Analysis", re.IGNORECASE)

# Direct canonical-title matches: present verbatim in the TOC and reused
# verbatim as the real body heading, in both INTC fixture years, so no
# chapter/next-subsection indirection is needed for these. "about"/"About"
# is covered by the case-insensitive match in `_find_title_heading`.
_TEN_K_INTEGRATED_DIRECT_TITLES: dict[str, str] = {
    "1A": "Risk Factors",
    "3": "Legal Proceedings",
    "7A": "Quantitative and Qualitative Disclosures about Market Risk",
}
_TEN_Q_INTEGRATED_DIRECT_TITLES: dict[str, str] = {
    "P2:1": "Legal Proceedings",
    "P2:1A": "Risk Factors",
}


def _build_integrated_sections(
    text: str,
    whitelist: list[_Header],
    chapter_span: dict[str, _Header],
    chapter_span_missing: list[str],
    single: dict[str, _Header],
    single_missing: list[str],
    label: dict[str, str] | None = None,
) -> SectionExtraction:
    """Two different boundary rules for two different kinds of resolved item.

    `chapter_span` items (1, 7, and the 10-Q's Part I Item 2) are resolved
    to their chapter's *first* rendered subsection, but their true content
    runs through every subsection of that chapter, not just the first --
    "Fundamentals of Our Business" also holds "Our Strategy" and "Our
    Capital" after "A Year in Review". Bounding these against the full TOC
    whitelist stops at the very next subsection and truncates almost all of
    the item's real content, so these are bounded against the other
    *target* anchors only, which correctly skips past a chapter's own
    interior subsections and stops at whichever target -- another chapter
    or a single-subsection item nested inside this one, like 7A sitting
    inside the MD&A chapter in the 2020 fixture -- comes next.

    `single` items (1A, 3, 7A, and the 10-Q's Part II items) are resolved
    directly to one named subsection with no children of their own, so the
    full whitelist (every TOC-listed heading plus the other targets) is the
    right, tighter boundary: it stops at the very next real heading,
    whichever kind it is.
    """
    all_targets = list(chapter_span.values()) + list(single.values())

    chapter_result = _build_sections(
        text, sorted(all_targets, key=lambda h: h.start),
        chapter_span, chapter_span_missing, label=label,
    )
    single_result = _build_sections(
        text, sorted(whitelist + all_targets, key=lambda h: h.start),
        single, single_missing, label=label,
    )
    return SectionExtraction(
        sections=chapter_result.sections + single_result.sections,
        failures=chapter_result.failures + single_result.failures,
    )


def _extract_10k_integrated_fallback(text: str) -> SectionExtraction:
    toc_titles = _parse_front_toc_titles(text)
    whitelist = _toc_whitelist_headers(text, toc_titles)

    chapter_span: dict[str, _Header] = {}
    chapter_span_missing: list[str] = []
    for item, chapter_pattern in (("1", _BUSINESS_CHAPTER_RE), ("7", _MDNA_CHAPTER_RE)):
        found = _resolve_chapter_first_rendered_subsection(text, toc_titles, chapter_pattern)
        if found is None:
            chapter_span_missing.append(item)
        else:
            chapter_span[item] = _Header(item=item, start=found[1], end=found[1])

    single: dict[str, _Header] = {}
    single_missing: list[str] = []
    for item, title in _TEN_K_INTEGRATED_DIRECT_TITLES.items():
        pos = _find_title_heading(text, title)
        if pos is None:
            single_missing.append(item)
        else:
            single[item] = _Header(item=item, start=pos, end=pos)

    return _build_integrated_sections(
        text, whitelist, chapter_span, chapter_span_missing, single, single_missing
    )


def _extract_10q_integrated_fallback(text: str) -> SectionExtraction:
    toc_titles = _parse_front_toc_titles(text)
    whitelist = _toc_whitelist_headers(text, toc_titles)

    chapter_span: dict[str, _Header] = {}
    chapter_span_missing: list[str] = []
    found = _resolve_chapter_first_rendered_subsection(text, toc_titles, _MDNA_CHAPTER_RE)
    if found is None:
        chapter_span_missing.append("P1:2")
    else:
        chapter_span["P1:2"] = _Header(item="P1:2", start=found[1], end=found[1])

    single: dict[str, _Header] = {}
    single_missing: list[str] = []
    for item, title in _TEN_Q_INTEGRATED_DIRECT_TITLES.items():
        pos = _find_title_heading(text, title)
        if pos is None:
            single_missing.append(item)
        else:
            single[item] = _Header(item=item, start=pos, end=pos)

    label = {
        "P1:2": "Part I Item 2",
        "P2:1": "Part II Item 1",
        "P2:1A": "Part II Item 1A",
    }
    return _build_integrated_sections(
        text, whitelist, chapter_span, chapter_span_missing, single, single_missing,
        label=label,
    )


# ---------------------------------------------------------------------------
# Short-Item-7 repairs
#
# Item 7 is the only target item that is never legitimately short. Measured
# over the 111 extractable 10-Ks in the corpus, its length distribution has a
# hard gap: seven filings land between 95 and 480 characters, the eighth is
# 16,752, and the median is 74,971. Nothing sits in between. Both ends of that
# gap are a different structural problem and neither is a regex bug, so both
# are repaired here rather than by loosening the heading patterns.
#
# The two repairs are tried in order, cheapest first, and only when Item 7's
# own span falls under `MIN_SECTION_CHARS["7"]`. If neither applies, Item 7 is
# reported as a failure. A short Item 7 is never emitted as valid.
# ---------------------------------------------------------------------------

# Repair 1, joint presentation. Some filers put the Item 7 and Item 7A
# headings back to back and then run one combined narrative under both
# (observed: RF's 2021 10-K, where the two headings are 95 characters apart
# and the whole 352,647-character MD&A lands under 7A). The next-heading end
# rule then gives Item 7 the 95 characters between the two headings and files
# the entire MD&A under Market Risk. That is a misattribution, not a
# truncation: no text is lost, it is filed under the wrong item, which would
# put a full MD&A into the Phase 5.2 "novelty by item type" table as Market
# Risk. The combined span is emitted under Item 7 and Item 7A is reported as
# jointly presented rather than emitted with the same offsets, because two
# sections sharing a span would double every sentence in it through the
# chunker and into both language models.

# Repair 2, incorporation by reference into the F-pages. Comerica's six 10-Ks
# satisfy Item 7 with a pointer: "Reference is made to the sections entitled
# ... on pages F-4 through F-39 of the Financial Section of this report." The
# narrative is in the same document, after an index block that maps each
# financial-section title to its F-page. The anchor is that index block's
# position, not any string in the pointer: the pointer's own section names
# roll forward year to year ("2019 Overview and 2020 Outlook" becomes "2024
# Overview"), and the index block's rendering picks up stray spaces inside the
# page numbers ("F- 4", "F -3") and inside the years ("20 20 Overview"), so
# matching the pointer text verbatim breaks on five of the six years.
#
# Everything below anchors by position -- first match after a known offset --
# never by rank. "Last match in the document" would break the moment a
# heading repeats inside an exhibit.

_FPAGE_INDEX_ROW_RE = re.compile(
    r"^[ \t]*(\S.{0,90}?)[ \t]{2,}F[ \t]*-[ \t]*\d[\d \t]{0,4}[ \t]*$",
    re.MULTILINE,
)

# An index block is a run of F-page rows with no large gap between them. Both
# numbers are structural, not tuned: index rows are one line each, so 400
# characters is several lines of slack and still far below the 9,000-plus
# characters of Performance Graph and Selected Financial Data content that
# separates the real block from anything else shaped like it, and the
# front-matter table of contents contributes at most one isolated F-page row
# ("FINANCIAL REVIEW AND REPORTS  F-1"), which five rows rules out.
_MAX_INDEX_ROW_GAP = 400
_MIN_INDEX_ROWS = 5

# The first financial-section heading the Item 7 pointer incorporates. Keyed
# on the four-digit-year-plus-Overview shape rather than a literal year, per
# the roll-forward above. Starting here rather than at the first body heading
# after the index block matters: the two headings before it, Performance Graph
# and Selected Financial Data, are what Item 6's own pointer incorporates, and
# handing Item 6's content to Item 7 would be a boundary error dressed up as a
# fix.
_YEAR_OVERVIEW_HEADING_RE = re.compile(r"^[ \t]*\d{4}[ \t]*OVERVIEW\b.*$", re.MULTILINE | re.IGNORECASE)

# Where the incorporated narrative stops: the first financial statement or
# audit-report heading after it. All three spellings are matched because
# filers disagree on which they use and the earliest one wins regardless
# ("Report of Management" is Comerica's; "Management's Report" is the phrasing
# elsewhere in the corpus). Each must be a standalone heading line, not a
# substring: "Consolidated Balance Sheet" appears 43 times inside Comerica's
# own MD&A prose, and a substring match would cut the section at the first
# sentence that mentions the balance sheet.
_INCORPORATED_END_RES = (
    re.compile(r"^[ \t]*Reports? of Independent\b.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^[ \t]*Consolidated Balance Sheets?[ \t]*$", re.MULTILINE | re.IGNORECASE),
    re.compile(
        r"^[ \t]*(?:Report of Management|Management.s Report)[ \t]*$",
        re.MULTILINE | re.IGNORECASE,
    ),
)


def _find_fpage_index_block(text: str, after: int) -> tuple[int, int] | None:
    """Span of the first F-page index block starting after `after`, or None."""
    run: list[re.Match[str]] = []
    for match in _FPAGE_INDEX_ROW_RE.finditer(text):
        if match.start() <= after:
            continue
        if run and match.start() - run[-1].end() > _MAX_INDEX_ROW_GAP:
            if len(run) >= _MIN_INDEX_ROWS:
                return run[0].start(), run[-1].end()
            run = []
        run.append(match)
    if len(run) >= _MIN_INDEX_ROWS:
        return run[0].start(), run[-1].end()
    return None


def _first_heading_after(text: str, pattern: re.Pattern[str], after: int) -> int | None:
    for match in pattern.finditer(text):
        if match.start() > after and _preceded_by_blank_line(text, match.start()):
            return match.start()
    return None


def _incorporated_item7_span(text: str, stub_start: int) -> tuple[int, int] | None:
    block = _find_fpage_index_block(text, stub_start)
    if block is None:
        return None
    start = _first_heading_after(text, _YEAR_OVERVIEW_HEADING_RE, block[1])
    if start is None:
        return None
    ends = [
        position
        for position in (
            _first_heading_after(text, pattern, start) for pattern in _INCORPORATED_END_RES
        )
        if position is not None
    ]
    if not ends:
        return None
    return start, min(ends)


def _repair_short_item7(
    text: str, sections: list[tuple[str, int, int]]
) -> tuple[list[tuple[str, int, int]], list[tuple[str, str]]]:
    """Replace a below-minimum Item 7 span with the repaired one, if either
    repair applies. Returns (sections, extra_failures)."""
    by_item = {item: (start, end) for item, start, end in sections}
    if "7" not in by_item:
        return sections, []
    start, end = by_item["7"]
    if end - start >= MIN_SECTION_CHARS["7"]:
        return sections, []

    joint = by_item.get("7A")
    if joint is not None and joint[0] == end and joint[1] - start >= MIN_SECTION_CHARS["7"]:
        repaired = [
            (item, s, e) for item, s, e in sections if item not in ("7", "7A")
        ] + [("7", start, joint[1])]
        return sorted(repaired, key=lambda row: row[1]), [
            (
                "7A",
                "presented jointly with Item 7: the two headings are adjacent and "
                "one combined narrative follows both. The combined span is emitted "
                "under Item 7 and is not repeated here, because two sections over "
                "the same offsets would double every sentence in it.",
            )
        ]

    incorporated = _incorporated_item7_span(text, start)
    if incorporated is not None:
        # This repair moves Item 7 to a span far later in the document, past
        # every other item's start marker, while leaving the other items where
        # the item-label path put them. Nothing in that path guarantees one of
        # them does not run into the new span: an item whose own end boundary
        # is the next item-label heading will overrun if there is no such
        # heading between it and the F-pages. Two sections over overlapping
        # offsets would insert the same sentences twice, into the chunker, both
        # indexes, and both language models.
        #
        # Measured over all 966 filings this does not happen, so the guard
        # never fires today. It is here because "measured not to happen" is a
        # property of this corpus and the repair is a property of the code.
        # Overlapping items are reported rather than silently trimmed, since a
        # section that overruns into the financial statements has a wrong end
        # boundary whether or not Item 7 moved.
        new_start, new_end = incorporated
        kept: list[tuple[str, int, int]] = [("7", new_start, new_end)]
        overlaps: list[tuple[str, str]] = []
        for item, section_start, section_end in sections:
            if item == "7":
                continue
            if section_start < new_end and new_start < section_end:
                overlaps.append((
                    item,
                    f"span {section_start}-{section_end} overlaps the Item 7 "
                    f"narrative recovered at {new_start}-{new_end}; dropped "
                    "rather than emitted, because two sections over the same "
                    "offsets would double every sentence between them",
                ))
                continue
            kept.append((item, section_start, section_end))
        return sorted(kept, key=lambda row: row[1]), overlaps

    return sections, []


def _extract_10k(text: str) -> SectionExtraction:
    headers = _find_item_headers(text)
    if _is_integrated_report(headers):
        return _extract_10k_integrated_fallback(text)
    resolved, missing = _resolve_in_order(headers, TEN_K_TARGET_ITEMS)
    built = _build_sections(text, headers, resolved, missing)
    sections, extra_failures = _repair_short_item7(text, built.sections)
    return SectionExtraction(
        sections=sections, failures=built.failures + extra_failures
    )


def _extract_10q(text: str) -> SectionExtraction:
    item_headers = _find_item_headers(text)
    if _is_integrated_report(item_headers):
        return _extract_10q_integrated_fallback(text)
    part_headers = sorted(
        _find_headers(text, _PART_LINE_RE, lambda m: m.group(2).upper()),
        key=lambda h: h.start,
    )

    # Last, not first: when a filer's front-matter caption reads "TABLE OF
    # CONTENTS" followed by a *blank* line before the TOC rows begin (rather
    # than the single newline most filers use), the TOC's own "Part I:
    # Financial Information" / "Part II: Other Information" caption rows are
    # themselves blank-line-preceded and indistinguishable from a real
    # heading by that rule alone (observed: ON's 10-Qs). The real Part I and
    # Part II transitions are always the *last* occurrence of each in the
    # document; any earlier one is the TOC's own caption row. Part I must
    # also precede Part II, guarding against a stray later cross-reference
    # ("as discussed in Part I, Item 1...") being mistaken for the marker.
    part2 = next((h for h in reversed(part_headers) if h.item == "II"), None)
    part1 = next(
        (h for h in reversed(part_headers)
         if h.item == "I" and (part2 is None or h.start < part2.start)),
        None,
    )

    all_target = list(TEN_Q_PART1_TARGET_ITEMS) + [
        f"P2:{i}" for i in TEN_Q_PART2_TARGET_ITEMS
    ]
    label = {i: f"Part I Item {i}" for i in TEN_Q_PART1_TARGET_ITEMS}
    label.update(
        {f"P2:{i}": f"Part II Item {i}" for i in TEN_Q_PART2_TARGET_ITEMS}
    )

    if part1 is None or part2 is None or part2.start <= part1.start:
        reason = (
            f"could not locate a clean Part I / Part II split "
            f"(found Part I={'yes' if part1 else 'no'}, "
            f"Part II={'yes' if part2 else 'no'})"
        )
        failures = [(label[i], reason) for i in all_target]
        return SectionExtraction(sections=[], failures=failures)

    part1_items = [h for h in item_headers if part1.start <= h.start < part2.start]
    part2_items = [h for h in item_headers if h.start >= part2.start]

    resolved1, missing1 = _resolve_in_order(part1_items, TEN_Q_PART1_TARGET_ITEMS)
    resolved2, missing2 = _resolve_in_order(part2_items, TEN_Q_PART2_TARGET_ITEMS)

    resolved = dict(resolved1)
    resolved.update({f"P2:{k}": v for k, v in resolved2.items()})
    missing = list(missing1) + [f"P2:{m}" for m in missing2]

    return _build_sections(text, item_headers, resolved, missing, label=label)


def _extract_8k(text: str) -> SectionExtraction:
    if not text.strip():
        return SectionExtraction(
            sections=[], failures=[(EX99_ITEM, "exhibit text is empty")]
        )
    return SectionExtraction(sections=[(EX99_ITEM, 0, len(text))], failures=[])


def extract_sections(text: str, form: str) -> SectionExtraction:
    normalized = form.strip().upper()
    if normalized == "10-K":
        return enforce_minimums(_extract_10k(text))
    if normalized == "10-Q":
        return enforce_minimums(_extract_10q(text))
    if normalized == "8-K":
        return enforce_minimums(_extract_8k(text))
    raise ValueError(f"extract_sections: unsupported form {form!r}")


def flag_length_outliers(
    entries: list[tuple[str, str, str, int]], z_thresh: float = 2.5
) -> list[tuple[str, str, str]]:
    """Flag sections whose length is a statistical outlier for the same
    firm and item across other filings in `entries`.

    `entries` is (accession, ticker, item, length). A span whose length is
    wildly different from the same item, same firm, across years is almost
    always a boundary miss, not real variation in how much a company wrote
    -- PLAN.md's stated check. Requires at least 4 other observations for
    the same (ticker, item) before scoring, because a z-score computed on
    fewer than a handful of points is noise, not signal, and would flag the
    smaller group's legitimate spread as an outlier.

    Each point is scored against the mean/stdev of the *other* points in its
    group (leave-one-out), not the group including itself. A population
    z-score that includes the outlier lets a single wild value inflate its
    own stdev enough to mask itself -- exactly the failure mode this check
    exists to catch, and worse the smaller the group, which is the common
    case here (roughly one observation per firm per year).

    Returns (accession, item, reason) tuples suitable for merging into a
    failure report.
    """
    by_group: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for accession, ticker, item, length in entries:
        by_group.setdefault((ticker, item), []).append((accession, length))

    flagged: list[tuple[str, str, str]] = []
    for (ticker, item), rows in by_group.items():
        if len(rows) < 5:
            continue
        lengths = [length for _, length in rows]
        for i, (accession, length) in enumerate(rows):
            others = lengths[:i] + lengths[i + 1:]
            mean = statistics.mean(others)
            stdev = statistics.pstdev(others)
            if stdev == 0:
                continue
            z = (length - mean) / stdev
            if abs(z) >= z_thresh:
                flagged.append((
                    accession,
                    item,
                    f"length {length} is a {z:.1f}-sigma outlier against the "
                    f"other {len(others)} {ticker} {item} filings (their mean "
                    f"{mean:.0f}); likely a boundary miss, not real variation",
                ))
    return flagged
