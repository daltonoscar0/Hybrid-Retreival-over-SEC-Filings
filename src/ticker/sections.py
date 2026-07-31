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


def _extract_10k(text: str) -> SectionExtraction:
    headers = _find_item_headers(text)
    if _is_integrated_report(headers):
        return _extract_10k_integrated_fallback(text)
    resolved, missing = _resolve_in_order(headers, TEN_K_TARGET_ITEMS)
    return _build_sections(text, headers, resolved, missing)


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
        return _extract_10k(text)
    if normalized == "10-Q":
        return _extract_10q(text)
    if normalized == "8-K":
        return _extract_8k(text)
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
