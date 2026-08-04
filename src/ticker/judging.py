"""Core judging-session logic, kept independent of terminal handling so it
is testable with a scripted sequence of keys instead of a real TTY.

`scripts/judge.py` supplies the real single-keystroke reader; everything
here takes `next_key` as a plain `Callable[[], str]` and never touches
`termios`/`tty` itself.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Sequence

import duckdb

from ticker.qrels import (
    Judgment,
    append_judgment,
    iter_judgments,
    judged_pairs,
    latest_by_pair,
    now_iso,
)

GRADE_KEYS = {"0": 0, "1": 1, "2": 2, "3": 3}
SKIP_KEY = "s"
QUIT_KEY = "q"
VALID_KEYS = set(GRADE_KEYS) | {SKIP_KEY, QUIT_KEY}

INSTRUCTIONS = {
    "standard": "relevant to the query",
    "novelty": "relevant AND new this period (not boilerplate carried over "
    "from a prior filing)",
}

GRADE_LEGEND = (
    "0 not relevant   1 marginally relevant   2 relevant   3 highly relevant"
    "   s skip   q quit"
)

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
HIT = "\033[1;33m"
OFF = "\033[0m"

# Words carrying no topical signal. Highlighting them would paint most of the
# passage and defeat the point, which is to make the relevant span findable
# without reading all 1.1k characters.
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the "
    "to was were will with".split()
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9']*")


def _stem(word: str) -> str:
    """Lowercase and drop a possessive or a plural `s`.

    Deliberately not a real stemmer. A query for "customer concentration risk"
    has to light up "customers" and "risks", and that is the whole of the
    morphology this needs. The `ss` guard keeps "business" from becoming
    "busines" and matching nothing.
    """
    word = word.lower()
    if word.endswith("'s"):
        word = word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word


def query_terms(query_text: str) -> frozenset[str]:
    return frozenset(
        stem
        for stem in (_stem(w) for w in _WORD_RE.findall(query_text))
        if stem and stem not in STOPWORDS
    )


def _is_hit(stem: str, terms: frozenset[str]) -> bool:
    for term in terms:
        if stem == term:
            return True
        # Prefix matching in both directions catches impairment/impair and
        # litigation/litigate. Floored at 5 characters because shorter prefixes
        # match across unrelated words.
        longer, shorter = (stem, term) if len(stem) > len(term) else (term, stem)
        if len(shorter) >= 5 and longer.startswith(shorter):
            return True
    return False


def highlight(text: str, terms: frozenset[str]) -> str:
    """Wrap query-matching words in `text` with the highlight escape.

    Applied after wrapping, never before. `_wrap` counts characters to find
    its break points, and an escape sequence inserted first would be counted
    as visible width, so every line would break short by the length of the
    codes it contains.
    """
    if not terms:
        return text

    def paint(match: re.Match[str]) -> str:
        word = match.group(0)
        return f"{HIT}{word}{OFF}" if _is_hit(_stem(word), terms) else word

    return _WORD_RE.sub(paint, text)


def rejudge_path(out_path: Path) -> Path:
    """standard.jsonl -> standard.rejudge.jsonl -- a sibling file, never
    merged into the primary qrels, per PLAN's two-qrel discipline extended
    to the rejudge pass."""
    return out_path.with_name(f"{out_path.stem}.rejudge{out_path.suffix}")


@dataclass(frozen=True, slots=True)
class ChunkDisplay:
    chunk_id: str
    text: str
    item: str
    ticker: str
    form: str
    filed_at: datetime
    period_end: date | None


def fetch_chunk_displays(
    con: duckdb.DuckDBPyConnection, chunk_ids: Sequence[str]
) -> dict[str, ChunkDisplay]:
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" for _ in chunk_ids)
    rows = con.execute(
        f"""
        SELECT c.chunk_id, c.text, sec.item, f.ticker, f.form, f.filed_at, f.period_end
        FROM chunks c
        JOIN sections sec ON c.section_id = sec.section_id
        JOIN filings f ON sec.accession = f.accession
        WHERE c.chunk_id IN ({placeholders})
        """,
        list(chunk_ids),
    ).fetchall()
    return {
        chunk_id: ChunkDisplay(
            chunk_id=chunk_id,
            text=text,
            item=item,
            ticker=ticker,
            form=form,
            filed_at=filed_at,
            period_end=period_end,
        )
        for chunk_id, text, item, ticker, form, filed_at, period_end in rows
    }


def sample_for_rejudge(
    qrels_path: Path, n: int, *, seed: int | None = None
) -> list[tuple[str, str]]:
    """Sample up to `n` distinct, already-graded (query_id, chunk_id) pairs
    for a second judging pass. Skips are excluded: re-judging "I couldn't
    grade this" tells you nothing about grading consistency."""
    graded = [
        pair
        for pair, judgment in latest_by_pair(iter_judgments(qrels_path)).items()
        if judgment.grade is not None
    ]
    if n >= len(graded):
        return graded
    return random.Random(seed).sample(graded, n)


def _wrap(text: str, width: int = 96) -> str:
    words = text.split()
    lines: list[str] = []
    line: list[str] = []
    length = 0
    for word in words:
        if length + len(word) + (1 if line else 0) > width:
            lines.append(" ".join(line))
            line, length = [], 0
        line.append(word)
        length += len(word) + (1 if len(line) > 1 else 0)
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


def _render(
    echo: Callable[[str], None],
    instruction: str,
    query_id: str,
    query_text: str,
    display: ChunkDisplay,
    query_progress: tuple[int, int],
    overall_progress: tuple[int, int],
) -> None:
    q_done, q_total = query_progress
    o_done, o_total = overall_progress
    pct = 100 * o_done / o_total if o_total else 0.0
    period = f"  period {display.period_end}" if display.period_end else ""
    echo("")
    echo(f"{BOLD}{CYAN}{query_id}{OFF}  {query_text}")
    echo(f"{DIM}judge for: {INSTRUCTIONS[instruction]}{OFF}")
    echo(
        f"{DIM}{display.ticker} {display.form} item {display.item}  "
        f"filed {display.filed_at.date()}{period}{OFF}"
    )
    echo(highlight(_wrap(display.text), query_terms(query_text)))
    echo(
        f"{DIM}this query {q_done + 1}/{q_total}   "
        f"overall {o_done + 1}/{o_total} ({pct:.0f}%){OFF}"
    )
    echo(f"[{GRADE_LEGEND}]")


@dataclass
class SessionSummary:
    judged_this_session: int
    grade_counts: Counter
    quit_early: bool
    total_pool_pairs: int
    already_judged_pairs: int


def run_judging_session(
    con: duckdb.DuckDBPyConnection,
    queries: Sequence[tuple[str, str]],
    pool: dict[str, list[str]],
    *,
    instruction: str,
    out_path: Path,
    session_id: str,
    next_key: Callable[[], str],
    echo: Callable[[str], None] = print,
    redo: bool = False,
) -> SessionSummary:
    """Walk `queries` in order, `pool[query_id]` in order, one chunk at a
    time. Writes one JSONL line per judged (or skipped) pair immediately,
    so killing the process loses at most the item on screen."""
    if instruction not in INSTRUCTIONS:
        raise ValueError(f"unknown instruction {instruction!r}, expected one of {list(INSTRUCTIONS)}")

    already = set() if redo else judged_pairs(out_path)
    per_query_done = Counter()
    for judgment in iter_judgments(out_path):
        per_query_done[judgment.query_id] += 1

    total_pool_pairs = sum(len(pool.get(qid, [])) for qid, _ in queries)
    already_judged_pairs = sum(
        1 for qid, _ in queries for cid in pool.get(qid, []) if (qid, cid) in already
    )

    done_this_session = 0
    grade_counts: Counter = Counter()
    quit_early = False

    for query_id, query_text in queries:
        if quit_early:
            break
        chunk_ids = pool.get(query_id, [])
        if not chunk_ids:
            continue
        displays = fetch_chunk_displays(con, chunk_ids)
        for chunk_id in chunk_ids:
            if (query_id, chunk_id) in already:
                continue
            display = displays.get(chunk_id)
            if display is None:
                echo(f"warning: {chunk_id} not in corpus, skipping")
                already.add((query_id, chunk_id))
                continue

            _render(
                echo,
                instruction,
                query_id,
                query_text,
                display,
                (per_query_done[query_id], len(chunk_ids)),
                (already_judged_pairs + done_this_session, total_pool_pairs),
            )

            key = next_key()
            while key not in VALID_KEYS:
                echo(f"  invalid key {key!r}, expected one of {sorted(VALID_KEYS)}")
                key = next_key()

            if key == QUIT_KEY:
                quit_early = True
                break

            grade = GRADE_KEYS.get(key)  # None for skip
            append_judgment(
                out_path,
                Judgment(
                    query_id=query_id,
                    chunk_id=chunk_id,
                    grade=grade,
                    judged_at=now_iso(),
                    session_id=session_id,
                ),
            )
            already.add((query_id, chunk_id))
            per_query_done[query_id] += 1
            done_this_session += 1
            grade_counts[key] += 1

    return SessionSummary(
        judged_this_session=done_this_session,
        grade_counts=grade_counts,
        quit_early=quit_early,
        total_pool_pairs=total_pool_pairs,
        already_judged_pairs=already_judged_pairs,
    )
