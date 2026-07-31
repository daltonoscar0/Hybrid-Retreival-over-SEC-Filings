#!/usr/bin/env python3
"""Blind novelty spot-check: PLAN.md validation 5.3.

100 sentences, ten from each novelty decile, labeled novel or boilerplate by
a human who cannot see the score. Session logic is plain functions in this
module; only `_read_key` and `main` touch the terminal, so the whole flow is
driven in `tests/test_spotcheck.py` by a scripted key sequence. That is the
same split `scripts/judge.py` and `ticker.judging` use, for the same reason:
a labeling tool that can only be exercised by a human at a real TTY is a
labeling tool nobody checks.

One keystroke per sentence, no Enter. n novel, b boilerplate, s skip, q quit.
Every keypress writes its line before the next sentence is drawn, so killing
the process loses at most the item on screen and re-running resumes exactly
where it stopped. 100 labels should be twenty minutes, not an afternoon.

Why the screen shows the sentence and nothing else
--------------------------------------------------
Blind means blind to the score, and the score is not the only thing that
leaks it. Rank and decile are the obvious leaks, so neither is rendered.
Position is the next one, so the ten strata are shuffled together before
presentation: a labeler who noticed the first ten items were all flat
boilerplate would start answering from position rather than from the text.

Filing metadata is the subtle leak. Validation 5.2 reports mean novelty by
item type, so a labeler who can see "Item 1A" carries a prior about risk
factors straight into the 5.3 labels and 5.3 stops being an independent
check on 5.2. Ticker, form, item and filing date are therefore all absent
from the screen. What is left is the sentence, the instruction, the progress
counter and the key legend. Deciding whether a sentence reads as templated
language is a judgment about the language, and the language is on screen.

This tool also never prints an agreement number. Seeing the running
agreement mid-pass would make the rest of the pass unblind; computing it is
a separate step over the finished file.

Input contract
--------------
`--scores` is a JSONL file, one object per line:

    {"sentence_id": "<id from the sentences table>", "novelty": <float>}

`novelty` is the raw contrast, firm-prior surprisal minus background-prior
surprisal. Not the within-document z-score: deciles are cut across the whole
file, so the values have to be comparable across documents, and project
invariant 3 says the z-scored form is not. Extra keys on a line are ignored,
a repeated `sentence_id` is an error, and a non-finite `novelty` is an error.

The Phase 4 scorer writes this file. This tool does not import it and has no
opinion on how the numbers were produced, which is why it runs today against
a file written by hand.

No `as_of` here. Nothing in this module fits a model or computes a statistic
that scores a document. It ranks a file of numbers that already exist and
cuts them into ten equal-count strata. The strict-`<` time discipline lives
in the scorer that wrote the file. An `as_of` parameter here would do
nothing and would falsely suggest the filter was applied at this step.

Usage:
  uv run python scripts/spotcheck.py --scores data/novelty/scores.jsonl
  -> reads sentence text from data/ticker.duckdb
  -> appends to data/spotcheck/labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import termios
import textwrap
import tty
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import duckdb

from ticker.qrels import now_iso

DEFAULT_DB_PATH = Path("data/ticker.duckdb")
DEFAULT_OUT = Path("data/spotcheck/labels.jsonl")
DEFAULT_PER_DECILE = 10
DECILES = 10

NOVEL_KEY = "n"
BOILERPLATE_KEY = "b"
SKIP_KEY = "s"
QUIT_KEY = "q"
LABEL_KEYS = {NOVEL_KEY: "novel", BOILERPLATE_KEY: "boilerplate"}
VALID_KEYS = set(LABEL_KEYS) | {SKIP_KEY, QUIT_KEY}

# Deliberately the novelty half of ticker.judging.INSTRUCTIONS["novelty"],
# minus its relevance clause. There is no query here, so "relevant AND" does
# not apply, but the boilerplate test the labeler applies must be the same
# one they applied while judging, or the two label sets are not comparable.
INSTRUCTION = "new this period (not boilerplate carried over from a prior filing)"

LABEL_LEGEND = "n novel   b boilerplate   s skip   q quit"
PROGRESS_PREFIX = "spot-check "
WRAP_WIDTH = 96

DIM = "\033[2m"
OFF = "\033[0m"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def visible_text(rendered: str) -> str:
    """`rendered` with the colour escapes removed, i.e. what a labeler reads.

    Exists so the blind guarantee is assertable. An escape sequence is not
    text on screen but it is text in the string, and `\\033[2m` contains a
    digit; a test checking that no digit of the score reaches the screen has
    to be able to tell the two apart.
    """
    return _ANSI_RE.sub("", rendered)


@dataclass(frozen=True, slots=True)
class ScoredSentence:
    sentence_id: str
    novelty: float


@dataclass(frozen=True, slots=True)
class SpotcheckItem:
    sentence_id: str
    text: str
    decile: int  # 1 = lowest novelty, 10 = highest. Never rendered.


@dataclass(frozen=True, slots=True)
class SpotcheckLabel:
    sentence_id: str
    label: str | None  # None = the labeler pressed skip
    decile: int
    labeled_at: str  # ISO 8601, UTC
    session_id: str


def load_scores(path: Path) -> list[ScoredSentence]:
    """Read the `--scores` JSONL. See the module docstring for the contract.

    Duplicates raise rather than dedupe. Deciles are rank-based over the
    whole file, so a sentence counted twice shifts every boundary below it;
    that is a bug in the scorer and silently repairing it here would hide it.
    """
    scored: list[ScoredSentence] = []
    seen: set[str] = set()
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "sentence_id" not in row or "novelty" not in row:
                raise ValueError(
                    f"{path}:{lineno} needs both 'sentence_id' and 'novelty', got "
                    f"keys {sorted(row)}"
                )
            sentence_id = str(row["sentence_id"])
            novelty = float(row["novelty"])
            if sentence_id in seen:
                raise ValueError(f"{path}:{lineno} duplicate sentence_id {sentence_id!r}")
            if novelty != novelty or novelty in (float("inf"), float("-inf")):
                raise ValueError(f"{path}:{lineno} non-finite novelty {row['novelty']!r}")
            seen.add(sentence_id)
            scored.append(ScoredSentence(sentence_id=sentence_id, novelty=novelty))
    return scored


def assign_deciles(scored: Sequence[ScoredSentence]) -> dict[str, int]:
    """Rank-based deciles, 1 lowest novelty to 10 highest.

    Rank-based rather than ten equal-width bins on the value: the novelty
    contrast piles up near zero with a long right tail, so equal-width bins
    would put nearly everything in one or two of them and the stratified
    sample would draw almost nothing from the tail this check exists to
    look at. Ties break on sentence_id so a tie straddling a boundary lands
    in the same place on every run.
    """
    ordered = sorted(scored, key=lambda s: (s.novelty, s.sentence_id))
    n = len(ordered)
    if n == 0:
        return {}
    return {s.sentence_id: min(DECILES, i * DECILES // n + 1) for i, s in enumerate(ordered)}


def stratified_sample(
    scored: Sequence[ScoredSentence], *, per_decile: int, seed: int
) -> list[tuple[str, int]]:
    """`per_decile` sentence_ids from each decile, returned in presentation
    order, which is the ten strata shuffled together.

    The shuffle is the reason the decile cannot be read off the screen
    position. Both the draw and the shuffle run off one seeded RNG, so the
    same score file and seed reproduce the same session, which is what makes
    a killed run resumable.
    """
    deciles = assign_deciles(scored)
    by_decile: dict[int, list[str]] = {d: [] for d in range(1, DECILES + 1)}
    for sentence_id, decile in deciles.items():
        by_decile[decile].append(sentence_id)

    rng = random.Random(seed)
    chosen: list[tuple[str, int]] = []
    for decile in range(1, DECILES + 1):
        members = sorted(by_decile[decile])  # deterministic pre-sample order
        take = min(per_decile, len(members))
        chosen.extend((sentence_id, decile) for sentence_id in rng.sample(members, take))
    rng.shuffle(chosen)
    return chosen


def fetch_sentence_texts(
    con: duckdb.DuckDBPyConnection, sentence_ids: Sequence[str]
) -> dict[str, str]:
    if not sentence_ids:
        return {}
    placeholders = ",".join("?" for _ in sentence_ids)
    rows = con.execute(
        f"SELECT sentence_id, text FROM sentences WHERE sentence_id IN ({placeholders})",
        list(sentence_ids),
    ).fetchall()
    return {sentence_id: text for sentence_id, text in rows}


def build_sample(
    con: duckdb.DuckDBPyConnection,
    scored: Sequence[ScoredSentence],
    *,
    per_decile: int = DEFAULT_PER_DECILE,
    seed: int = 0,
    echo: Callable[[str], None] = print,
) -> list[SpotcheckItem]:
    """The presentation-ordered sample, with text joined from the corpus.

    A short decile or a sentence_id the corpus does not have is reported
    loudly and the sample comes back smaller. Quietly topping up from a
    neighbouring decile would break the equal-per-decile stratification that
    5.3's agreement number is computed against.
    """
    chosen = stratified_sample(scored, per_decile=per_decile, seed=seed)
    texts = fetch_sentence_texts(con, [sentence_id for sentence_id, _ in chosen])

    items = [
        SpotcheckItem(sentence_id=sentence_id, text=texts[sentence_id], decile=decile)
        for sentence_id, decile in chosen
        if sentence_id in texts
    ]

    missing = len(chosen) - len(items)
    if missing:
        echo(
            f"warning: {missing} sampled sentence_ids are not in the corpus and were "
            "dropped; the score file and the database are out of sync"
        )
    drawn = Counter(decile for _, decile in chosen)
    short = [d for d in range(1, DECILES + 1) if drawn[d] < per_decile]
    if short:
        echo(
            f"warning: deciles {short} had fewer than {per_decile} sentences to draw "
            "from; the sample is not evenly stratified"
        )
    return items


def append_label(path: Path, label: SpotcheckLabel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(label)) + "\n")


def iter_labels(path: Path) -> Iterator[SpotcheckLabel]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield SpotcheckLabel(**json.loads(line))


def load_labels(path: Path) -> list[SpotcheckLabel]:
    return list(iter_labels(path))


def labeled_ids(path: Path) -> set[str]:
    """Every sentence_id with a recorded outcome, label or skip -- what a
    resumed session can stop re-showing."""
    return {label.sentence_id for label in iter_labels(path)}


def render_item(item: SpotcheckItem, progress: tuple[int, int]) -> str:
    """The whole screen for one sentence, returned as a string.

    Returns text where `ticker.judging._render` echoes line by line, so a
    test can assert on everything the labeler sees in one place. The blind
    guarantee is only worth what the test that checks it is worth, and that
    test needs the screen.
    """
    done, total = progress
    pct = 100 * done / total if total else 0.0
    return "\n".join(
        [
            "",
            textwrap.fill(item.text, width=WRAP_WIDTH),
            f"{DIM}judge for: {INSTRUCTION}{OFF}",
            f"{DIM}{PROGRESS_PREFIX}{done + 1}/{total} ({pct:.0f}%){OFF}",
            f"[{LABEL_LEGEND}]",
        ]
    )


@dataclass
class SpotcheckSummary:
    labeled_this_session: int
    label_counts: Counter
    quit_early: bool
    total_items: int
    already_labeled: int


def run_spotcheck_session(
    items: Sequence[SpotcheckItem],
    *,
    out_path: Path,
    session_id: str,
    next_key: Callable[[], str],
    echo: Callable[[str], None] = print,
    redo: bool = False,
) -> SpotcheckSummary:
    """Walk `items` in presentation order, one sentence per screen, writing
    one JSONL line per keypress before drawing the next."""
    already = set() if redo else labeled_ids(out_path)
    total = len(items)
    already_labeled = sum(1 for item in items if item.sentence_id in already)

    done_this_session = 0
    label_counts: Counter = Counter()
    quit_early = False

    for item in items:
        if item.sentence_id in already:
            continue

        echo(render_item(item, (already_labeled + done_this_session, total)))

        key = next_key()
        while key not in VALID_KEYS:
            echo(f"  invalid key {key!r}, expected one of {sorted(VALID_KEYS)}")
            key = next_key()

        if key == QUIT_KEY:
            quit_early = True
            break

        label = LABEL_KEYS.get(key)  # None for skip
        append_label(
            out_path,
            SpotcheckLabel(
                sentence_id=item.sentence_id,
                label=label,
                decile=item.decile,
                labeled_at=now_iso(),
                session_id=session_id,
            ),
        )
        already.add(item.sentence_id)
        done_this_session += 1
        label_counts[label or "skip"] += 1

    return SpotcheckSummary(
        labeled_this_session=done_this_session,
        label_counts=label_counts,
        quit_early=quit_early,
        total_items=total,
        already_labeled=already_labeled,
    )


def _read_key() -> str:
    """One character, no Enter, restoring terminal settings on exit."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if ch in ("\x03", "\x04"):  # ctrl-c / ctrl-d
        return "q"
    return ch.lower()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--scores", type=Path, required=True,
        help="JSONL of {sentence_id, novelty}; see the module docstring for the contract",
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-decile", type=int, default=DEFAULT_PER_DECILE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--session-id", default=None)
    parser.add_argument(
        "--redo", action="store_true",
        help="re-present sentences already labeled in the output file instead of skipping them",
    )
    args = parser.parse_args()

    if not args.scores.exists():
        print(
            f"{args.scores} does not exist. This tool needs a JSONL file of "
            '{"sentence_id", "novelty"} from the Phase 4 novelty scorer; pass its '
            "path with --scores."
        )
        return

    scored = load_scores(args.scores)
    if not scored:
        print(f"{args.scores} has no scored sentences, nothing to spot-check")
        return

    session_id = args.session_id or uuid.uuid4().hex[:12]

    # Read-only: the corpus is a shared artifact and this tool has no reason
    # to hold a write lock on it while a human reads sentences for an hour.
    con = duckdb.connect(str(args.db), read_only=True)
    try:
        items = build_sample(con, scored, per_decile=args.per_decile, seed=args.seed)
    finally:
        con.close()

    if not items:
        print("no sampled sentences resolved to corpus text, nothing to spot-check")
        return

    print(f"instruction: {INSTRUCTION}")
    print(f"writing to {args.out}   session {session_id}")
    print(f"keys: {LABEL_LEGEND} -- no Enter needed")
    print("the score, its decile and its rank are not shown, by design\n")

    summary = run_spotcheck_session(
        items,
        out_path=args.out,
        session_id=session_id,
        next_key=_read_key,
        redo=args.redo,
    )

    print("\n" + "-" * 40)
    print(f"labeled this session: {summary.labeled_this_session}")
    if summary.label_counts:
        counts = ", ".join(f"{k}:{v}" for k, v in sorted(summary.label_counts.items()))
        print(f"label counts this session: {counts}")
    done = summary.already_labeled + summary.labeled_this_session
    print(f"total labeled: {done}/{summary.total_items}")
    if summary.quit_early:
        print("stopped early -- rerun to resume where you left off")
    print("no agreement number is printed here; reading it mid-pass would unblind the rest")


if __name__ == "__main__":
    main()
