"""JSONL record shape and I/O for relevance judgments.

Shared by `scripts/judge.py` (writer), `scripts/agreement.py` and
`scripts/evaluate.py` (readers) so the on-disk shape is defined in exactly
one place. The files this module reads and writes -- everything under
`data/qrels/` -- are human-only. Nothing in this module fabricates a grade;
`grade=None` means the judge pressed skip, not that this module guessed.

Append-only, resumable: `latest_by_pair` resolves duplicate (query_id,
chunk_id) pairs by last-line-wins, so a `--redo` pass that re-presents an
already-judged pair does not require editing or truncating the file -- the
correction is a new line appended after the original.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True, slots=True)
class Judgment:
    query_id: str
    chunk_id: str
    grade: int | None  # None = judge pressed skip
    judged_at: str  # ISO 8601, UTC, e.g. now_iso()
    session_id: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_judgment(path: Path, judgment: Judgment) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(judgment)) + "\n")


def iter_judgments(path: Path) -> Iterator[Judgment]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield Judgment(**row)


def load_judgments(path: Path) -> list[Judgment]:
    return list(iter_judgments(path))


def latest_by_pair(judgments: Iterable[Judgment]) -> dict[tuple[str, str], Judgment]:
    """Last line wins per (query_id, chunk_id); relies on file append order,
    which is chronological since `append_judgment` only ever appends."""
    latest: dict[tuple[str, str], Judgment] = {}
    for judgment in judgments:
        latest[(judgment.query_id, judgment.chunk_id)] = judgment
    return latest


def load_qrels_dict(path: Path) -> dict[str, dict[str, int]]:
    """`{query_id: {chunk_id: grade}}` for ranx, skips excluded.

    A ranx `Qrels` cannot represent "judged and found not relevant" versus
    "never judged" -- both are simply absent -- so a skip and an ungraded
    pair look identical to the metric. That is correct: `evaluate.py` scores
    against what was actually judged, and a skip is not a judgment.
    """
    qrels: dict[str, dict[str, int]] = {}
    for judgment in latest_by_pair(iter_judgments(path)).values():
        if judgment.grade is None:
            continue
        qrels.setdefault(judgment.query_id, {})[judgment.chunk_id] = judgment.grade
    return qrels


def judged_pairs(path: Path) -> set[tuple[str, str]]:
    """Every (query_id, chunk_id) with a recorded outcome, grade or skip --
    used to decide what a resumed judging session can skip re-showing."""
    return set(latest_by_pair(iter_judgments(path)).keys())
