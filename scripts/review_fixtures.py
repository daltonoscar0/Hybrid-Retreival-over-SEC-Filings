"""Review candidate section spans and write the approved ones as fixtures.

The extractor proposes; a human disposes. This script only presents evidence
and records a verdict. It never proposes a span of its own and never edits an
offset, because tests/fixtures/sections/expected/ is the ground truth the
extractor is scored against, and ground truth a machine wrote scores the
machine against itself.

What it shows per item is chosen for the two ways a span goes wrong. A bad
start shows up as head text that does not begin at the item heading. A bad end
shows up as trailing text that runs past the next heading, which is why the
context after the span matters more than the last line inside it.

    uv run python scripts/review_fixtures.py            # every unreviewed accession
    uv run python scripts/review_fixtures.py --accession 0001045810-25-000023
    uv run python scripts/review_fixtures.py --redo     # include already-approved

Keys: y approve, n reject, s skip, a approve the rest of this filing, q quit.
Progress is written after each filing, so quitting midway loses nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sections"
RAW = FIXTURES / "raw"
EXPECTED = FIXTURES / "expected"

HEAD_CHARS = 400
TAIL_CHARS = 240
AFTER_CHARS = 240
BEFORE_CHARS = 120

BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
OFF = "\033[0m"


def _wrap(text: str, width: int = 96) -> str:
    """Collapse runs of blank lines so a 400-char excerpt stays on screen."""
    lines: list[str] = []
    blank = 0
    for raw_line in text.splitlines():
        stripped = raw_line.rstrip()
        if not stripped.strip():
            blank += 1
            if blank > 1:
                continue
            lines.append("")
            continue
        blank = 0
        while len(stripped) > width:
            cut = stripped.rfind(" ", 0, width)
            cut = cut if cut > width // 2 else width
            lines.append(stripped[:cut])
            stripped = stripped[cut:].lstrip()
        lines.append(stripped)
    return "\n".join(lines)


def _prompt(valid: str) -> str:
    while True:
        try:
            key = input(f"  [{valid}] > ").strip().lower()
        except EOFError:
            return "q"
        if key in valid:
            return key
        print(f"  {RED}expected one of {valid}{OFF}")


def _show(item: str, text: str, start: int, end: int) -> None:
    length = end - start
    print(f"\n{BOLD}{CYAN}Item {item}{OFF}  chars {start}-{end}  ({length:,} long)")

    before = text[max(0, start - BEFORE_CHARS):start]
    if before.strip():
        print(f"{DIM}...{_wrap(before)}{OFF}")

    print(f"{GREEN}>>> SPAN STARTS{OFF}")
    print(_wrap(text[start:start + HEAD_CHARS]))

    if length > HEAD_CHARS + TAIL_CHARS:
        print(f"{DIM}      [ ... {length - HEAD_CHARS - TAIL_CHARS:,} chars omitted ... ]{OFF}")
        print(_wrap(text[end - TAIL_CHARS:end]))
    print(f"{GREEN}>>> SPAN ENDS{OFF}")

    after = text[end:end + AFTER_CHARS]
    if after.strip():
        print(f"{DIM}{_wrap(after)}...{OFF}")
    else:
        print(f"{YELLOW}  (nothing follows: span runs to end of document){OFF}")


def review_accession(candidate_path: Path, approved: dict) -> str:
    meta = json.loads(candidate_path.read_text())
    accession = meta["accession"]
    text_path = RAW / f"{accession}.txt"
    if not text_path.exists():
        print(f"{RED}{accession}: raw text dump missing, skipping{OFF}")
        return "next"
    text = text_path.read_text()

    print(f"\n{BOLD}{'=' * 74}{OFF}")
    print(f"{BOLD}{meta['ticker']} {meta['form']} {accession}{OFF}  "
          f"({meta['text_length']:,} chars)")
    for item, reason in meta.get("failures", []):
        print(f"  {YELLOW}extractor reported no span for {item}: {reason}{OFF}")
    print(f"{BOLD}{'=' * 74}{OFF}")

    verdicts: dict[str, list[int]] = {}
    approve_rest = False
    for item, (start, end) in meta["sections"].items():
        if approve_rest:
            verdicts[item] = [start, end]
            continue
        _show(item, text, start, end)
        key = _prompt("ynsaq")
        if key == "q":
            return "quit"
        if key == "y":
            verdicts[item] = [start, end]
        elif key == "a":
            verdicts[item] = [start, end]
            approve_rest = True
        elif key == "n":
            print(f"  {RED}rejected, item omitted from the fixture{OFF}")

    if verdicts:
        EXPECTED.mkdir(parents=True, exist_ok=True)
        out = EXPECTED / f"{accession}.json"
        out.write_text(json.dumps(verdicts, indent=2) + "\n")
        approved[accession] = len(verdicts)
        print(f"\n  {GREEN}wrote {out.relative_to(Path.cwd())} "
              f"({len(verdicts)} of {len(meta['sections'])} spans){OFF}")
    else:
        print(f"\n  {YELLOW}nothing approved, no fixture written{OFF}")
    return "next"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accession", action="append", default=[])
    parser.add_argument("--redo", action="store_true",
                        help="include accessions that already have a fixture")
    args = parser.parse_args()

    candidates = sorted(RAW.glob("*.candidate.json"))
    if args.accession:
        wanted = set(args.accession)
        candidates = [p for p in candidates
                      if p.name.removesuffix(".candidate.json") in wanted]
    if not args.redo:
        candidates = [p for p in candidates
                      if not (EXPECTED / f"{p.name.removesuffix('.candidate.json')}.json").exists()]

    if not candidates:
        print("nothing to review. --redo revisits approved fixtures.")
        return 0

    print(f"{len(candidates)} filing(s) to review. "
          "Check that each span starts at its item heading and ends before the next one.")

    approved: dict[str, int] = {}
    for i, path in enumerate(candidates, 1):
        print(f"\n{DIM}filing {i} of {len(candidates)}{OFF}")
        if review_accession(path, approved) == "quit":
            print("\nstopped. Rerun to continue where you left off.")
            break

    if approved:
        total = sum(approved.values())
        print(f"\n{GREEN}{BOLD}approved {total} spans across "
              f"{len(approved)} filing(s){OFF}")
        print("run: uv run pytest tests/test_section_extraction.py -q")
    return 0


if __name__ == "__main__":
    sys.exit(main())
