#!/usr/bin/env python3
"""Render the report tables and figures into one static page.

No web framework, per invariant 6. This reads the markdown reports that the
pipeline already writes and emits a single self-contained HTML file with the
figures inlined as data URIs, so the page can be opened from disk or dropped
anywhere without carrying an asset directory.

Tables come from the reports rather than from a fresh computation. A page that
recomputed its own numbers could disagree with the reports, and then there
would be two answers and no way to tell which one the README quoted.

Usage:
  uv run python scripts/render_html.py
  open reports/index.html
"""

from __future__ import annotations

import argparse
import base64
import html
import re
from pathlib import Path

DEFAULT_REPORTS = Path("reports")
DEFAULT_OUT = Path("reports/index.html")

SECTIONS = [
    ("Validation 5.1 and 5.2", "phase5_validation.md"),
    ("Retrieval results", "results.md"),
    ("Fusion", "fusion_tuning.md"),
]

FIGURES = [
    ("Novelty by item type", "figures/novelty_by_item.png"),
    ("Ablation ladder", "figures/ablation_ladder.png"),
]

CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0 auto; padding: 3rem 1.5rem; max-width: 60rem;
  font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #1a1a1a; background: #fdfdfc;
}
h1 { font-size: 1.9rem; margin: 0 0 .25rem; letter-spacing: -.02em; }
h2 { font-size: 1.25rem; margin: 3rem 0 .75rem; letter-spacing: -.01em; }
h3 { font-size: 1rem; margin: 1.75rem 0 .5rem; color: #444; }
.sub { color: #6b6b6b; margin: 0 0 2rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .9rem; }
th, td { text-align: left; padding: .5rem .7rem; border-bottom: 1px solid #e7e7e4; }
th { font-weight: 600; color: #444; border-bottom: 2px solid #d8d8d4; }
td:not(:first-child), th:not(:first-child) { font-variant-numeric: tabular-nums; }
figure { margin: 1.5rem 0; }
img { max-width: 100%; height: auto; display: block; }
figcaption { color: #6b6b6b; font-size: .85rem; margin-top: .5rem; }
.note {
  border-left: 3px solid #c2410c; padding: .6rem 0 .6rem .9rem;
  margin: 1.25rem 0; color: #555; font-size: .92rem; background: #fbf6f3;
}
.scroll { overflow-x: auto; }
code { font-size: .88em; background: #f2f2ef; padding: .1em .35em; border-radius: 3px; }
footer { margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid #e7e7e4;
         color: #8a8a8a; font-size: .85rem; }
@media (prefers-color-scheme: dark) {
  body { color: #e8e8e6; background: #16161a; }
  h3 { color: #b5b5b2; } .sub, figcaption, footer { color: #9a9a97; }
  th, td { border-bottom-color: #2c2c31; } th { color: #c5c5c2; border-bottom-color: #3a3a40; }
  .note { background: #221a16; color: #c5c5c2; }
  code { background: #24242a; }
}
"""


def _md_table_to_html(lines: list[str]) -> str:
    rows = [l for l in lines if l.strip().startswith("|")]
    if len(rows) < 2:
        return ""
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, body = cells[0], cells[2:]  # cells[1] is the --- separator
    out = ["<div class='scroll'><table><thead><tr>"]
    out += [f"<th>{html.escape(c)}</th>" for c in header]
    out.append("</tr></thead><tbody>")
    for row in body:
        out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def _inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)


def markdown_to_html(text: str) -> str:
    out: list[str] = []
    buffer: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            out.append(f"<p>{_inline(' '.join(paragraph))}</p>")
            paragraph.clear()

    def flush_table() -> None:
        if buffer:
            out.append(_md_table_to_html(buffer))
            buffer.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            flush_paragraph()
            buffer.append(line)
            continue
        flush_table()
        if not stripped:
            flush_paragraph()
        elif stripped.startswith("### "):
            flush_paragraph()
            out.append(f"<h3>{_inline(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            flush_paragraph()
            out.append(f"<h3>{_inline(stripped[3:])}</h3>")
        elif stripped.startswith("# "):
            flush_paragraph()
        elif stripped.startswith("*") and stripped.endswith("*") and len(stripped) > 2:
            flush_paragraph()
            out.append(f"<div class='note'>{_inline(stripped.strip('*'))}</div>")
        else:
            paragraph.append(stripped)

    flush_paragraph()
    flush_table()
    return "\n".join(out)


def _figure(reports_dir: Path, title: str, rel: str) -> str:
    path = reports_dir / rel
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode()
    return (
        f"<figure><img alt='{html.escape(title)}' "
        f"src='data:image/png;base64,{encoded}'>"
        f"<figcaption>{html.escape(title)}</figcaption></figure>"
    )


def render(reports_dir: Path, out_path: Path) -> Path:
    parts = [
        "<h1>Ticker</h1>",
        "<p class='sub'>Hybrid BM25 and dense retrieval over SEC filings, "
        "with a per sentence novelty layer.</p>",
    ]

    for title, rel in FIGURES:
        block = _figure(reports_dir, title, rel)
        if block:
            parts.append(block)

    for title, filename in SECTIONS:
        path = reports_dir / filename
        if not path.exists():
            continue
        parts.append(f"<h2>{html.escape(title)}</h2>")
        parts.append(markdown_to_html(path.read_text()))

    parts.append(
        "<footer>Generated from the report files by "
        "<code>scripts/render_html.py</code>. Every number here is written by "
        "the pipeline, not by this page.</footer>"
    )

    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Ticker</title><style>{CSS}</style></head><body>"
        + "\n".join(parts)
        + "</body></html>"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    path = render(args.reports_dir, args.out)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
