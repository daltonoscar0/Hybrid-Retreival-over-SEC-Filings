#!/usr/bin/env python3
"""The two figures PLAN section 3 Phase 7 asks for.

  reports/figures/novelty_by_item.png   novelty distribution by item type
  reports/figures/ablation_ladder.png   the retrieval comparison

The second needs relevance judgments. Without them this writes a placeholder
carrying the reason rather than an empty axis, because a blank chart in a
README reads as a broken pipeline and the real state is "not measured yet".

Usage:
  uv run python scripts/figures.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_NOVELTY_DIR = Path("data/novelty")
DEFAULT_OUT_DIR = Path("reports/figures")
DEFAULT_QRELS = Path("data/qrels/standard.jsonl")

SCORE_FILES = ("scores.jsonl", "scores_10-Q.jsonl", "scores_8-K.jsonl")

# Keyed by (form, item) because item numbers mean different things across
# forms: 10-K Item 1 is Business, 10-Q Part II Item 1 is Legal Proceedings.
# Collapsing them onto one axis by number alone would put unrelated prose in
# the same violin.
ITEM_NAMES = {
    ("10-K", "1"): "Business\n10-K 1",
    ("10-K", "1A"): "Risk Factors\n10-K 1A",
    ("10-K", "3"): "Legal\n10-K 3",
    ("10-K", "7"): "MD&A\n10-K 7",
    ("10-K", "7A"): "Market Risk\n10-K 7A",
    ("10-Q", "Part I Item 2"): "MD&A\n10-Q I.2",
    ("10-Q", "Part II Item 1"): "Legal\n10-Q II.1",
    ("10-Q", "Part II Item 1A"): "Risk Factors\n10-Q II.1A",
    ("8-K", "EX-99.1"): "Earnings release\n8-K 99.1",
}

# Below this many sentences a per-item mean describes what the item is made of
# more than how novel it is, so the label says so.
THIN_SECTION_FLOOR = 5_000

INK = "#1a1a1a"
MUTED = "#8a8a8a"
ACCENT = "#c2410c"
GRID = "#e5e5e5"


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _read_by_item(novelty_dir: Path) -> dict[tuple[str, str], list[float]]:
    by_item: dict[tuple[str, str], list[float]] = defaultdict(list)
    for name in SCORE_FILES:
        path = novelty_dir / name
        if not path.exists():
            continue
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                by_item[(row["form"], row["item"])].append(row["novelty"])
    return by_item


def novelty_by_item(novelty_dir: Path, out_path: Path) -> Path:
    by_item = _read_by_item(novelty_dir)
    if not by_item:
        raise SystemExit(f"no novelty scores under {novelty_dir}")

    items = sorted(by_item, key=lambda i: statistics.fmean(by_item[i]))
    data = [by_item[i] for i in items]
    labels = [ITEM_NAMES.get(i, f"{i[0]} {i[1]}") for i in items]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    parts = ax.violinplot(data, showmeans=True, showextrema=False, widths=0.8)
    for body in parts["bodies"]:
        body.set_facecolor(ACCENT)
        body.set_alpha(0.25)
        body.set_edgecolor(ACCENT)
    parts["cmeans"].set_color(ACCENT)
    parts["cmeans"].set_linewidth(2)

    ax.set_xticks(range(1, len(items) + 1))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("novelty (raw contrast, bits per token)", fontsize=9, color=INK)
    ax.set_title(
        "Novelty by item type, ordered by mean",
        fontsize=11, color=INK, loc="left", pad=12,
    )
    bottom = ax.get_ylim()[0]
    for i, item in enumerate(items, 1):
        n = len(by_item[item])
        note = f"n={n:,}" + ("\nthin" if n < THIN_SECTION_FLOOR else "")
        ax.annotate(
            note, (i, bottom), ha="center", va="bottom", fontsize=6.5, color=MUTED,
        )
    _style(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _placeholder(out_path: Path, title: str, reason: str) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", fontsize=12, color=INK)
    ax.text(0.5, 0.40, reason, ha="center", fontsize=9, color=MUTED, wrap=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def ablation_ladder(qrels_path: Path, out_path: Path) -> Path:
    if not qrels_path.exists() or not qrels_path.stat().st_size:
        return _placeholder(
            out_path,
            "Ablation ladder: not measured",
            "Needs relevance judgments. Run scripts/judge.py, then "
            "scripts/evaluate.py, then regenerate this figure.",
        )

    from ticker.evaluation import discover_runs, evaluate_runs  # local: heavy import

    table = evaluate_runs(discover_runs(Path("data/runs")), qrels_path)
    systems = [row.system for row in table]
    scores = [row.ndcg_at_10 for row in table]
    lows = [row.ci_low for row in table]
    highs = [row.ci_high for row in table]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = range(len(systems))
    ax.barh(list(y), scores, color=ACCENT, alpha=0.75, height=0.55)
    ax.errorbar(
        scores, list(y),
        xerr=[[s - lo for s, lo in zip(scores, lows)],
              [hi - s for s, hi in zip(scores, highs)]],
        fmt="none", ecolor=INK, elinewidth=1.2, capsize=4,
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(systems, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("nDCG@10 with bootstrap 95% CI", fontsize=9, color=INK)
    ax.set_title("Ablation ladder", fontsize=11, color=INK, loc="left", pad=12)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--novelty-dir", type=Path, default=DEFAULT_NOVELTY_DIR)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    a = novelty_by_item(args.novelty_dir, args.out_dir / "novelty_by_item.png")
    print(f"wrote {a}")
    b = ablation_ladder(args.qrels, args.out_dir / "ablation_ladder.png")
    print(f"wrote {b}")


if __name__ == "__main__":
    main()
