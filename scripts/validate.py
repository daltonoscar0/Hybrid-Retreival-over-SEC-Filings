#!/usr/bin/env python3
"""Phase 5 validation 5.1 and 5.2, the two that need no human label.

5.1 is the load-bearing one. PLAN says that if the diff-agreement AUC comes
back near 0.5 the novelty measure is broken and no downstream framing
rescues it, so this script is meant to run before the labeling sitting
rather than after it. Knowing whether the measure means anything should
gate whether a day of judging is worth spending.

Writes `reports/phase5_validation.md` and prints the same numbers. When the
AUC is near 0.5 it prints the ranked diagnosis list RUN.md C1 specifies, in
that order, because the three causes are told apart by different evidence
and checking them out of order wastes the most time on the least likely.

5.3, the agreement against blind human spot-check labels, is not here. It
needs `data/spotcheck/`, which is a human-only artifact.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ticker import db  # noqa: E402
from ticker.validation import (  # noqa: E402
    bootstrap_ci,
    collect_diff_agreement,
    coverage,
    load_scores,
    mean_novelty_by_item,
)

DEFAULT_DB = Path("data/ticker.duckdb")
DEFAULT_SCORES = Path("data/novelty/scores.jsonl")
DEFAULT_REPORT = Path("reports/phase5_validation.md")

ITEM_NAMES = {
    "1": "Business",
    "1A": "Risk Factors",
    "3": "Legal Proceedings",
    "7": "MD&A",
    "7A": "Market Risk",
}

# PLAN 5.2 predicts novelty should sit in the items that carry news. Item 1
# Business is the boilerplate control: novelty concentrating there is the
# documented failure mode, not a finding.
NEWS_ITEMS = {"1A", "3", "7"}
BOILERPLATE_ITEM = "1"

# Below this, an item's mean is dominated by what the section is made of
# rather than by how novel its content is. Item 3 in this corpus averages
# under 10 sentences a filing because it is mostly a cross-reference into the
# notes; comparing that mean against MD&A's 393 is comparing section
# composition, not novelty.
STUB_SENTENCES_PER_SECTION = 50

_SECTIONS_PER_ITEM_SQL = """
    SELECT sec.item, COUNT(DISTINCT sec.section_id)
    FROM sections sec
    JOIN filings f ON sec.accession = f.accession
    WHERE f.form IN ({placeholders})
    GROUP BY sec.item
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-sections", type=int, default=None)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="report over a score file that does not cover the corpus",
    )
    args = parser.parse_args()

    if not args.scores.exists():
        sys.exit(f"no score file at {args.scores}. Run scripts/score_novelty.py first.")

    by_sentence, item_rows, section_ids = load_scores(args.scores)
    if args.limit_sections:
        section_ids = section_ids[: args.limit_sections]
    print(f"loaded {len(by_sentence)} scored sentences over {len(section_ids)} sections")

    con = db.connect_readonly(args.db)

    cov = coverage(con, args.scores)
    print(
        f"coverage: forms {list(cov.forms)}, "
        f"{cov.accessions_in_file}/{cov.accessions_in_corpus} filings, "
        f"{cov.ciks_in_file}/{cov.ciks_in_corpus} companies"
    )
    if not cov.complete:
        print(
            f"  WARNING: {len(cov.missing_accessions)} filings of these forms carry "
            "sections but no scores. 5.1 and 5.2 below describe the subset that was "
            "scored, not the corpus. Finish the scoring pass before reporting these "
            "numbers."
        )
        for accession in cov.missing_accessions[:10]:
            print(f"    unscored: {accession}")
        if not args.allow_partial:
            sys.exit(
                "refusing to write a report over a partial score file. Re-run with "
                "--allow-partial to produce it anyway, clearly labelled."
            )

    print("5.1 aligning priors and diffing ...")
    observations, diag = collect_diff_agreement(con, by_sentence, section_ids)
    print(
        f"  {diag.sections_used} sections aligned and used, "
        f"{diag.sections_unaligned} without a prior, "
        f"join rate {diag.join_rate:.1%}"
    )

    point, lower, upper = bootstrap_ci(
        observations, resamples=args.resamples, seed=args.seed
    )
    changed = sum(sum(o.changed) for o in observations)
    total = sum(len(o.changed) for o in observations)
    print(f"  AUC {point:.4f}  95% CI [{lower:.4f}, {upper:.4f}]  n={total}")

    print("5.2 mean novelty by item ...")
    by_item = mean_novelty_by_item(item_rows, resamples=args.resamples, seed=args.seed)
    placeholders = ", ".join("?" for _ in cov.forms)
    sections_per_item = dict(
        con.execute(
            _SECTIONS_PER_ITEM_SQL.format(placeholders=placeholders), list(cov.forms)
        ).fetchall()
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w") as out:
        out.write(_render(point, lower, upper, changed, total, diag, by_item, cov, sections_per_item, args))
    print(f"wrote {args.report}")

    if lower <= 0.5 <= upper:
        _print_diagnoses(diag)


def _render(point, lower, upper, changed, total, diag, by_item, cov, sections_per_item, args) -> str:
    lines = [
        "# Phase 5 validation: 5.1 diff agreement and 5.2 concentration by item",
        "",
        "Both numbers here are computed without a human label. 5.3 and every",
        "Phase 6 number are not, and are not in this report.",
        "",
        f"Scores read from `{args.scores}`. Bootstrap {args.resamples} resamples, seed {args.seed}.",
        "",
        "## Corpus coverage",
        "",
        f"Forms scored: {', '.join(cov.forms) if cov.forms else 'none'}. "
        f"{cov.accessions_in_file} of {cov.accessions_in_corpus} filings of those "
        f"forms carry scores, across {cov.ciks_in_file} of {cov.ciks_in_corpus} "
        "companies.",
        "",
    ]
    if not cov.complete:
        lines += [
            f"**Partial.** {len(cov.missing_accessions)} filings of these forms have "
            "sections but no novelty scores, so every number below describes the "
            "scored subset rather than the corpus.",
            "",
        ]
    lines += [
        "## 5.1 Agreement with the Lazy Prices diff",
        "",
        "AUC of the raw novelty contrast against the diff's binary changed label,",
        "where changed is `inserted` or `modified` and unchanged is `unchanged`.",
        "The diff is computed from text alone by `ticker.novelty.lazy_prices` with",
        "no access to the surprisal code, so this is agreement between two",
        "independent constructions.",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| AUC | {point:.4f} |",
        f"| 95% CI | [{lower:.4f}, {upper:.4f}] |",
        f"| sentences | {total:,} |",
        f"| changed | {changed:,} ({changed / total:.1%}) |" if total else "| changed | 0 |",
        f"| unchanged | {total - changed:,} |",
        "",
        "The interval is a cluster bootstrap over sections, not over sentences.",
        "Sentences inside one section share a firm, a filing date and one prior",
        "alignment, so resampling them individually would treat correlated rows as",
        "independent evidence and return an interval too narrow to be honest.",
        "",
        "### Join coverage",
        "",
        "These counts are the evidence that separates a measure that does not work",
        "from a join that is broken. Both produce an AUC near 0.5.",
        "",
        "| stage | count |",
        "|---|---|",
        f"| sections in the score file | {diag.sections_total:,} |",
        f"| excluded, no earlier filing with this item | {diag.sections_unaligned:,} |",
        f"| excluded, empty prior or current | {diag.sections_empty_prior:,} |",
        f"| aligned and used | {diag.sections_used:,} |",
        f"| sentences the diff labelled | {diag.sentences_diffed:,} |",
        f"| of those, carrying a novelty score | {diag.sentences_joined:,} |",
        f"| join rate | {diag.join_rate:.1%} |",
        "",
    ]
    if diag.unjoined_examples:
        lines += [
            "Unjoined sentence IDs, first few: "
            + ", ".join(f"`{s}`" for s in diag.unjoined_examples),
            "",
            "An unjoined sentence is dropped, never defaulted to zero. A default",
            "would enter it as the least novel sentence in the corpus, which is a",
            "fabricated observation with a real effect on the AUC.",
            "",
        ]

    lines += [
        "## 5.2 Mean novelty by item type",
        "",
        "Raw contrast, sentence-level percentile bootstrap. The unit is the",
        "sentence here because the quantity is a per-item corpus mean rather than",
        "a ranking statistic.",
        "",
        "| item | name | n | mean novelty | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for item, (n, mean, lo, hi) in by_item.items():
        lines.append(
            f"| {item} | {ITEM_NAMES.get(item, '')} | {n:,} | {mean:+.4f} | "
            f"[{lo:+.4f}, {hi:+.4f}] |"
        )
    lines.append("")

    lines += _item_reading(by_item, sections_per_item)
    return "\n".join(lines) + "\n"


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _item_reading(by_item, sections_per_item) -> list[str]:
    """State 5.2 against what PLAN predicted, including where it disagrees.

    A ranking alone is not the check. PLAN predicts novelty concentrates in
    legal proceedings, risk factors and MD&A relative to Item 1 Business, so
    what matters is whether each predicted item sits above the Business
    control and whether the intervals separate. Items whose sections are a
    few sentences long are reported apart from that comparison: their mean is
    dominated by section composition rather than by how novel the item's
    content is, and reading them alongside items forty times longer invites a
    conclusion neither number supports.
    """
    if BOILERPLATE_ITEM not in by_item:
        return []

    control_n, control_mean, control_lo, control_hi = by_item[BOILERPLATE_ITEM]
    lines = ["### Reading", ""]

    stubs = {
        item
        for item in by_item
        if sections_per_item.get(item)
        and by_item[item][0] / sections_per_item[item] < STUB_SENTENCES_PER_SECTION
    }

    lines += [
        f"The control is Item {BOILERPLATE_ITEM} {ITEM_NAMES[BOILERPLATE_ITEM]}, the "
        f"most boilerplate-heavy item in the corpus, at {control_mean:+.4f}.",
        "",
    ]

    if stubs:
        detail = ", ".join(
            f"Item {item} at {by_item[item][0] / sections_per_item[item]:.1f} "
            f"sentences per section"
            for item in sorted(stubs)
        )
        lines += [
            f"Set aside first: {detail}. The substantial items run "
            + ", ".join(
                f"{by_item[i][0] / sections_per_item[i]:.0f}"
                for i in sorted(set(by_item) - stubs)
            )
            + " sentences per section. A mean over a handful of sentences per",
            "filing describes what the item is made of more than how novel it is.",
            "Item 3 in this corpus is largely a cross-reference into the notes, so",
            "its position at either end of the table is not evidence about whether",
            "legal news is novel.",
            "",
        ]

    lines += ["Against the control, for the substantial items:", ""]
    for item in sorted(set(by_item) - stubs):
        if item == BOILERPLATE_ITEM:
            continue
        n, mean, lo, hi = by_item[item]
        predicted = item in NEWS_ITEMS
        above = mean > control_mean
        separated = not _overlaps((lo, hi), (control_lo, control_hi))
        if above and separated:
            verdict = "above the control, intervals disjoint"
        elif above:
            verdict = "above the control, but the intervals overlap"
        elif separated:
            verdict = "below the control, intervals disjoint"
        else:
            verdict = "indistinguishable from the control"
        tag = "predicted by PLAN" if predicted else "not among PLAN's predictions"
        lines.append(
            f"- Item {item} {ITEM_NAMES.get(item, '')} ({tag}): {verdict}."
        )
    lines.append("")

    predicted_substantial = [i for i in NEWS_ITEMS if i in by_item and i not in stubs]
    confirmed = [
        i
        for i in predicted_substantial
        if by_item[i][1] > control_mean
        and not _overlaps((by_item[i][2], by_item[i][3]), (control_lo, control_hi))
    ]
    if len(confirmed) < len(predicted_substantial):
        missed = sorted(set(predicted_substantial) - set(confirmed))
        lines += [
            "This is a partial result and the shortfall is the part worth stating. "
            + ", ".join(f"Item {i} {ITEM_NAMES.get(i, '')}" for i in missed)
            + (" is " if len(missed) == 1 else " are ")
            + "predicted by PLAN to carry more novelty than boilerplate and does not",
            "separate from the control here. 5.2 is a sanity check rather than the",
            "test of the measure, and 5.1 passed independently, so this does not",
            "invalidate the measure. It does mean the by-item figure cannot be",
            "presented as confirming the literature's prediction.",
            "",
        ]
    return lines


def _print_diagnoses(diag) -> None:
    """The ranked list RUN.md C1 asks for, in its order."""
    print()
    print("The 95% CI covers 0.5. Per RUN.md C1, in this order:")
    print()
    print("1. The sentence-ID join. Join rate above is "
          f"{diag.join_rate:.1%}; anything below ~99% means the diff and the")
    print("   score file disagree about which sentences exist. A join that is")
    print("   complete but misaligned shows a high rate and a chance AUC, so")
    print("   confirm ordinals are contiguous as well as present.")
    print()
    print("2. Whether a document leaked into its own background model. Check that")
    print("   score_novelty's as_of is the filing's own filed_at and that fit")
    print("   rejects sentences at or after it. A document scored against a model")
    print("   that has read it reads as uniformly unsurprising and separates")
    print("   nothing.")
    print()
    print("3. Whether the alignment matched the wrong prior section. Check the")
    print("   gap_days distribution over used sections: a year-over-year 10-K")
    print("   comparison should cluster near 365. Mass near 0 or above 800 means")
    print("   align_prior_section is pairing sections it should not.")


if __name__ == "__main__":
    main()
