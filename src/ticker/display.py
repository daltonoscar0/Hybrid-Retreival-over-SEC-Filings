"""Turning a retrieved chunk into something readable, with per-sentence
novelty shown as colour.

Two things here are load-bearing and neither is cosmetic.

The z-score is computed for display and cannot escape into aggregation.
Invariant 3 says within-document z-scores are for colouring one document's
sentences and nothing else. `novelty_colours` therefore takes the raw
contrasts for one document, z-scores them locally, and returns colour bands.
It returns bands, not numbers, so there is no float on the way out that a
caller could average across documents. The raw score stays available on
`SentenceView.raw` for anyone who needs the real quantity.

Novelty never enters the relevance score. Invariant 4. `rerank_by_novelty`
reorders an existing ranked list and is a separate call the caller has to
make; nothing in the scoring path can reach it. The CLI exposes it as a flag,
off by default, which is the "toggle rather than a silent term" that PLAN
section 6 asks for.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_SCORE_FILES = (
    "scores.jsonl",
    "scores_10-Q.jsonl",
    "scores_8-K.jsonl",
)

# Bands, not a continuous scale. A reader distinguishes three or four levels of
# emphasis, not thirty, and naming them keeps the display honest about how much
# resolution the z-score really carries.
BAND_HIGH = "high"
BAND_MID = "mid"
BAND_LOW = "low"

HIGH_Z = 1.0
MID_Z = 0.25


@dataclass(frozen=True, slots=True)
class SentenceView:
    sentence_id: str
    text: str
    raw: float | None
    band: str | None


def load_novelty(
    novelty_dir: Path | str = Path("data/novelty"),
    filenames: Sequence[str] = DEFAULT_SCORE_FILES,
) -> dict[str, float]:
    """sentence_id -> raw contrast, over every form that has been scored.

    Missing files are skipped rather than raising: a corpus scored for 10-K
    only is a legitimate state, and the display degrades to uncoloured text
    for the forms that carry no score instead of refusing to run.
    """
    novelty_dir = Path(novelty_dir)
    scores: dict[str, float] = {}
    for name in filenames:
        path = novelty_dir / name
        if not path.exists():
            continue
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                scores[row["sentence_id"]] = row["novelty"]
    return scores


def novelty_bands(raw: Sequence[float | None]) -> list[str | None]:
    """Z-score within this document only, then bucket into three bands.

    Returns bands rather than z-scores on purpose. A float leaving this
    function is a float someone can average across documents six weeks later,
    and a mean of within-document z-scores measures nothing.
    """
    present = [value for value in raw if value is not None]
    if len(present) < 2:
        return [None] * len(raw)
    mean = statistics.fmean(present)
    sd = statistics.pstdev(present)
    if sd == 0:
        return [None] * len(raw)

    bands: list[str | None] = []
    for value in raw:
        if value is None:
            bands.append(None)
            continue
        z = (value - mean) / sd
        if z >= HIGH_Z:
            bands.append(BAND_HIGH)
        elif z >= MID_Z:
            bands.append(BAND_MID)
        else:
            bands.append(BAND_LOW)
    return bands


def build_sentence_views(
    sentence_ids: Sequence[str],
    texts: Sequence[str],
    novelty: dict[str, float],
) -> list[SentenceView]:
    raw = [novelty.get(sid) for sid in sentence_ids]
    bands = novelty_bands(raw)
    return [
        SentenceView(sentence_id=sid, text=text, raw=value, band=band)
        for sid, text, value, band in zip(sentence_ids, texts, raw, bands)
    ]


def document_novelty(raw: Iterable[float | None]) -> float | None:
    """Document-level aggregate, from RAW contrasts only.

    This is the function that would be wrong if it took z-scores, so it takes
    the raw values and there is no variant that accepts anything else.
    """
    present = [value for value in raw if value is not None]
    return statistics.fmean(present) if present else None


def rerank_by_novelty(
    ranked: Sequence[tuple[str, float]],
    chunk_novelty: dict[str, float | None],
    *,
    weight: float = 0.5,
) -> list[tuple[str, float]]:
    """Reorder a ranked list by relevance rank blended with novelty rank.

    Rank-based rather than score-based, because relevance scores and novelty
    contrasts have no common unit and adding them would make the blend depend
    on whichever happened to have the wider range. Both sides become ranks
    first, which is the same reasoning RRF rests on.

    Chunks with no novelty score keep their relevance rank and are neither
    promoted nor penalised. Treating an unscored chunk as zero-novelty would
    push every 10-Q below every 10-K on a corpus scored for 10-K only.

    One asymmetry worth knowing about: novelty ranks run over the scored
    subset while relevance ranks run over the whole list, so when most of a
    result set is unscored the two scales have different lengths and an
    unscored chunk low in relevance is compared on the longer one. The effect
    is small and it moves unscored chunks down rather than up, so it cannot
    manufacture a novelty win. It disappears entirely once every form is
    scored.
    """
    if not ranked:
        return []

    relevance_rank = {chunk_id: i for i, (chunk_id, _) in enumerate(ranked)}
    scored = [c for c, _ in ranked if chunk_novelty.get(c) is not None]
    novelty_rank = {
        chunk_id: i
        for i, chunk_id in enumerate(
            sorted(scored, key=lambda c: chunk_novelty[c], reverse=True)
        )
    }

    def blended(chunk_id: str) -> float:
        r = relevance_rank[chunk_id]
        if chunk_id not in novelty_rank:
            return float(r)
        return (1 - weight) * r + weight * novelty_rank[chunk_id]

    order = sorted((c for c, _ in ranked), key=blended)
    original = dict(ranked)
    return [(chunk_id, original[chunk_id]) for chunk_id in order]
