"""4-sentence window, stride 2, per PLAN.md.

Chunks carry the stable sentence_ids that built them so per-sentence novelty
scores map onto a chunk for display and re-ranking with no recomputation.
"""

from __future__ import annotations

from ticker.records import Chunk, Sentence

WINDOW = 4
STRIDE = 2


def chunk_sentences(sentences: list[Sentence]) -> list[Chunk]:
    """Window `sentences` (already in document order, one section) into chunks."""
    if not sentences:
        return []

    section_id = sentences[0].section_id
    if any(s.section_id != section_id for s in sentences):
        raise ValueError("chunk_sentences expects sentences from a single section")

    chunks: list[Chunk] = []
    n = len(sentences)
    start = 0
    while start < n:
        window = sentences[start:start + WINDOW]
        chunks.append(
            Chunk(
                chunk_id=f"{section_id}#chunk{start}",
                section_id=section_id,
                sentence_ids=tuple(s.sentence_id for s in window),
                text=" ".join(s.text for s in window),
            )
        )
        if start + WINDOW >= n:
            break
        start += STRIDE
    return chunks
