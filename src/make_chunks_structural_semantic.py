"""
Structural-Semantic chunking: section headings as hard boundaries,
semantic sub-splitting (cosine similarity) inside each section.

Privacy property: no chunk ever crosses a section boundary, so sensitive
content bound to one section cannot contaminate chunks belonging to another
section via the embedding index. No sentence overlap is applied (semantic cuts
are clean topic boundaries; an overlap prefix would only blur embeddings).
"""

from typing import List, Tuple, TYPE_CHECKING

from make_chunks_structural import (
    _detect_sections,
    _word_count,
)
from make_chunks_semantic import (
    _semantic_subsplit,
)

if TYPE_CHECKING:
    from ai import AI


def make_chunks_structural_semantic(
    text: str,
    ai: 'AI',
    max_words: int = 200,
    min_words: int = 25,
    similarity_threshold: float = 0.45,
    min_sentences: int = 1,
    max_sentences: int = 8,
) -> List[str]:
    """
    Chunks respect both section boundaries (hard) and semantic flow (soft).

    Parameters
    ----------
    max_words            : soft word limit per chunk (default 200)
    min_words            : chunks shorter than this merge into the previous (default 25)
    similarity_threshold : cosine-sim threshold for semantic cuts (default 0.45)
    min_sentences        : minimum sentences before a semantic cut (default 1)
    max_sentences        : force a cut after this many sentences (default 8)

    No sentence overlap is applied (semantic boundaries are clean topic cuts).
    """
    if not text or not text.strip():
        return []

    # Phase 1 — Structural skeleton
    sections = _detect_sections(text)
    section_blocks: List[Tuple[int, str]] = []

    if not sections:
        section_blocks.append((0, text.strip()))
    else:
        preamble = text[:sections[0][0]].strip()
        if preamble and _word_count(preamble) >= min_words:
            section_blocks.append((0, preamble))
        boundaries = [s[0] for s in sections] + [len(text)]
        for i, (offset, *_) in enumerate(sections):
            block = text[offset:boundaries[i + 1]].strip()
            if block:
                section_blocks.append((i + 1, block))

    # Phase 2 — Semantic sub-splitting inside each section
    refined: List[Tuple[int, str]] = []
    for section_id, block in section_blocks:
        if _word_count(block) <= max_words:
            refined.append((section_id, block))
        else:
            for chunk in _semantic_subsplit(block, ai, max_words, similarity_threshold, min_sentences, max_sentences):
                refined.append((section_id, chunk))

    # Phase 3 — Merge undersized chunks within the same section
    merged: List[Tuple[int, str]] = []
    for section_id, chunk in refined:
        if merged and merged[-1][0] == section_id and _word_count(chunk) < min_words:
            merged[-1] = (merged[-1][0], merged[-1][1] + ' ' + chunk)
        else:
            merged.append((section_id, chunk))

    if not merged:
        return []

    # Nessun overlap: i tagli semantici sono confini di argomento netti,
    # un prefisso di overlap sporcherebbe solo l'embedding del chunk successivo.
    return [text for _, text in merged]
