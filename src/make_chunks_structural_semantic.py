"""
Structural-Semantic chunking.

Hybrid strategy: structural boundaries (headings) act as HARD cuts that no
chunk may cross; within each section, when the block is too long, semantic
boundaries (drops in sentence-embedding similarity) provide soft sub-cuts.

Rationale
---------
- Pure WORD chunking ignores topic and structure entirely.
- Pure SEMANTIC chunking may merge across heading boundaries if topics happen
  to be similar across sections.
- Pure STRUCTURAL chunking respects sections but inside a long section it
  cannot tell sub-topics apart.

Combining the two produces chunks that are simultaneously:
  (a) confined to a single logical section of the document     → privacy
  (b) topically tight inside that section                       → completeness

The key privacy property is that NO chunk and NO overlap-prefix ever crosses
a heading boundary, so a sensitive category bound to one section cannot
contaminate chunks belonging to a different section via the embedding index.
"""

import re
from typing import List, Tuple, TYPE_CHECKING

# Reuse the heading detector and the word counter from the structural chunker:
# behaviour stays consistent across the two non-standard strategies.
from make_chunks_structural import (
    _detect_sections,
    _word_count,
    _split_sentences as _structural_split_sentences,
)
# Reuse the more robust sentence splitter from the semantic chunker:
# it handles PDF soft line-wraps and Italian abbreviations (S.p.A., art. 3.4).
from make_chunks_semantic import (
    _split_sentences as _semantic_split_sentences,
    _cosine_sim,
)


if TYPE_CHECKING:
    from ai import AI


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_chunks_structural_semantic(
    text: str,
    ai: 'AI',
    max_words: int = 200,
    min_words: int = 20,
    similarity_threshold: float = 0.45,
    min_sentences: int = 1,
    max_sentences: int = 8,
    overlap_sentences: int = 1,
) -> List[str]:
    """
    Split *text* into chunks that respect both the document's structural
    sections (headings) AND the semantic flow inside each section.

    Strategy
    --------
    1. Detect all section headings and slice the text at heading boundaries.
       The slices are called "section blocks" and their boundaries are
       INVIOLABLE: no chunk produced by this function will ever span across
       a heading.
    2. For every section block:
         - if it fits inside max_words → keep it as one chunk.
         - otherwise → split it semantically: cut after a sentence whenever
           cosine similarity to the next sentence drops below
           similarity_threshold, the running chunk reaches max_sentences,
           OR the running chunk reaches max_words. Hard paragraph breaks
           (blank lines) inside the section are also honoured as cuts.
    3. Merge undersized chunks (< min_words) into the previous chunk
       belonging to the SAME section. Chunks from different sections are
       NEVER merged.
    4. Apply sentence-level overlap inside each section: the first chunk of
       a section receives no prefix from the previous section. This is the
       privacy-preserving property of the algorithm.

    Parameters
    ----------
    text : str
        Raw document text (paragraphs separated by blank lines, headings
        as described in make_chunks_structural._detect_sections).
    ai : AI
        SD-RAG AI object. Uses ai.e.embed() — same nomic-embed model used
        elsewhere in the pipeline, so embeddings get cached in ecache.
    max_words : int
        Soft upper limit (words) for a chunk before semantic sub-splitting
        kicks in inside a section. Default: 200.
    min_words : int
        Chunks shorter than this are merged into the previous chunk of the
        SAME section. Default: 20.
    similarity_threshold : float
        Cosine similarity below which a sentence boundary becomes a chunk
        boundary inside an oversized section. Lower = fewer, larger chunks.
        Higher = more, smaller chunks. Default: 0.45.
    min_sentences : int
        Never cut a chunk shorter than this many sentences, even on
        similarity drop. Default: 1.
    max_sentences : int
        Force a cut after this many sentences regardless of similarity.
        Guards against run-on chunks on very uniform text. Default: 8.
    overlap_sentences : int
        Sentences from the tail of chunk N prepended to chunk N+1, ONLY
        when N and N+1 belong to the same section. Set 0 to disable.
        Default: 1.

    Returns
    -------
    List[str]
        Chunk strings ready to embed and index.
    """

    if not text or not text.strip():
        return []

    # -----------------------------------------------------------------------
    # Phase 1 — Structural skeleton: slice text into section blocks
    # -----------------------------------------------------------------------
    sections = _detect_sections(text)

    # Each item: (section_id, block_text).
    # section_id is a stable integer used later to forbid cross-section
    # merging and cross-section overlap.
    section_blocks: List[Tuple[int, str]] = []

    if not sections:
        # No headings detected at all → degrade to a single "section 0".
        # The semantic sub-splitter below will still produce sensible chunks.
        section_blocks.append((0, text.strip()))
    else:
        # Optional preamble before the first heading
        preamble = text[:sections[0][0]].strip()
        if preamble and _word_count(preamble) >= min_words:
            section_blocks.append((0, preamble))

        boundaries = [s[0] for s in sections] + [len(text)]
        for i, (offset, _label, _heading, _level) in enumerate(sections):
            block_text = text[offset:boundaries[i + 1]].strip()
            if block_text:
                # +1 to keep preamble = 0 distinct from the first real section
                section_blocks.append((i + 1, block_text))

    # -----------------------------------------------------------------------
    # Phase 2 — Semantic refinement inside each section block
    # -----------------------------------------------------------------------
    # Each item: (section_id, chunk_text)
    refined: List[Tuple[int, str]] = []

    for section_id, block in section_blocks:
        if _word_count(block) <= max_words:
            # Small enough → no need to split; semantic boundaries inside a
            # already-short block would just produce noisy fragments.
            refined.append((section_id, block))
            continue

        # Block is oversized → semantic sub-split inside the section
        sub_chunks = _semantic_subsplit(
            block,
            ai=ai,
            max_words=max_words,
            similarity_threshold=similarity_threshold,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
        )
        for sc in sub_chunks:
            refined.append((section_id, sc))

    # -----------------------------------------------------------------------
    # Phase 3 — Merge undersized chunks WITHIN THE SAME SECTION only
    # -----------------------------------------------------------------------
    merged: List[Tuple[int, str]] = []
    for section_id, chunk in refined:
        if (
            merged
            and merged[-1][0] == section_id          # same section
            and _word_count(chunk) < min_words
        ):
            prev_sid, prev_text = merged[-1]
            merged[-1] = (prev_sid, prev_text + ' ' + chunk)
        else:
            merged.append((section_id, chunk))

    if not merged:
        return []

    # -----------------------------------------------------------------------
    # Phase 4 — Sentence-level overlap, but ONLY inside the same section
    # -----------------------------------------------------------------------
    # PRIVACY-PRESERVING DETAIL: the first chunk of every section starts
    # fresh; it never gets a prefix from the tail of the previous section.
    # This prevents sensitive content from one section from leaking into
    # the embedding of a chunk that logically belongs to another section.
    if overlap_sentences <= 0 or len(merged) < 2:
        return [text for _sid, text in merged]

    # Same word-cap as the structural chunker, for the same reason:
    # an unsplit "sentence" that is really a whole paragraph (e.g. a list
    # without punctuation) must not inflate the next chunk beyond max_words.
    max_overlap_words = max(10, max_words // 4)

    overlapped: List[str] = [merged[0][1]]
    for i in range(1, len(merged)):
        prev_sid, prev_text = merged[i - 1]
        curr_sid, curr_text = merged[i]

        if curr_sid != prev_sid:
            # Section boundary crossed → NO overlap prefix.
            overlapped.append(curr_text)
            continue

        prev_sentences = _structural_split_sentences(prev_text)
        n = min(overlap_sentences, len(prev_sentences))
        if n == 0:
            overlapped.append(curr_text)
            continue

        prefix = ' '.join(prev_sentences[-n:])
        prefix_words = prefix.split()
        if len(prefix_words) > max_overlap_words:
            prefix = ' '.join(prefix_words[-max_overlap_words:])
        overlapped.append((prefix + ' ' + curr_text).strip())

    return overlapped


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _semantic_subsplit(
    block: str,
    ai: 'AI',
    max_words: int,
    similarity_threshold: float,
    min_sentences: int,
    max_sentences: int,
) -> List[str]:
    """
    Semantic splitter to be applied INSIDE a single section block.

    Compared with make_chunks_semantic, this variant:
      - works on a single section block (no inter-section logic),
      - adds a hard word budget (max_words) on top of max_sentences,
        because sections of regulatory text can have long sentences,
      - does NOT emit overlap (overlap is handled by the outer function,
        section-aware).
    """
    # _semantic_split_sentences returns sentences interleaved with empty
    # strings marking paragraph (blank-line) boundaries. We keep those
    # sentinels and treat them as additional hard cuts.
    tokens = _semantic_split_sentences(block)
    if not tokens:
        return []

    real_sentences: List[str] = []
    para_cuts: set = set()  # index (into real_sentences) AFTER which a hard cut occurs

    for t in tokens:
        if t == '':
            if real_sentences:
                para_cuts.add(len(real_sentences) - 1)
        else:
            real_sentences.append(t)

    if not real_sentences:
        return []

    # Single sentence in this block → return as-is (even if oversized,
    # we have no boundary to cut on; outer code accepts a >max_words chunk
    # in this degenerate case).
    if len(real_sentences) == 1:
        return [real_sentences[0]]

    # Batch-embed all sentences in one call (cache hits expected).
    embeddings = ai.e.embed(real_sentences)  # shape (N, dim)

    # Consecutive cosine similarities
    sims: List[float] = []
    for i in range(len(real_sentences) - 1):
        sims.append(_cosine_sim(embeddings[i], embeddings[i + 1]))

    # Decide cut positions
    cut_after = [False] * len(real_sentences)
    current_sentences = 1
    current_words = _word_count(real_sentences[0])

    for i in range(len(sims)):
        next_words = current_words + _word_count(real_sentences[i + 1])

        is_para_break = i in para_cuts
        is_sim_break  = sims[i] < similarity_threshold and current_sentences >= min_sentences
        is_sent_cap   = current_sentences >= max_sentences
        is_word_cap   = next_words > max_words and current_sentences >= min_sentences

        if is_para_break or is_sim_break or is_sent_cap or is_word_cap:
            cut_after[i] = True
            current_sentences = 1
            current_words = _word_count(real_sentences[i + 1])
        else:
            current_sentences += 1
            current_words = next_words

    # Group sentences into chunks
    chunks: List[str] = []
    current_group: List[str] = []
    for i, sent in enumerate(real_sentences):
        current_group.append(sent)
        if cut_after[i]:
            chunks.append(' '.join(current_group))
            current_group = []
    if current_group:
        chunks.append(' '.join(current_group))

    return chunks