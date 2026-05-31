import re
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Heading-detection helpers
# ---------------------------------------------------------------------------

# Matches numbered headings like:
#   1   Introduction
#   2.1 Soggetti e aree
#   3.1.2 Dettaglio
#   Art. 4 – Norme
# The number must start at the beginning of the line (possibly after whitespace).
_NUMBERED_HEADING_RE = re.compile(
    r'^\s*'
    r'(?:Art\.?\s*)?'           # optional "Art." prefix
    r'(\d+(?:\.\d+)*)'          # numeric label: 1 | 2.1 | 3.1.2
    r'[\.\s\-–—]+'             # separator: dot, dash, em-dash, or whitespace
    r'([A-ZÀ-Ùa-zà-ù].{2,})',  # heading text (must have at least 3 chars)
    re.MULTILINE,
)

# Matches ALL-CAPS headings (common in airport/aviation procedures)
# e.g. "CAMPO DI APPLICAZIONE", "PRESCRIZIONI DI SICUREZZA"
_ALLCAPS_HEADING_RE = re.compile(
    r'^\s*([A-ZÀÈÌÒÙÁÉÍÓÚ][A-ZÀÈÌÒÙÁÉÍÓÚ\s\-–—]{4,}[A-ZÀÈÌÒÙÁÉÍÓÚ])\s*$',
    re.MULTILINE,
)


def _heading_level(label: str) -> int:
    """Return the depth of a dotted label: '1' → 1, '2.3' → 2, '3.1.2' → 3."""
    return label.count('.') + 1


def _detect_sections(text: str) -> List[Tuple[int, str, str, int]]:
    """
    Scan *text* and return a list of
        (char_offset, label, heading_text, level)
    for every structural section found.

    Sections are detected by two patterns (in priority order):
      1. Numbered headings  → level = depth of dot-notation
      2. ALL-CAPS headings  → level = 1  (treated as top-level)

    The returned list is sorted by char_offset.
    """
    found: List[Tuple[int, str, str, int]] = []
    seen_offsets: set = set()

    # --- Pattern 1: numbered headings ---
    for m in _NUMBERED_HEADING_RE.finditer(text):
        label = m.group(1)
        heading = m.group(2).strip()
        level = _heading_level(label)
        offset = m.start()
        found.append((offset, label, heading, level))
        seen_offsets.add(offset)

    # --- Pattern 2: ALL-CAPS headings (only if not already captured) ---
    for m in _ALLCAPS_HEADING_RE.finditer(text):
        offset = m.start()
        # Skip if a numbered heading already starts within ±5 chars
        already = any(abs(offset - o) < 5 for o in seen_offsets)
        if not already:
            heading = m.group(1).strip()
            found.append((offset, '', heading, 1))
            seen_offsets.add(offset)

    found.sort(key=lambda x: x[0])
    return found


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_chunks_structural(
    text: str,
    max_words: int = 200,
    overlap_sentences: int = 1,
    min_words: int = 20,
) -> List[str]:
    """
    Split *text* into chunks that respect the document's structural sections
    (numbered headings, ALL-CAPS headings).

    Strategy
    --------
    1. Detect all section headings and their byte offsets.
    2. Slice the text at heading boundaries → one raw "section block" per heading.
    3. If a section block exceeds *max_words*, split it further by paragraph
       boundaries (blank lines), then by sentence if still too large.
    4. If a section block is smaller than *min_words*, merge it into the
       previous chunk (avoids degenerate 2-word chunks).
    5. Apply sentence-level overlap: prepend the last *overlap_sentences*
       sentences of chunk N to chunk N+1 for retrieval continuity.

    Parameters
    ----------
    text : str
        Raw document text (paragraphs separated by blank lines).
    max_words : int
        Soft upper limit (words) for a chunk before it is sub-split.
        Default: 200.
    overlap_sentences : int
        Sentences from the tail of chunk N prepended to chunk N+1.
        Set 0 to disable. Default: 1.
    min_words : int
        Chunks shorter than this are merged into the previous one.
        Default: 20.

    Returns
    -------
    List[str]
        Chunk strings ready to embed and index.
    """

    if not text or not text.strip():
        return []

    sections = _detect_sections(text)

    # --- Build raw section blocks ---
    # Each block is (heading_prefix, body_text)
    raw_blocks: List[str] = []

    if not sections:
        # No structure detected → fall back to paragraph splitting
        raw_blocks = _split_by_paragraphs(text, max_words)
    else:
        boundaries = [s[0] for s in sections] + [len(text)]
        for i, (offset, label, heading, level) in enumerate(sections):
            block_text = text[offset:boundaries[i + 1]].strip()
            if block_text:
                raw_blocks.append(block_text)

        # Also capture any preamble text before the first heading
        preamble = text[:sections[0][0]].strip() if sections else ''
        if preamble and _word_count(preamble) >= min_words:
            raw_blocks.insert(0, preamble)

    # --- Sub-split oversized blocks ---
    chunks: List[str] = []
    for block in raw_blocks:
        if _word_count(block) <= max_words:
            chunks.append(block)
        else:
            chunks.extend(_split_by_paragraphs(block, max_words))

    # --- Merge undersized chunks ---
    merged: List[str] = []
    for chunk in chunks:
        if merged and _word_count(chunk) < min_words:
            merged[-1] = merged[-1] + ' ' + chunk
        else:
            merged.append(chunk)

    if not merged:
        return []

    # --- Apply sentence-level overlap ---
    # IMPORTANT: overlap is taken from merged[i-1] (the original chunk),
    # NOT from overlapped[-1] (which already contains a prefix from the
    # previous iteration and would cause cascading overlap growth).
    # The prefix is also word-capped at max_words // 4 to prevent sentence
    # splitter failures on structured text (lists, tables without punctuation)
    # from inflating the next chunk beyond max_words.
    if overlap_sentences > 0 and len(merged) > 1:
        max_overlap_words = max(10, max_words // 4)
        overlapped: List[str] = [merged[0]]
        for i in range(1, len(merged)):
            prev_sentences = _split_sentences(merged[i - 1])
            n = min(overlap_sentences, len(prev_sentences))
            prefix = ' '.join(prev_sentences[-n:])
            # Hard-cap the prefix so a single "sentence" that is really a
            # whole paragraph can't blow up the next chunk.
            prefix_words = prefix.split()
            if len(prefix_words) > max_overlap_words:
                prefix = ' '.join(prefix_words[-max_overlap_words:])
            overlapped.append((prefix + ' ' + merged[i]).strip())
        return overlapped

    return merged


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _word_count(s: str) -> int:
    return len(s.split())


def _split_sentences(text: str) -> List[str]:
    """Minimal sentence splitter (no NLP dependency)."""
    parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÀÈÌÒÙÁÉÍÓÚ\d])', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_by_paragraphs(text: str, max_words: int) -> List[str]:
    """
    Split *text* at blank-line paragraph boundaries.
    If a single paragraph exceeds *max_words*, split it further by sentence.
    """
    paragraphs = re.split(r'\n{2,}', text.strip())
    result: List[str] = []
    current = ''

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if _word_count(current) + _word_count(para) <= max_words:
            current = (current + '\n\n' + para).strip() if current else para
        else:
            if current:
                result.append(current)
            if _word_count(para) <= max_words:
                current = para
            else:
                # Paragraph itself too large → split by sentence
                result.extend(_split_by_sentences(para, max_words))
                current = ''

    if current:
        result.append(current)

    return result if result else [text.strip()]


def _split_by_sentences(text: str, max_words: int) -> List[str]:
    """
    Last-resort splitter: accumulate sentences up to *max_words*.
    If a single sentence still exceeds *max_words* (e.g. a table row with no
    punctuation), force-split it by words to guarantee the hard limit.
    """
    sentences = _split_sentences(text)
    result: List[str] = []
    current = ''

    for sent in sentences:
        if _word_count(current) + _word_count(sent) <= max_words:
            current = (current + ' ' + sent).strip() if current else sent
        else:
            if current:
                result.append(current)
            if _word_count(sent) <= max_words:
                current = sent
            else:
                # Sentence too long (e.g. table row without punctuation)
                # Force-split by words to guarantee the limit is respected.
                result.extend(_split_by_words(sent, max_words))
                current = ''

    if current:
        result.append(current)

    return result if result else [text.strip()]


def _split_by_words(text: str, max_words: int) -> List[str]:
    """
    Hard word-count split. Used only when no sentence boundary is available.
    Produces chunks of exactly *max_words* words, last chunk has the remainder.
    """
    words = text.split()
    return [
        ' '.join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]