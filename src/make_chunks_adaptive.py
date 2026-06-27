"""
Adaptive chunking: section headings act as hard boundaries, and the sub-splitting
strategy is chosen per content type (TABULAR / LIST_HEAVY / NARRATIVE).

Pipeline
--------
1. _detect_sections        -> impassable section boundaries (numbered and ALL-CAPS
                              headings, same logic as make_chunks_structural).
2. _split_into_typed_blocks -> line-by-line scan; groups consecutive lines of the
                              same type into homogeneous sub-blocks.
3. Adaptive sub-splitting per type:
   - TABULAR    -> groups of tabular_max_rows rows within tabular_max_words words
                  (a row is never split); long cells are split by sentence. No
                  header logic: rows pass through unchanged from the input.
   - LIST_HEAVY -> item by item (explicit markers or ~70% indented blocks).
   - NARRATIVE  -> semantic chunking (cosine between consecutive sentences).
4. Merge chunks below min_words (same section). No overlap: adaptive boundaries are
   already clean, so an overlap prefix would only blur the embedding.
"""

import re
from typing import List, Tuple, TYPE_CHECKING

from make_chunks_structural import (
    _detect_sections,
    _word_count,
    _split_sentences,
    _NUMBERED_HEADING_RE as _NHR,
    _ALLCAPS_HEADING_RE as _AHR,
)
from make_chunks_semantic import _semantic_subsplit

if TYPE_CHECKING:
    from ai import AI


# ---------------------------------------------------------------------------
# Constants and patterns
# ---------------------------------------------------------------------------

_LIST_ITEM_RE      = re.compile(r'^\s*(?:[a-zA-Z]\)|[•\-–—*→➔►▶])\s+\S', re.MULTILINE)
_INDENTED_ITEM_RE  = re.compile(r'^ {1,4}\S')

_INDENTED_LIST_DENSITY  = 0.70 # threshold to reclassify a NARRATIVE block as LIST_HEAVY

_TABULAR    = 'TABULAR'
_LIST_HEAVY = 'LIST_HEAVY'
_NARRATIVE  = 'NARRATIVE'


# ---------------------------------------------------------------------------
# Line / block classification
# ---------------------------------------------------------------------------

def _classify_line(line: str) -> str:
    """
    Classify a non-empty line as TABULAR, LIST_HEAVY or NARRATIVE.

    A line is TABULAR only if it is a REAL table row, recognised by explicit ' | '
    cell separators (the format the PDF extraction emits for tables). A merely
    numbered or space-aligned line is NOT a table: it is a heading/prose (NARRATIVE)
    or, with a marker, a list item (LIST_HEAVY).
    """
    if ' | ' in line:
        return _TABULAR
    if _LIST_ITEM_RE.match(line):
        return _LIST_HEAVY
    return _NARRATIVE


def _has_tabular_line(text: str) -> bool:
    """
    True if at least one non-empty line of the text is classified as TABULAR
    (i.e. contains explicit ' | ' table cells).

    Used to decide section-heading injection: a table row (e.g. "4.3.2 | ... | ...")
    is cryptic without the section title even when the chunk as a whole is labelled
    NARRATIVE. This happens with short sections (<= max_words) that start with a
    prose sentence and continue with a table (typed by their first block), and with
    chunks where a TABULAR block was merged into a NARRATIVE one during the merge phase.
    """
    return any(
        stripped and _classify_line(stripped) == _TABULAR
        for stripped in (line.strip() for line in text.split('\n'))
    )


def _is_heading_only(text: str) -> bool:
    """
    True if the text is a SINGLE line that is a heading (numbered or ALL-CAPS).

    A "bare" heading -- a section whose body is only the title, because it is
    immediately followed by a sub-section (e.g. "GESTIONE DELLE EMERGENZE" followed
    by "4.1 ...") -- would become a tiny stub chunk. It carries no sensitive content,
    only the title, so it can be merged forward into the first sub-section ACROSS the
    section boundary without breaking the privacy separation (it transfers no data,
    only title context).
    """
    s = text.strip()
    return '\n' not in s and bool(_NHR.match(s) or _AHR.match(s))


def _is_dense_indented_block(lines: List[str], density: float = _INDENTED_LIST_DENSITY) -> bool:
    """True if at least `density` of the non-empty lines start with 1-4 spaces (e.g. RIFERIMENTI)."""
    non_empty = [l for l in lines if l.strip()]
    if len(non_empty) < 3:
        return False
    indented = sum(1 for l in non_empty if _INDENTED_ITEM_RE.match(l))
    return indented / len(non_empty) >= density


def _split_into_typed_blocks(
    text: str,
    indented_list_density: float = _INDENTED_LIST_DENSITY,
) -> List[Tuple[str, str]]:
    """
    Group consecutive lines of the same type into sub-blocks.
    Empty lines are absorbed into the current block.
    NARRATIVE blocks with indent density >= indented_list_density are reclassified LIST_HEAVY.
    """
    lines = text.split('\n')
    blocks: List[Tuple[str, str]] = []
    current_type: str = ''
    current_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_lines:
                current_lines.append(line)
            continue
        line_type = _classify_line(stripped)
        if line_type != current_type:
            if current_lines:
                block_text = '\n'.join(current_lines).strip()
                if block_text:
                    blocks.append((current_type, block_text))
            current_type = line_type
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        block_text = '\n'.join(current_lines).strip()
        if block_text:
            blocks.append((current_type, block_text))

    # Reclassify NARRATIVE blocks that are indented lists without explicit markers
    processed = [
        (_LIST_HEAVY if btype == _NARRATIVE and _is_dense_indented_block(btext.split('\n'), indented_list_density) else btype, btext)
        for btype, btext in blocks
    ]
    return processed if processed else [(_NARRATIVE, text.strip())]


def _attach_list_intros(typed_blocks: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Attach the introductory line to its list: if a NARRATIVE block ends with ':' and
    is followed by a LIST_HEAVY block, ONLY its last line (the intro ending in ':') is
    prepended to the list block; the rest of the narrative stays a block of its own.
    This way a long narrative is not swallowed into the list (which would not split it
    semantically, producing a huge chunk): only the line that introduces the list goes
    with it.
    """
    out: List[Tuple[str, str]] = []
    skip = False
    for j, (ctype, ctext) in enumerate(typed_blocks):
        if skip:
            skip = False
            continue
        if (ctype == _NARRATIVE and j + 1 < len(typed_blocks)
                and typed_blocks[j + 1][0] == _LIST_HEAVY
                and ctext.rstrip().endswith(':')):
            lines = ctext.rstrip().split('\n')
            intro = lines[-1].strip()
            rest = '\n'.join(lines[:-1]).strip()
            if rest:
                out.append((_NARRATIVE, rest))
            out.append((_LIST_HEAVY, intro + '\n' + typed_blocks[j + 1][1]))
            skip = True
        else:
            out.append((ctype, ctext))
    return out


# ---------------------------------------------------------------------------
# Sub-splitting strategies
# ---------------------------------------------------------------------------

def _split_last_cell(last_cell: str, combined_prefix: str, max_words: int) -> List[str]:
    """
    Split `last_cell` into segments of max_words (minus the prefix),
    with a sentence -> clause -> word-level fallback cascade.
    """
    prefix_words = _word_count(combined_prefix)
    segment_size = max(10, max_words - prefix_words)

    for splitter in (
        lambda t: _split_sentences(t),
        lambda t: [s.strip() for s in re.split(r'(?<=[,;:])\s+', t) if s.strip()],
        lambda t: [' '.join(t.split()[i:i + segment_size])
                   for i in range(0, len(t.split()), segment_size)],
    ):
        parts = splitter(last_cell)
        if len(parts) > 1:
            break

    chunks: List[str] = []
    current: List[str] = []
    current_words = prefix_words

    for part in parts:
        pw = _word_count(part)
        if current_words + pw > max_words and current:
            chunks.append((combined_prefix + ' '.join(current)).strip())
            current = []
            current_words = prefix_words
        current.append(part)
        current_words += pw

    if current:
        chunks.append((combined_prefix + ' '.join(current)).strip())

    return chunks


def _subsplit_tabular(text: str, max_words: int, max_rows: int = 0) -> List[str]:
    """
    Group consecutive rows into chunks of at most max_rows rows (if max_rows > 0) and
    in any case <= max_words words (whichever limit is hit first). A table that fits in
    a single group stays whole. A header row, if present, stays as the first line of the
    first chunk and is NOT repeated in the following ones (repeating it in every chunk
    would only blur the embeddings). Oversize pipe-separated rows are split in the last cell.
    """
    lines = [l for l in text.split('\n') if l.strip()]
    if not lines:
        return [text.strip()] if text.strip() else []

    def _emit(line_group: List[str]) -> str:
        return '\n'.join(line_group)

    def _split_pipe_row(line: str) -> List[str]:
        if ' | ' not in line:
            return [line]
        parts = line.split(' | ')
        row_prefix = ' | '.join(parts[:-1]) + ' | '
        if _word_count(row_prefix) + _word_count(parts[-1]) <= max_words:
            return [line]
        result = _split_last_cell(parts[-1], row_prefix + ' ', max_words)
        return result if result else [line]

    chunks: List[str] = []
    current_lines: List[str] = []
    current_words = 0
    current_rows = 0

    for line in lines:
        lw = _word_count(line)
        row_limit_hit = max_rows > 0 and current_rows >= max_rows and current_lines
        if lw > max_words:
            if current_lines:
                chunks.append(_emit(current_lines))
                current_lines = []
                current_words = 0
                current_rows = 0
            chunks.extend(_split_pipe_row(line))
            continue
        if (current_words + lw > max_words or row_limit_hit) and current_lines:
            chunks.append(_emit(current_lines))
            current_lines = []
            current_words = 0
            current_rows = 0
        current_lines.append(line)
        current_words += lw
        current_rows += 1

    if current_lines:
        chunks.append(_emit(current_lines))

    return chunks if chunks else [text.strip()]


def _subsplit_list(text: str, max_words: int) -> List[str]:
    """Item by item: each item (marker + continuation lines) is an atomic unit."""
    lines = text.split('\n')
    item_groups: List[str] = []
    current: List[str] = []

    for line in lines:
        if _LIST_ITEM_RE.match(line):
            if current:
                item_groups.append('\n'.join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        item_groups.append('\n'.join(current))

    chunks: List[str] = []
    current_parts: List[str] = []
    current_words = 0

    for item in item_groups:
        iw = _word_count(item)
        if current_words + iw > max_words and current_words > 0:
            chunks.append('\n'.join(current_parts))
            current_parts = []
            current_words = 0
        current_parts.append(item)
        current_words += iw

    if current_parts:
        chunks.append('\n'.join(current_parts))

    return chunks if chunks else [text.strip()]




# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_chunks_adaptive(
    text: str,
    ai: 'AI',
    max_words: int = 200,
    min_words: int = 25,
    similarity_threshold: float = 0.45,
    min_sentences: int = 1,
    max_sentences: int = 8,
    tabular_max_rows: int = 3,
    tabular_max_words: int = 200,
    inject_section_heading: bool = True,
    indented_list_density: float = _INDENTED_LIST_DENSITY,
) -> List[str]:
    """
    Adaptive chunking for mixed procedural documents (tables, lists, prose).

    Parameters
    ----------
    text                  : raw document text
    ai                    : AI object (used only for embed() on NARRATIVE blocks)
    max_words             : soft word limit per chunk (default 200)
    min_words             : chunks below this threshold are merged into a neighbour (default 25)
    similarity_threshold  : cosine threshold for a semantic cut (default 0.45)
    min_sentences         : minimum sentences before a semantic cut (default 1)
    max_sentences         : maximum sentences per NARRATIVE chunk (default 8)
    tabular_max_rows      : maximum rows per TABULAR chunk (0 = word limit only).
                            Groups a few coherent steps per chunk: focused on the
                            specific values but not fragmented to a single row (default 3)
    tabular_max_words     : word cap for a TABULAR chunk, a safety limit that fires
                            before tabular_max_rows for very long rows (default 200)
    inject_section_heading: if True, prepend the section title to chunks that contain
                            table rows -- both TABULAR-labelled ones and NARRATIVE ones
                            that nevertheless contain a table (short sections typed by
                            their first block, or a TABULAR->NARRATIVE merge): table rows
                            are cryptic without context. Lists and pure prose stay without
                            it, since they are self-contained (the list intro is attached
                            by _attach_list_intros) (default True)
    indented_list_density : fraction of indented lines above which a NARRATIVE block
                            is reclassified as LIST_HEAVY (default 0.70)
    """
    if not text or not text.strip():
        return []

    # --- Phase 1: structural skeleton ---
    sections = _detect_sections(text)
    # section_blocks: (section_id, heading, block_text)
    section_blocks: List[Tuple[int, str, str]] = []

    if not sections:
        section_blocks.append((0, '', text.strip()))
    else:
        preamble = text[:sections[0][0]].strip()
        if preamble and _word_count(preamble) >= min_words:
            section_blocks.append((0, '', preamble))

        boundaries = [s[0] for s in sections] + [len(text)]
        for i, (offset, _label, heading, _level) in enumerate(sections):
            block = text[offset:boundaries[i + 1]].strip()
            if not block:
                continue

            # Drop the heading line from the body (avoids mis-classification)
            block_lines = block.split('\n')
            if len(block_lines) > 1:
                first = block_lines[0].strip()
                if _NHR.match(first) or _AHR.match(first):
                    body = '\n'.join(block_lines[1:]).strip()
                    if body:
                        block = body

            # Sections that are too short: merge into the previous one (avoid stub chunks)
            if section_blocks and _word_count(block) < min_words:
                prev_sid, prev_heading, prev_block = section_blocks[-1]
                section_blocks[-1] = (prev_sid, prev_heading, prev_block + '\n\n' + block)
            else:
                section_blocks.append((i + 1, heading, block))

    # --- Phases 2+3: typed-block scan and adaptive sub-splitting ---
    raw_chunks: List[Tuple[int, str, str]] = []

    for section_id, _section_heading, block in section_blocks:
        if _word_count(block) <= max_words:
            first_type = _split_into_typed_blocks(block, indented_list_density)[0][0]
            raw_chunks.append((section_id, first_type, block))
            continue

        typed_blocks = _attach_list_intros(
            _split_into_typed_blocks(block, indented_list_density))
        for content_type, typed_text in typed_blocks:
            # Tables always go through the tabular sub-split: group by rows
            # (tabular_max_rows) within a word cap (tabular_max_words). This way
            # each chunk keeps a few coherent steps together, without fragmenting
            # to a single row nor diluting the specific values in too large a block.
            if content_type == _TABULAR:
                for chunk in _subsplit_tabular(typed_text, tabular_max_words, tabular_max_rows):
                    if chunk.strip():
                        raw_chunks.append((section_id, content_type, chunk.strip()))
                continue

            if _word_count(typed_text) <= max_words:
                raw_chunks.append((section_id, content_type, typed_text))
                continue

            if content_type == _LIST_HEAVY:
                sub = _subsplit_list(typed_text, max_words)
            else:
                sub = _semantic_subsplit(
                    typed_text, ai=ai, max_words=max_words,
                    similarity_threshold=similarity_threshold,
                    min_sentences=min_sentences, max_sentences=max_sentences,
                )

            for chunk in sub:
                if chunk.strip():
                    raw_chunks.append((section_id, content_type, chunk.strip()))

    # --- Phase 4: merge chunks that are too short (same section) ---
    merged: List[Tuple[int, str, str]] = []
    for section_id, content_type, chunk_text in raw_chunks:
        if (merged and merged[-1][0] == section_id
                and _word_count(chunk_text) < min_words):
            prev_sid, prev_type, prev_text = merged[-1]
            # Separator: a newline when one of the two chunks is structured
            # (table/list), so rows and items stay on distinct lines and the
            # embedding keeps the structure; a space for prose that flows.
            sep = '\n' if (prev_type in (_TABULAR, _LIST_HEAVY)
                           or content_type in (_TABULAR, _LIST_HEAVY)) else ' '
            merged[-1] = (prev_sid, prev_type, prev_text + sep + chunk_text)
        else:
            merged.append((section_id, content_type, chunk_text))

    # --- Phase 4 (forward): start-of-section orphans ---
    # A chunk below min_words that could NOT merge into the previous one (because it is
    # the FIRST of its section: backward merge does not cross section boundaries, for the
    # privacy guarantee) is merged into the NEXT chunk. Two cases:
    #   - same section: e.g. a short intro that qualifies the heading
    #     ("(diametro fino a ~3 m...)") is not left isolated, staying inside the section;
    #   - "bare" heading (section title with no body, because immediately followed by a
    #     sub-section, e.g. "GESTIONE DELLE EMERGENZE" before "4.1 ..."): merged into the
    #     first sub-section ACROSS the section boundary -- a title transfers no sensitive
    #     content (see _is_heading_only).
    # If there is no suitable successor, it stays as is.
    forwarded: List[Tuple[int, str, str]] = []
    i = 0
    while i < len(merged):
        sid, ctype, text = merged[i]
        heading_only = _is_heading_only(text)
        if (_word_count(text) < min_words and i + 1 < len(merged)
                and (merged[i + 1][0] == sid or heading_only)):
            nsid, ntype, ntext = merged[i + 1]
            sep = '\n' if (heading_only or ctype in (_TABULAR, _LIST_HEAVY)
                           or ntype in (_TABULAR, _LIST_HEAVY)) else ' '
            merged[i + 1] = (nsid, ntype, text + sep + ntext)
            i += 1
            continue
        forwarded.append(merged[i])
        i += 1
    merged = forwarded

    if not merged:
        return []

    # --- Phase 4b: section heading injection ---
    # Prepend the section title to a chunk if inject_section_heading=True and the chunk
    # does not already start with that heading (avoids duplicates on small sections that
    # are not split and keep the heading line in their text).
    if inject_section_heading:
        # Build a section_id -> heading map
        sid_to_heading = {i + 1: heading for i, (_, _, heading, _) in enumerate(sections)} if sections else {}
        updated: List[Tuple[int, str, str]] = []
        for section_id, content_type, chunk_text in merged:
            heading = sid_to_heading.get(section_id, '')
            # The section heading is injected into chunks that contain table rows:
            # there the rows (e.g. "4.3.2 | ... | ...") are cryptic without the section
            # context. The chunk's content_type is not enough: a short section
            # (<= max_words) is typed by its first block, so it may end up NARRATIVE
            # while containing a table, and a TABULAR block may have been merged into a
            # NARRATIVE one -- hence we also look inside the text with _has_tabular_line.
            # Lists and pure prose stay without a heading (self-contained, see
            # _attach_list_intros), which would only blur the embedding.
            if (heading and not chunk_text.startswith(heading)
                    and (content_type == _TABULAR
                         or _has_tabular_line(chunk_text))):
                chunk_text = heading + '\n' + chunk_text
            updated.append((section_id, content_type, chunk_text))
        merged = updated

    # No overlap: adaptive boundaries (table row, list item, semantic cut) are already
    # clean. Re-pasting the tail of the previous chunk recovers nothing at the edges and
    # only blurs the new chunk's embedding, hurting retrieval. Sections therefore stay
    # clean and disjoint.
    return [t for _sid, _ctype, t in merged]
