"""
extract_pdf.py
==============
Extracts the text from a PDF and produces a JSON corpus ready for SD-RAG.

Pipeline
--------
1. Extraction  -- pdfplumber for everything. For each page:
   - Find the valid tables (>= 2 columns). A table whose rows are separated by
     whitespace instead of horizontal rules (ruled columns only) is rebuilt from
     its vertical lines + the text layout (_extract_spaced_table).
   - Narrative text: page.filter() excludes the characters inside the table
     bounding boxes -> clean lines without cell fragments.
   - Tabular text: rows "col1 | col2 | col3".
   - Narrative lines and tables are merged and sorted by vertical position
     (top), so the document's reading order is preserved even when a table
     and text coexist on the same page.

2. Cleaning    -- on each content page: drop recurring header/footer lines,
   repeated table headers, isolated page numbers and Figure/Table captions, and
   join continuation lines.

3. Sections    -- primary: read the table of contents and take its CHAPTER titles
   (the entries printed in BOLD; sub-entries are not). The chapters are then
   located in the body. Fallback (no parseable ToC): detect headings by pattern
   (numbered / all-caps). Sections below SECTION_MIN_WORDS are merged into a
   neighbour; sections above SECTION_MAX_WORDS are split to fit the LLM context.

4. Corpus      -- assembles each section between consecutive headings, then caps
   oversized ones, then emits JSON { "TITLE": "text" }.

Generality: no document-specific string is hardcoded; sectioning relies on the
ToC's own bold styling, which marks chapters in virtually every document.
"""

import re
import json
import argparse
from collections import Counter

import pdfplumber


# ===========================================================================
# PARAMETERS
# ===========================================================================

TABLE_MIN_COLS           = 2
VLINE_MIN_SEGMENTS       = 10   # a vertical line repeated >= this many times is a real
                                # column separator: lets us read tables whose rows are
                                # separated by whitespace instead of horizontal rules,
                                # without mistaking plain prose for a table.
RECURRING_PAGE_FRACTION  = 0.5
TABLE_HEADER_MAX_REPEATS = 1
MAX_SECTION_NUM          = 20
MIN_WORDS                = 40
SECTION_MIN_WORDS        = 80   # a standalone section needs at least this many words;
                                # smaller (but informative) sections are merged into the
                                # previous one, so the forge does not produce tiny
                                # single-fact gold answers (e.g. a 42-word "REGISTRAZIONI").
SECTION_MAX_WORDS        = 2500 # a section larger than this is split into sub-sections,
                                # so each one fits the forge/eval context window
                                # (local models: n_ctx 5000 tok - prompt/output ~= 3500 tok
                                # ~= 2500 words). Keeps the LLM from truncating the context.
DOTS_RATIO_MAX           = 0.02
AVG_WORDS_MIN            = 7.0   # mean words/line -- filters out short lists (acronyms)

PAGE_NUM_RE   = re.compile(r'^\d{1,3}$')
CAPTION_RE    = re.compile(r'^(figura|figure|fig\.|tabella)\s+\d', re.IGNORECASE)
ALLCAPS_RE    = re.compile(r'^[A-ZÀÈÉÌÒÙÄ\s\-–/&]{4,}$')
SUBSECTION_RE = re.compile(r'^\d+\.\d+')

# Pattern for section headings in the pdfplumber format:
# a line containing ONLY an integer followed by an uppercase title.
# E.g. "6    REGISTRAZIONI" or "3  RESPONSABILITÀ ED AZIONI – OPERAZIONI NORMALI"
# In the table of contents the same lines have "....." or page numbers after the
# title -> they do not match the pattern -> dedup (last occurrence) discards them.
SEC_RE = re.compile(
    r'(?m)^(\d+)[ \t]+([A-ZÀÈÉÌÒÙÄ][A-ZÀÈÉÌÒÙÄ \-–/&]{2,})\s*$'
)


# ===========================================================================
# STEP 1: Text extraction
# ===========================================================================

def _table_to_text(table: list[list[str | None]]) -> str:
    """
    Convert a pdfplumber table into 'col1 | col2 | col3' rows.

    pdfplumber splits a wrapped cell into several table rows with the first columns
    empty (e.g. ['', '', '', 'continuation']). A row whose first cell is empty is
    therefore the continuation of the previous one: its cells are merged column by
    column. This reassembles multi-line steps and split headers (making them
    dedup-able). Trailing empty cells are removed so as not to leave spurious pipes
    ('X |  |  | ').
    """
    merged: list[list[str]] = []
    for row in table:
        cells = [(c.replace('\n', ' ').strip() if c else '') for c in row]
        if not any(cells):
            continue
        # Continuation: first cell empty and there is a row to attach to
        if merged and not cells[0]:
            prev = merged[-1]
            for i, c in enumerate(cells):
                if not c:
                    continue
                if i < len(prev):
                    prev[i] = (prev[i] + ' ' + c).strip()
                else:
                    prev.append(c)
        else:
            merged.append(cells)

    lines = []
    for cells in merged:
        # Remove trailing empty cells (keep internal empty ones for alignment)
        while cells and not cells[-1]:
            cells.pop()
        if cells:
            lines.append(' | '.join(cells))
    return '\n'.join(lines)


def _not_in_bbox(obj: dict, bboxes: list[tuple]) -> bool:
    """
    True if the object (PDF character) does NOT fall inside any of the bboxes.
    Used by page.filter() to exclude the text inside the table cells.
    1 pt tolerance to handle slight typographic misalignments.
    """
    for x0, top, x1, bottom in bboxes:
        if (obj.get('x0', 0) >= x0 - 1 and
                obj.get('x1', 0) <= x1 + 1 and
                obj.get('top', 0) >= top - 1 and
                obj.get('bottom', 0) <= bottom + 1):
            return False
    return True


def _extract_spaced_table(page) -> list[tuple]:
    """
    Fallback for tables whose COLUMNS are ruled (vertical lines) but whose ROWS are
    separated by whitespace instead of horizontal lines: find_tables() needs a grid and
    misses them. Columns are rebuilt from the vertical lines that repeat down the page
    (>= VLINE_MIN_SEGMENTS times) and rows from the text layout -- the same process as a
    ruled table, with whitespace in place of horizontal rules. Rows outside the vertical
    lines' span (running header / page number) are dropped.

    Returns [(bbox, data), ...], or [] if the page has no such ruled-column table -- so
    plain narrative pages, which lack these repeated column lines, are left untouched.
    """
    xcount = Counter(round(e['x0']) for e in page.vertical_edges)
    col_x = sorted(x for x, n in xcount.items() if n >= VLINE_MIN_SEGMENTS)
    if len(col_x) < 2:
        return []

    ys = [e[k] for e in page.vertical_edges
          if round(e['x0']) in col_x for k in ('top', 'bottom')]
    top, bottom = min(ys), max(ys)
    lines_x = sorted({round(page.bbox[0]) + 5, *col_x, round(page.bbox[2]) - 5})

    out: list[tuple] = []
    for t in page.find_tables(table_settings={
            'vertical_strategy': 'explicit', 'explicit_vertical_lines': lines_x,
            'horizontal_strategy': 'text', 'explicit_horizontal_lines': [top, bottom]}):
        # Keep only the rows inside the column-lines region (drops header/footer bleed)
        data = [cells for row, cells in zip(t.rows, t.extract())
                if top - 2 <= row.bbox[1] <= bottom + 2]
        if data and data[0] and len(data[0]) >= TABLE_MIN_COLS:
            out.append(((t.bbox[0], top, t.bbox[2], bottom), data))
    return out


def extract_pages_text(pdf_path: str) -> list[str]:
    """
    Return one text string per PDF page.

    Pages without tables: page.extract_text() directly.
    Pages with tables: narrative lines (excluding the characters inside the bboxes)
    and tabular blocks are collected with their vertical coordinate (top) and sorted
    by position, so the reading order stays correct even when a table and text
    coexist on the same page.
    """
    pages_text: list[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Find valid tables and pre-load the data
            valid: list[tuple] = []
            for t in page.find_tables():
                data = t.extract()
                if data and data[0] and len(data[0]) >= TABLE_MIN_COLS:
                    valid.append((t.bbox, data))

            if not valid:
                # No ruled grid: maybe a table with ruled columns but whitespace rows.
                valid = _extract_spaced_table(page)

            if not valid:
                pages_text.append(page.extract_text() or '')
                continue

            # Positioned blocks: (top, text).
            bboxes = [bbox for bbox, _ in valid]
            blocks: list[tuple[float, str]] = []

            # Tables: anchored to the top of the bbox (bbox = x0, top, x1, bottom)
            for bbox, data in valid:
                table_text = _table_to_text(data)
                if table_text.strip():
                    blocks.append((bbox[1], table_text))

            # Narrative lines: exclude the characters inside the table bboxes
            narrative_lines = (
                page.filter(lambda obj: _not_in_bbox(obj, bboxes))
                .extract_text_lines()
            )
            for line in narrative_lines:
                if line['text'].strip():
                    blocks.append((line['top'], line['text']))

            blocks.sort(key=lambda b: b[0])
            pages_text.append('\n'.join(text for _, text in blocks))

    return pages_text


# ===========================================================================
# STEP 2: Content text cleaning
# ===========================================================================

def _join_continuations(text: str) -> str:
    """
    Join continuation lines to the previous line.
    A line is a continuation if it starts with a lowercase letter or a comma
    and the previous line does not end with final punctuation.
    Table rows (with '|') and sub-section headings are not joined.
    """
    lines = text.split('\n')
    result: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            result.append(line)
            continue
        if (result and result[-1].strip()
                and '|' not in result[-1]
                and '|' not in s
                and not SUBSECTION_RE.match(s)
                and not SUBSECTION_RE.match(result[-1].strip())
                and result[-1].strip()[-1] not in '.!?:;'
                and (s[0].islower() or s[0] in ',;')):
            result[-1] = result[-1].rstrip() + ' ' + s
        else:
            result.append(line)
    return '\n'.join(result)


def clean_content_pages(pages_text: list[str]) -> list[str]:
    """Apply five cleaning filters to each content page."""

    # a) Lines recurring in >= RECURRING_PAGE_FRACTION of the pages
    line_page_count: Counter = Counter()
    for page in pages_text:
        for s in {l.strip() for l in page.split('\n') if l.strip()}:
            line_page_count[s] += 1
    threshold = max(2, len(pages_text) * RECURRING_PAGE_FRACTION)
    recurring = {s for s, n in line_page_count.items() if n >= threshold}

    # a2) Page-numbered header/footer: the same line up to a varying number, on most
    # pages (e.g. "... Page 149 of 342"). The exact line is never recurring (the number
    # differs each page), so count it on the digit-normalized form and remove ALL copies.
    norm_count: Counter = Counter()
    for page in pages_text:
        for s in {re.sub(r'\d+', '#', l.strip()) for l in page.split('\n') if l.strip()}:
            norm_count[s] += 1
    numbered_recurring = {s for s, n in norm_count.items() if n >= threshold}

    # b) Table headers (lines with '|') that repeat too often
    pipe_count: Counter = Counter()
    for page in pages_text:
        for line in page.split('\n'):
            s = line.strip()
            if ' | ' in s:
                pipe_count[s] += 1
    repeated_headers = {s for s, n in pipe_count.items()
                        if n > TABLE_HEADER_MAX_REPEATS}

    UNICODE_JUNK_RE = re.compile(r'[\uf000-\uf8ff]')

    seen_recurring: set[str] = set()
    cleaned: list[str] = []

    for page in pages_text:
        page = UNICODE_JUNK_RE.sub('', page)
        lines    = page.split('\n')
        page_out = []
        for idx, line in enumerate(lines):
            s = line.strip()
            if not s:
                page_out.append(line)
                continue
            # d) Figure/table captions
            if CAPTION_RE.match(s):
                continue
            # c) Isolated number: keep it only if it precedes an uppercase title
            if PAGE_NUM_RE.match(s):
                nxt = next((lines[j].strip() for j in range(idx + 1, len(lines))
                            if lines[j].strip()), '')
                if not ALLCAPS_RE.match(nxt):
                    continue
            # a2) Page-numbered footer/header (varying number) -> remove every copy
            if s not in recurring and re.sub(r'\d+', '#', s) in numbered_recurring:
                continue
            # b) Repeated table headers -> removed entirely (including the first
            #    occurrence). Must come BEFORE (a): a header recurring in >=50% of
            #    the pages would otherwise be caught by 'recurring' and kept at its
            #    first copy. Data rows are not repeated -> safe.
            if s in repeated_headers:
                continue
            # a) Recurring non-tabular lines (page header/footer) -> keep the first
            if s in recurring:
                if s not in seen_recurring:
                    page_out.append(line)
                    seen_recurring.add(s)
                continue
            page_out.append(line)

        # e) Join continuation lines
        cleaned.append(_join_continuations('\n'.join(page_out)))

    return cleaned


# ===========================================================================
# STEP 3: Section detection
#
# Primary: read the table of contents and take its CHAPTER titles -- the entries
# printed in BOLD (sub-entries are not bold). Generic for any document whose ToC
# marks chapters in bold. Fallback (no parseable ToC): detect headings by pattern.
# ===========================================================================

LEADER_RE       = re.compile(r'^(.+?)[.\s]{2,}\d+\s*$')  # ToC entry: "Title ..... 42"
TOC_MIN_LEADERS = 5                                       # >= this many leader lines = a ToC page


def _norm(s: str) -> str:
    """Lowercase and collapse whitespace, for tolerant title matching."""
    return re.sub(r'\s+', ' ', s).strip().lower()


def _is_bold(fontname: str) -> bool:
    return 'bold' in (fontname or '').lower()


def parse_toc(pdf_path: str) -> tuple[list[str], set[int]]:
    """
    Return (chapter_titles, toc_page_indices). Chapter titles are the ToC entries
    in BOLD; toc_page_indices are the pages recognised as table of contents (a page
    with >= TOC_MIN_LEADERS dot-leader lines) so they can be dropped from the body.
    """
    titles: list[str] = []
    toc_pages: set[int] = set()
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ''
            if sum(1 for l in text.split('\n') if LEADER_RE.match(l.strip())) < TOC_MIN_LEADERS:
                continue
            toc_pages.add(i)
            for ln in page.extract_text_lines():     # read fonts only on ToC pages
                m = LEADER_RE.match(ln['text'].strip())
                if not m:
                    continue
                head = [c for c in ln['chars'] if c['text'].strip(' .')][:6]
                if head and all(_is_bold(c['fontname']) for c in head):
                    title = m.group(1).strip(' .')
                    if len(title) >= 3:
                        titles.append(title)
    return titles, toc_pages


def detect_sections(cleaned_pages: list[str], toc_titles: list[str],
                    max_section_num: int = MAX_SECTION_NUM) -> list[tuple[int, str, int]]:
    """Section boundaries: from the ToC chapters if available, else by pattern."""
    if toc_titles:
        return _sections_from_toc(cleaned_pages, toc_titles)
    return _sections_from_patterns(cleaned_pages, max_section_num)


def _sections_from_toc(cleaned_pages: list[str], toc_titles: list[str]) -> list[tuple[int, str, int]]:
    """Locate each bold ToC chapter in the body -> (order, body_heading, page_index)."""
    sections: list[tuple[int, str, int]] = []
    for title in toc_titles:
        nt = _norm(title)
        for i, ptext in enumerate(cleaned_pages):
            line = next((ln for ln in ptext.split('\n') if nt and _norm(ln).startswith(nt)), None)
            if line is not None:
                sections.append((len(sections), line.strip(), i))
                break
    return sorted(sections, key=lambda s: s[2])


# Pattern for UNNUMBERED section titles: a line with only uppercase text
# (e.g. "DEFINIZIONI", "RIFERIMENTI") -- max 30 characters to exclude long
# table headers like "PIAZZOLE DI SOSTA NON ABILITATE...".
# No ':' or '|' allowed, to exclude inline labels like "NOTA:".
UNNUMBERED_SEC_RE = re.compile(
    r'(?m)^([A-ZÀÈÉÌÒÙÄ][A-ZÀÈÉÌÒÙÄ \-–/&]{2,28})\s*$'
)


def _sections_from_patterns(pages_text: list[str], max_section_num: int = MAX_SECTION_NUM) -> list[tuple[int, str, int]]:
    """
    Fallback when the document has no parseable ToC.
    Scan the text of each page looking for two kinds of heading:

    1. Numbered: a line with only '<number> <UPPERCASE TITLE>'.
       E.g. "3  RESPONSABILITÀ ED AZIONI – OPERAZIONI NORMALI"

    2. Unnumbered: a line with only uppercase text (max 30 chars).
       E.g. "DEFINIZIONI", "RIFERIMENTI".
       Lets us extract sections that have no number but are rich in content
       useful for the RAG, such as the definitions of key terms.

    In both cases it keeps the last occurrence (body > table of contents):
    in the ToC the titles appear with "....." or page numbers after them, so
    they do NOT match the patterns and are not captured there -- only the
    occurrence in the document body is captured.

    Sections without enough content (INDICE, acronyms, etc.) are then discarded
    by is_informative.

    Returns [(number, title, page_index)] sorted by page. Unnumbered titles get a
    conventional number (1000+page) for ordering only, not visible in the final corpus.
    """
    numbered:   dict[int, tuple[int, str, int]] = {}
    unnumbered: dict[str, tuple[int, str, int]] = {}

    for page_idx, page_text in enumerate(pages_text):

        # Numbered sections
        for m in SEC_RE.finditer(page_text):
            num   = int(m.group(1))
            title = m.group(2).strip()
            if num > max_section_num or len(title) < 3:
                continue
            numbered[num] = (num, title, page_idx)

        # Unnumbered sections
        for m in UNNUMBERED_SEC_RE.finditer(page_text):
            title = m.group(1).strip()
            if len(title) < 4:
                continue
            # Skip table column-headers: an all-caps line immediately followed by a
            # table row (pipe) is a header inside a table, not a section title
            # (e.g. "ATC FIS" above a phraseology table).
            next_line = page_text[m.end():].lstrip('\n').split('\n', 1)[0]
            if ' | ' in next_line:
                continue
            # Last occurrence wins (the first is in the ToC with ".....", which
            # does not match the pattern -- so here we are already in the body)
            unnumbered[title] = (1000 + page_idx, title, page_idx)

    combined = list(numbered.values()) + list(unnumbered.values())
    return sorted(combined, key=lambda x: x[2])


# ===========================================================================
# STEP 4: Informative-section filter
# ===========================================================================

def is_informative(text: str) -> bool:
    """True if the section has enough content for the SD-RAG forge."""
    words = text.split()
    if len(words) < MIN_WORDS:
        return False
    if (text.count('...') * 3) / len(text) > DOTS_RATIO_MAX:
        return False
    non_empty = [l for l in text.split('\n') if l.strip()]
    if not non_empty:
        return False
    # Exclude pipe rows from the denominator: they have ~3-5 words per line and
    # would lower the average, penalizing procedural sections with a tabular structure
    narrative_lines = [l for l in non_empty if ' | ' not in l]
    if not narrative_lines:
        return True  # purely tabular section with enough words
    return len(words) / len(narrative_lines) >= AVG_WORDS_MIN


# ===========================================================================
# STEP 5: Corpus construction
# ===========================================================================

def _strip_trailing_nums(text: str) -> str:
    """Remove blank lines and isolated numbers at the end of the text."""
    lines = text.split('\n')
    while lines and (not lines[-1].strip()
                     or PAGE_NUM_RE.match(lines[-1].strip())):
        lines.pop()
    return '\n'.join(lines).strip()


def build_corpus(
    cleaned_pages: list[str],
    sections:      list[tuple[int, str, int]]
) -> dict[str, str]:
    """
    Assemble each section by page indices:
    - first page from the title onward
    - whole intermediate pages
    - tail: text before the next title on the same sheet
    Final cleanup: trailing numbers and bleed from adjacent short sections.
    """
    corpus: dict[str, str] = {}
    last_title: str | None = None
    pending:    str = ''   # leading small section(s) to fold into the NEXT section
    total  = len(cleaned_pages)

    for i, (_, title, start_page) in enumerate(sections):
        end_page   = sections[i + 1][2] if i + 1 < len(sections) else total
        next_title = sections[i + 1][1] if i + 1 < len(sections) else None

        first = cleaned_pages[start_page]
        pos   = first.find(title)
        first_from_title = first[pos:] if pos != -1 else first

        if start_page == end_page:
            if next_title:
                nxt = first_from_title.find(next_title)
                section_text = (
                    first_from_title[:nxt] if nxt != -1 else first_from_title
                ).strip()
            else:
                section_text = first_from_title.strip()
        else:
            middle = '\n'.join(cleaned_pages[start_page + 1:end_page])
            tail = ''
            if next_title and end_page < total:
                last = cleaned_pages[end_page]
                nxt  = last.find(next_title)
                if nxt != -1:
                    tail = _strip_trailing_nums(last[:nxt])
            parts        = [p for p in [first_from_title, middle, tail] if p.strip()]
            section_text = '\n'.join(parts).strip()

        # Final cleanup
        section_text = _strip_trailing_nums(section_text)
        if next_title:
            leak_re = re.compile(r'\n\d*\s*\n?' + re.escape(next_title))
            m = leak_re.search(section_text)
            if m:
                section_text = _strip_trailing_nums(section_text[:m.start()])

        # Join sentences split across page boundaries: _join_continuations runs
        # per page during cleaning, but the boundaries between consecutive pages
        # are assembled in build_corpus -- a second pass over the final text
        # catches the remaining cases.
        section_text = _join_continuations(section_text)

        if not is_informative(section_text):
            continue   # genuine non-content (TOC bleed, dots-heavy): drop as before

        if len(section_text.split()) < SECTION_MIN_WORDS:
            # Real content, but too small to stand alone: fold it into a neighbour so it
            # is not lost and does not become a tiny single-fact gold for the forge. Its
            # heading stays inline as a sub-heading.
            if last_title is not None:
                # merge BACKWARD into the previous section
                corpus[last_title] = (corpus[last_title] + '\n\n' + section_text).strip()
            else:
                # nothing before yet: defer to merge FORWARD into the next section
                pending = (pending + '\n\n' + section_text).strip() if pending else section_text
            continue

        if pending:                     # flush deferred leading small section(s) into this one
            section_text = (pending + '\n\n' + section_text).strip()
            pending = ''
        corpus[title] = section_text
        last_title    = title

    if pending:                         # whole document was only small sections: keep them as one
        corpus[sections[0][1]] = pending

    return corpus


def cap_sections(corpus: dict[str, str], max_words: int = SECTION_MAX_WORDS) -> dict[str, str]:
    """
    Split any section longer than max_words into sub-sections of <= max_words, so
    each one fits the forge/eval context window. The cut is on line boundaries (never
    mid-line, so table rows stay intact); the parts keep the chapter title, suffixed
    " (2)", " (3)", ...
    """
    out: dict[str, str] = {}
    for title, text in corpus.items():
        if len(text.split()) <= max_words:
            out[title] = text
            continue
        pieces, part, count = [], [], 0
        for line in text.split('\n'):
            w = len(line.split())
            if count + w > max_words and part:
                pieces.append('\n'.join(part).strip())
                part, count = [], 0
            part.append(line)
            count += w
        if part:
            pieces.append('\n'.join(part).strip())
        for k, piece in enumerate(pieces, start=1):
            out[title if k == 1 else f'{title} ({k})'] = piece
    return out


# ===========================================================================
# STEP 6: Save
# ===========================================================================

def save_corpus(corpus: dict[str, str], output_path: str) -> None:
    """Save the JSON corpus with a blank line between each section."""
    items = []
    for title, body in corpus.items():
        key_j = json.dumps(title, ensure_ascii=False)
        val_j = json.dumps(body,  ensure_ascii=False, indent=2)
        val_j = '\n  '.join(val_j.splitlines())
        items.append(f'  {key_j}: {val_j}')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('{\n' + ',\n\n'.join(items) + '\n}')


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extracts text from a structured PDF and produces a JSON corpus for SD-RAG.'
    )
    parser.add_argument('pdf', help='Path of the PDF file to process')
    parser.add_argument('output', help='Path of the output JSON file')
    parser.add_argument(
        '--max-sections', type=int, default=MAX_SECTION_NUM,
        help=f'Maximum top-level section number (default: {MAX_SECTION_NUM})'
    )
    args = parser.parse_args()

    print('1. Text extraction...')
    pages_text = extract_pages_text(args.pdf)
    table_pages = sum(1 for p in pages_text if ' | ' in p)
    print(f'   Pages: {len(pages_text)}  |  with tables: {table_pages}')

    print('2. Table of contents...')
    toc_titles, toc_pages = parse_toc(args.pdf)
    print(f'   ToC pages: {sorted(toc_pages)}  |  bold chapters: {len(toc_titles)}')

    print('3. Text cleaning...')
    cleaned_pages = clean_content_pages(pages_text)
    for i in toc_pages:                 # drop the table of contents from the body
        if i < len(cleaned_pages):
            cleaned_pages[i] = ''
    print(f'   Total characters: {sum(len(p) for p in cleaned_pages)}')

    print('4. Section detection...')
    sections = detect_sections(cleaned_pages, toc_titles, max_section_num=args.max_sections)
    print(f'   Sections: {len(sections)}  ({"from ToC" if toc_titles else "from patterns"})')
    for _, title, pg in sections:
        print(f'      page={pg:3d}  "{title[:55]}"')

    print('5. Corpus construction...')
    corpus = cap_sections(build_corpus(cleaned_pages, sections))
    for title, body in corpus.items():
        print(f'      OK  "{title[:55]}"  ({len(body.split())} words)')

    print('6. Saving corpus...')
    save_corpus(corpus, args.output)
    print(f'   Saved to: {args.output}')
    print('OK')