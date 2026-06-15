"""
Adaptive chunking: sezioni strutturali come confini duri, strategia di
sub-splitting scelta per tipo di contenuto (TABULAR / LIST_HEAVY / NARRATIVE).

Pipeline
--------
1. _detect_sections  → confini di sezione invalicabili (intestazioni numerate
                        e ALL-CAPS, stessa logica di make_chunks_structural).
2. _split_into_typed_blocks  → scansione riga per riga; raggruppa righe
   consecutive dello stesso tipo in sotto-blocchi omogenei.
3. Sub-splitting adattivo per tipo:
   - TABULAR    → gruppi di tabular_max_rows righe entro tabular_max_words parole
                  (header ripetuto per gruppo, mai una riga spezzata); celle lunghe
                  divise per frase.
   - LIST_HEAVY → voce per voce (marker espliciti o blocchi rientrati ~70%).
   - NARRATIVE  → semantic chunking (coseno tra frasi consecutive).
4. Merge chunk sotto min_words (stesso tipo/sezione). Nessun overlap: i confini
   adattivi sono gia' netti, l'overlap sporcherebbe solo l'embedding.
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
# Costanti e pattern
# ---------------------------------------------------------------------------

_TABULAR_ROW_RE    = re.compile(r'^\s*\d+\.\d+(?:\.\d+)*\s+\S', re.MULTILINE)
_MULTI_COLUMN_RE   = re.compile(r'\S[ \t]{3,}\S')
_MULTI_COLUMN_2_RE = re.compile(r'\S[ \t]{2,}\S')
_LIST_ITEM_RE      = re.compile(r'^\s*(?:[a-zA-Z]\)|[•\-–—*→➔►▶])\s+\S', re.MULTILINE)
_INDENTED_ITEM_RE  = re.compile(r'^ {1,4}\S')

_HEADING_LIKE_MAX_WORDS = 10   # riga numerata con ≤ N parole → intestazione, non tabella
_INDENTED_LIST_DENSITY  = 0.70 # soglia per riclassificare blocco NARRATIVE → LIST_HEAVY

_TABULAR    = 'TABULAR'
_LIST_HEAVY = 'LIST_HEAVY'
_NARRATIVE  = 'NARRATIVE'


# ---------------------------------------------------------------------------
# Classificazione riga / blocco
# ---------------------------------------------------------------------------

def _classify_line(line: str, heading_like_max_words: int = _HEADING_LIKE_MAX_WORDS) -> str:
    """Classifica una riga non vuota come TABULAR, LIST_HEAVY o NARRATIVE."""
    if ' | ' in line:
        return _TABULAR
    # Intestazione numerata breve (es. "3.2 Prescrizioni…"): no celle → NARRATIVE
    if (_TABULAR_ROW_RE.match(line)
            and not _MULTI_COLUMN_2_RE.search(line)
            and len(line.split()) <= heading_like_max_words):
        return _NARRATIVE
    if _TABULAR_ROW_RE.match(line) or _MULTI_COLUMN_RE.search(line):
        return _TABULAR
    if _LIST_ITEM_RE.match(line):
        return _LIST_HEAVY
    return _NARRATIVE


def _is_dense_indented_block(lines: List[str], density: float = _INDENTED_LIST_DENSITY) -> bool:
    """True se almeno `density` delle righe non vuote iniziano con 1-4 spazi (es. RIFERIMENTI)."""
    non_empty = [l for l in lines if l.strip()]
    if len(non_empty) < 3:
        return False
    indented = sum(1 for l in non_empty if _INDENTED_ITEM_RE.match(l))
    return indented / len(non_empty) >= density


def _looks_like_table_header(row: str) -> bool:
    """True se la riga è un'intestazione pipe con celle brevi (≤5 parole ciascuna)."""
    if ' | ' not in row:
        return False
    cells = [c.strip() for c in row.split(' | ')]
    return len(cells) >= 2 and all(len(c.split()) <= 5 for c in cells)


def _split_into_typed_blocks(
    text: str,
    heading_like_max_words: int = _HEADING_LIKE_MAX_WORDS,
    indented_list_density: float = _INDENTED_LIST_DENSITY,
) -> List[Tuple[str, str]]:
    """
    Raggruppa righe consecutive dello stesso tipo in sotto-blocchi.
    Righe vuote assorbite nel blocco corrente.
    Blocchi NARRATIVE con densità rientrata >= indented_list_density riclassificati LIST_HEAVY.
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
        line_type = _classify_line(stripped, heading_like_max_words)
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

    # Riclassifica blocchi NARRATIVE che sono liste rientrate senza marker
    processed = [
        (_LIST_HEAVY if btype == _NARRATIVE and _is_dense_indented_block(btext.split('\n'), indented_list_density) else btype, btext)
        for btype, btext in blocks
    ]
    return processed if processed else [(_NARRATIVE, text.strip())]


# ---------------------------------------------------------------------------
# Strategie di sub-splitting
# ---------------------------------------------------------------------------

def _split_last_cell(last_cell: str, combined_prefix: str, max_words: int) -> List[str]:
    """
    Spezza `last_cell` in segmenti da max_words (dedotto il prefisso),
    con fallback frase → clausola → word-level.
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


def _subsplit_tabular(text: str, max_words: int, min_words: int, max_rows: int = 0) -> List[str]:
    """
    Raggruppa righe consecutive in chunk di al piu' max_rows righe (se max_rows>0)
    e comunque <= max_words parole (scatta il primo dei due limiti), header (se
    presente) ripetuto in testa a ogni gruppo. Una tabella che entra in un solo
    gruppo resta intera. Righe pipe-separate oversize sono spezzate nell'ultima cella.
    """
    lines = [l for l in text.split('\n') if l.strip()]
    if not lines:
        return [text.strip()] if text.strip() else []

    header = ''
    if _looks_like_table_header(lines[0]):
        header = lines[0]
        lines = lines[1:]
        if not lines:
            return [header]

    def _emit(line_group: List[str]) -> str:
        block = '\n'.join(line_group)
        return (header + '\n' + block).strip() if header else block

    def _split_pipe_row(line: str) -> List[str]:
        if ' | ' not in line:
            return [((header + '\n' + line).strip() if header else line)]
        parts = line.split(' | ')
        row_prefix = ' | '.join(parts[:-1]) + ' | '
        combined = (header + '\n' + row_prefix).strip('\n') if header else row_prefix
        if _word_count(combined) + _word_count(parts[-1]) <= max_words:
            return [((header + '\n' + line).strip('\n') if header else line)]
        result = _split_last_cell(parts[-1], combined + ' ', max_words)
        return result if result else [((header + '\n' + line).strip('\n') if header else line)]

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


def _subsplit_list(text: str, max_words: int, min_words: int) -> List[str]:
    """Voce per voce: ogni item (marker + continuazioni) è unità atomica."""
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
    min_words: int = 20,
    similarity_threshold: float = 0.45,
    min_sentences: int = 1,
    max_sentences: int = 8,
    tabular_max_rows: int = 3,
    tabular_max_words: int = 200,
    inject_section_heading: bool = True,
    heading_like_max_words: int = _HEADING_LIKE_MAX_WORDS,
    indented_list_density: float = _INDENTED_LIST_DENSITY,
) -> List[str]:
    """
    Chunking adattivo su documenti procedurali misti (tabelle, liste, prosa).

    Parametri
    ---------
    text                  : testo grezzo del documento
    ai                    : oggetto AI (usato solo per embed() sui blocchi NARRATIVE)
    max_words             : limite morbido di parole per chunk (default 200)
    min_words             : chunk sotto questa soglia vengono uniti al precedente (default 20)
    similarity_threshold  : soglia coseno per taglio semantico (default 0.45)
    min_sentences         : frasi minime prima di un taglio semantico (default 1)
    max_sentences         : frasi massime per chunk NARRATIVE (default 8)
    tabular_max_rows      : righe massime per chunk TABULAR (0 = solo limite parole).
                            Raggruppa pochi step coerenti per chunk: focalizzato sui
                            valori puntuali ma non frammentato a riga singola (default 3)
    tabular_max_words     : tetto di parole per un chunk TABULAR, limite di sicurezza che
                            scatta prima di tabular_max_rows per righe molto lunghe
                            (default 200)
    inject_section_heading: se True, prefissa ogni chunk con il titolo della sezione
                            di appartenenza per ancorare il retrieval (default True)
    heading_like_max_words: righe numerate con <= N parole sono trattate come
                            intestazioni (NARRATIVE) e non come righe tabellari (default 10)
    indented_list_density : frazione di righe rientrate oltre la quale un blocco
                            NARRATIVE viene riclassificato come LIST_HEAVY (default 0.70)
    """
    if not text or not text.strip():
        return []

    # --- Fase 1: scheletro strutturale ---
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

            # Rimuovi la riga di intestazione dal body (evita classificazione errata)
            block_lines = block.split('\n')
            if len(block_lines) > 1:
                first = block_lines[0].strip()
                if _NHR.match(first) or _AHR.match(first):
                    body = '\n'.join(block_lines[1:]).strip()
                    if body:
                        block = body

            # Sezioni troppo corte: unisci alla precedente (evita chunk-stub)
            if section_blocks and _word_count(block) < min_words:
                prev_sid, prev_heading, prev_block = section_blocks[-1]
                section_blocks[-1] = (prev_sid, prev_heading, prev_block + '\n\n' + block)
            else:
                section_blocks.append((i + 1, heading, block))

    # --- Fase 2+3: typed-block scan e sub-splitting adattivo ---
    raw_chunks: List[Tuple[int, str, str]] = []

    for section_id, section_heading, block in section_blocks:
        if _word_count(block) <= max_words:
            first_type = _split_into_typed_blocks(block, heading_like_max_words, indented_list_density)[0][0]
            raw_chunks.append((section_id, first_type, block))
            continue

        for content_type, typed_text in _split_into_typed_blocks(block, heading_like_max_words, indented_list_density):
            # Le tabelle passano sempre dal sub-split tabellare: raggruppa per righe
            # (tabular_max_rows) entro un tetto di parole (tabular_max_words). Cosi'
            # ogni chunk tiene insieme pochi step coerenti, senza frammentare a riga
            # singola ne' diluire i valori puntuali in un blocco troppo grande.
            if content_type == _TABULAR:
                for chunk in _subsplit_tabular(typed_text, tabular_max_words, min_words, tabular_max_rows):
                    if chunk.strip():
                        raw_chunks.append((section_id, content_type, chunk.strip()))
                continue

            if _word_count(typed_text) <= max_words:
                raw_chunks.append((section_id, content_type, typed_text))
                continue

            if content_type == _LIST_HEAVY:
                sub = _subsplit_list(typed_text, max_words, min_words)
            else:
                sub = _semantic_subsplit(
                    typed_text, ai=ai, max_words=max_words,
                    similarity_threshold=similarity_threshold,
                    min_sentences=min_sentences, max_sentences=max_sentences,
                )

            for chunk in sub:
                if chunk.strip():
                    raw_chunks.append((section_id, content_type, chunk.strip()))

    # --- Fase 4: unione chunk troppo corti (stessa sezione) ---
    merged: List[Tuple[int, str, str]] = []
    for section_id, content_type, chunk_text in raw_chunks:
        if (merged and merged[-1][0] == section_id
                and _word_count(chunk_text) < min_words):
            prev_sid, prev_type, prev_text = merged[-1]
            merged[-1] = (prev_sid, prev_type, prev_text + ' ' + chunk_text)
        else:
            merged.append((section_id, content_type, chunk_text))

    if not merged:
        return []

    # --- Fase 4b: section heading injection ---
    # Prefissa ogni chunk con il titolo della sezione se inject_section_heading=True
    # e il chunk non inizia già con quell'intestazione (evita duplicati su sezioni
    # piccole che non vengono splittate e conservano la riga di heading nel testo).
    if inject_section_heading:
        # Costruisce mappa section_id → heading
        sid_to_heading = {i + 1: heading for i, (_, _, heading, _) in enumerate(sections)} if sections else {}
        updated: List[Tuple[int, str, str]] = []
        for section_id, content_type, chunk_text in merged:
            heading = sid_to_heading.get(section_id, '')
            if heading and not chunk_text.startswith(heading):
                chunk_text = heading + '\n' + chunk_text
            updated.append((section_id, content_type, chunk_text))
        merged = updated

    # Nessun overlap: i confini adattivi (riga di tabella, voce di lista, taglio
    # semantico) sono gia' netti. Ri-incollare la coda del chunk precedente non
    # recupera nulla ai bordi e sporca soltanto l'embedding del nuovo chunk,
    # peggiorando il retrieval. Le sezioni restano quindi pulite e disgiunte.
    return [t for _sid, _ctype, t in merged]
