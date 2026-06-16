import re
from typing import List, Tuple


_NUMBERED_HEADING_RE = re.compile(
    r'^\s*(?:Art\.?\s*)?(\d+(?:\.\d+)*)[\.\s\-–—]+([A-ZÀ-Ùa-zà-ù].{2,})',
    re.MULTILINE,
)
_ALLCAPS_HEADING_RE = re.compile(
    r'^\s*([A-ZÀÈÌÒÙÁÉÍÓÚ][A-ZÀÈÌÒÙÁÉÍÓÚ\s\-–—]{4,}[A-ZÀÈÌÒÙÁÉÍÓÚ])\s*$',
    re.MULTILINE,
)


def _heading_level(label: str) -> int:
    return label.count('.') + 1


def _detect_sections(text: str) -> List[Tuple[int, str, str, int]]:
    """Return (char_offset, label, heading_text, level) for every section heading."""
    found: List[Tuple[int, str, str, int]] = []
    seen: set = set()

    for m in _NUMBERED_HEADING_RE.finditer(text):
        label, heading = m.group(1), m.group(2).strip()
        found.append((m.start(), label, heading, _heading_level(label)))
        seen.add(m.start())

    for m in _ALLCAPS_HEADING_RE.finditer(text):
        if not any(abs(m.start() - o) < 5 for o in seen):
            found.append((m.start(), '', m.group(1).strip(), 1))
            seen.add(m.start())

    found.sort(key=lambda x: x[0])
    return found


def _word_count(s: str) -> int:
    return len(s.split())



# Abbreviazioni comuni nei documenti procedurali/normativi italiani che non devono
# triggerare un taglio di frase (es. "par. 3.2", "art. 4", "Reg. (UE) n. 139/2014").
_ABBREV_RE = re.compile(
    r'\b(par|art|n|cfr|rev|agg|es|ecc|pag|fig|tab|vol|vs|all|lett|comma|cap|sez|'
    r'p\.es|S\.p\.A|S\.r\.l|S\.n\.c|S\.a\.s|D\.Lgs|D\.M|D\.P\.R|L)\.'
    r'(?=\s)',
    re.IGNORECASE,
)
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+(?=[A-ZÀÈÌÒÙÁÉÍÓÚ\d\"])')


def _split_sentences(text: str) -> List[str]:
    """
    Sentence splitter condiviso da tutti i chunker.

    Gestisce:
    - Soft line-wrap da PDF (singolo \\n dentro una frase → spazio)
    - Abbreviazioni comuni (par., art., n., cfr., D.Lgs., S.p.A. ecc.)
      che non devono spezzare la frase
    - Separatori di paragrafo (\\n\\n) come confini duri tra frasi
    - Righe pipe (col1 | col2 | col3) trattate come unità atomiche:
      ogni riga pipe diventa il proprio paragrafo così non viene mai
      fusa con la riga successiva dal collasso dei soft wrap
    """
    text = text.strip()
    # Righe pipe → confini duri (ogni riga pipe resta atomica)
    lines = text.split('\n')
    normalized: List[str] = []
    for line in lines:
        normalized.append(line)
        if ' | ' in line:
            normalized.append('')
    text = '\n'.join(normalized)
    # Proteggi i parabreak prima di collassare i soft wrap
    text = re.sub(r'\n{2,}', '\x00', text)
    text = text.replace('\n', ' ')
    text = text.replace('\x00', '\n\n')

    # Maschera temporaneamente i punti di abbreviazione con un placeholder
    # in modo che il pattern di frase non ci cada sopra
    text = _ABBREV_RE.sub(lambda m: m.group(0)[:-1] + '\x01', text)

    result: List[str] = []
    for para in text.split('\n\n'):
        para = para.strip()
        if not para:
            continue
        for sent in _SENTENCE_RE.split(para):
            sent = sent.replace('\x01', '.').strip()
            if sent:
                result.append(sent)

    return result


def _split_by_paragraphs(text: str, max_words: int) -> List[str]:
    """Split at blank-line boundaries; recurse to sentence level if still too large."""
    result: List[str] = []
    current = ''
    for para in re.split(r'\n{2,}', text.strip()):
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
                result.extend(_split_by_sentences(para, max_words))
                current = ''
    if current:
        result.append(current)
    return result if result else [text.strip()]


def _split_by_sentences(text: str, max_words: int) -> List[str]:
    """Accumulate sentences up to max_words; word-split any sentence that still overflows."""
    result: List[str] = []
    current = ''
    for sent in _split_sentences(text):
        if _word_count(current) + _word_count(sent) <= max_words:
            current = (current + ' ' + sent).strip() if current else sent
        else:
            if current:
                result.append(current)
            if _word_count(sent) <= max_words:
                current = sent
            else:
                words = sent.split()
                result.extend(' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words))
                current = ''
    if current:
        result.append(current)
    return result if result else [text.strip()]


def make_chunks_structural(
    text: str,
    max_words: int = 200,
    min_words: int = 25,
) -> List[str]:
    """
    Split text into chunks that respect document structure (numbered and ALL-CAPS headings).

    1. Detect section headings → slice into section blocks.
    2. Sub-split oversized blocks by paragraph, then sentence.
    3. Merge chunks shorter than min_words into the previous one.

    No sentence overlap: structural boundaries are clean, so an overlap prefix
    would only blur chunk embeddings and hurt retrieval.
    """
    if not text or not text.strip():
        return []

    sections = _detect_sections(text)

    raw_blocks: List[str] = []
    if not sections:
        raw_blocks = _split_by_paragraphs(text, max_words)
    else:
        preamble = text[:sections[0][0]].strip()
        if preamble and _word_count(preamble) >= min_words:
            raw_blocks.append(preamble)
        boundaries = [s[0] for s in sections] + [len(text)]
        for i, (offset, *_) in enumerate(sections):
            block = text[offset:boundaries[i + 1]].strip()
            if block:
                raw_blocks.append(block)

    # Sub-split oversized blocks
    chunks: List[str] = []
    for block in raw_blocks:
        if _word_count(block) <= max_words:
            chunks.append(block)
        else:
            chunks.extend(_split_by_paragraphs(block, max_words))

    # Merge undersized chunks
    merged: List[str] = []
    for chunk in chunks:
        if merged and _word_count(chunk) < min_words:
            merged[-1] += ' ' + chunk
        else:
            merged.append(chunk)

    if not merged:
        return []

    # Nessun overlap: i confini strutturali sono netti, l'overlap sporcherebbe
    # solo l'embedding del chunk successivo.
    return merged
