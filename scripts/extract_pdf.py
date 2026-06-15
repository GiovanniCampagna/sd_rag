"""
extract_pdf.py
==============
Estrae il testo da un PDF e produce un corpus JSON pronto per SD-RAG.

Pipeline
--------
1. Estrazione  — pdfplumber per tutto. Per ogni pagina:
   · Trova le tabelle valide (>= 2 colonne).
   · Testo narrativo: page.filter() esclude i caratteri nelle bbox
     delle tabelle → righe pulite senza frammenti di celle.
   · Testo tabellare: righe "col1 | col2 | col3".
   · Righe narrative e tabelle vengono fuse e ordinate per posizione
     verticale (top), così l'ordine di lettura del documento è preservato
     anche quando una tabella e del testo coesistono sulla stessa pagina.
   · Pagine senza tabelle: page.extract_text() direttamente.

2. Pulizia     — su ogni pagina di contenuto:
   a) righe ricorrenti >= 50% delle pagine → solo prima occorrenza
   b) intestazioni tabellari ripetute (righe con '|') → rimosse del tutto (prima di (a))
   c) numeri isolati 1-3 cifre → rimossi, salvo se su riga con titolo
      maiuscolo (sarebbero numeri di sezione, non di pagina)
   d) didascalie Figura/Tabella X → rimosse
   e) righe di continuazione → unite alla riga precedente

3. Sezioni     — scansiona il testo di ogni pagina cercando il pattern
   pdfplumber per le intestazioni: riga che contiene solo
   <numero> <TITOLO IN MAIUSCOLO>. Tiene l'ultima occorrenza di ogni
   numero (corpo > indice). Restituisce (numero, titolo, pagina).

4. Corpus      — assembla ogni sezione per indici di pagina:
   · prima pagina dal titolo in poi
   · pagine intermedie intere
   · tail: testo prima del titolo successivo sul suo stesso foglio
   Cleanup finale: rimuove numeri in coda e contenuto sfuggito.

5. Output      — JSON { "TITOLO": "testo" } con riga vuota tra sezioni.

Genericità: nessuna stringa specifica del documento è hardcodata.
"""

import re
import json
import argparse
from collections import Counter

import pdfplumber


# ===========================================================================
# PARAMETRI
# ===========================================================================

TABLE_MIN_COLS           = 2
RECURRING_PAGE_FRACTION  = 0.5
TABLE_HEADER_MAX_REPEATS = 1
MAX_SECTION_NUM          = 20
MIN_WORDS                = 40
DOTS_RATIO_MAX           = 0.02
AVG_WORDS_MIN            = 7.0   # parole medie/riga — filtra liste corte (acronimi)

PAGE_NUM_RE   = re.compile(r'^\d{1,3}$')
CAPTION_RE    = re.compile(r'^(figura|figure|fig\.|tabella)\s+\d', re.IGNORECASE)
ALLCAPS_RE    = re.compile(r'^[A-ZÀÈÉÌÒÙÄ\s\-–/&]{4,}$')
SUBSECTION_RE = re.compile(r'^\d+\.\d+')

# Pattern per intestazioni di sezione nel formato pdfplumber:
# riga che contiene SOLO un numero intero seguito da titolo in maiuscolo.
# Es: "6    REGISTRAZIONI" oppure "3  RESPONSABILITÀ ED AZIONI – OPERAZIONI NORMALI"
# Nell'indice le stesse righe hanno "....." o numeri di pagina dopo il titolo
# → non corrispondono al pattern → la deduplicazione (ultima occorrenza) le scarta.
SEC_RE = re.compile(
    r'(?m)^(\d+)[ \t]+([A-ZÀÈÉÌÒÙÄ][A-ZÀÈÉÌÒÙÄ \-–/&]{2,})\s*$'
)


# ===========================================================================
# PASSO 1: Estrazione testo
# ===========================================================================

def _table_to_text(table: list[list[str | None]]) -> str:
    """
    Converte una tabella pdfplumber in righe 'col1 | col2 | col3'.

    pdfplumber spezza una cella che va a capo in più righe-tabella con le prime
    colonne vuote (es. ['', '', '', 'continua']). Una riga con prima cella vuota
    è quindi la continuazione di quella precedente: le sue celle vengono fuse
    colonna per colonna. Così si ricompongono i passi multi-linea e gli header
    spezzati (rendendoli deduplicabili). Le celle vuote in coda sono rimosse per
    non lasciare pipe spurie ('X |  |  | ').
    """
    merged: list[list[str]] = []
    for row in table:
        cells = [(c.replace('\n', ' ').strip() if c else '') for c in row]
        if not any(cells):
            continue
        # Continuazione: prima cella vuota e c'è una riga a cui agganciarsi
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
        # Rimuovi celle vuote in coda (mantieni quelle vuote interne per allineamento)
        while cells and not cells[-1]:
            cells.pop()
        if cells:
            lines.append(' | '.join(cells))
    return '\n'.join(lines)


def _not_in_bbox(obj: dict, bboxes: list[tuple]) -> bool:
    """
    True se l'oggetto (carattere PDF) NON cade dentro nessuna delle bbox.
    Usato da page.filter() per escludere il testo nelle celle delle tabelle.
    Tolleranza di 1 pt per gestire leggeri disallineamenti tipografici.
    """
    for x0, top, x1, bottom in bboxes:
        if (obj.get('x0', 0) >= x0 - 1 and
                obj.get('x1', 0) <= x1 + 1 and
                obj.get('top', 0) >= top - 1 and
                obj.get('bottom', 0) <= bottom + 1):
            return False
    return True


def extract_pages_text(pdf_path: str) -> list[str]:
    """
    Restituisce una stringa di testo per ogni pagina del PDF.

    Pagine senza tabelle: page.extract_text() diretto.
    Pagine con tabelle: righe narrative (escludendo i caratteri nelle bbox) e
    blocchi tabellari vengono raccolti con la loro coordinata verticale (top) e
    ordinati per posizione, così l'ordine di lettura resta corretto anche quando
    tabella e testo coesistono sulla stessa pagina.
    """
    pages_text: list[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Trova tabelle valide e pre-carica i dati
            valid: list[tuple] = []
            for t in page.find_tables():
                data = t.extract()
                if data and data[0] and len(data[0]) >= TABLE_MIN_COLS:
                    valid.append((t.bbox, data))

            if not valid:
                pages_text.append(page.extract_text() or '')
                continue

            # Blocchi posizionati: (top, testo).
            bboxes = [bbox for bbox, _ in valid]
            blocks: list[tuple[float, str]] = []

            # Tabelle: ancorate al top della bbox (bbox = x0, top, x1, bottom)
            for bbox, data in valid:
                table_text = _table_to_text(data)
                if table_text.strip():
                    blocks.append((bbox[1], table_text))

            # Righe narrative: escludi i caratteri dentro le bbox tabellari
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
# PASSO 2: Pulizia testo contenuto
# ===========================================================================

def _join_continuations(text: str) -> str:
    """
    Unisce le righe di continuazione alla riga precedente.
    Una riga è continuazione se inizia con lettera minuscola o virgola
    e la riga precedente non termina con punteggiatura finale.
    Righe tabellari (con '|') e intestazioni di sottosezione non vengono unite.
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
    """Applica cinque filtri di pulizia a ogni pagina di contenuto."""

    # a) Righe ricorrenti in >= RECURRING_PAGE_FRACTION delle pagine
    line_page_count: Counter = Counter()
    for page in pages_text:
        for s in {l.strip() for l in page.split('\n') if l.strip()}:
            line_page_count[s] += 1
    threshold = max(2, len(pages_text) * RECURRING_PAGE_FRACTION)
    recurring = {s for s, n in line_page_count.items() if n >= threshold}

    # b) Intestazioni tabellari (righe con '|') che si ripetono troppo
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
            # d) Didascalie figura/tabella
            if CAPTION_RE.match(s):
                continue
            # c) Numero isolato: tienilo solo se precede un titolo maiuscolo
            if PAGE_NUM_RE.match(s):
                nxt = next((lines[j].strip() for j in range(idx + 1, len(lines))
                            if lines[j].strip()), '')
                if not ALLCAPS_RE.match(nxt):
                    continue
            # b) Header tabellari ripetuti → rimossi del tutto (anche la prima
            #    occorrenza). Va PRIMA di (a): un header che ricorre in >=50%
            #    delle pagine sarebbe altrimenti intercettato da 'ricorrenti' e
            #    tenuto alla prima copia. Le righe-dato non sono ripetute → salve.
            if s in repeated_headers:
                continue
            # a) Righe ricorrenti non tabellari (header/footer di pagina) → tieni la prima
            if s in recurring:
                if s not in seen_recurring:
                    page_out.append(line)
                    seen_recurring.add(s)
                continue
            page_out.append(line)

        # e) Unisci righe di continuazione
        cleaned.append(_join_continuations('\n'.join(page_out)))

    return cleaned


# ===========================================================================
# PASSO 3: Rilevamento sezioni
# ===========================================================================

# Pattern per titoli di sezione NON numerati: riga con solo testo maiuscolo
# (es. "DEFINIZIONI", "RIFERIMENTI") — max 30 caratteri per escludere
# intestazioni di tabelle lunghe come "PIAZZOLE DI SOSTA NON ABILITATE...".
# Nessun ':' né '|' permessi per escludere etichette inline come "NOTA:".
UNNUMBERED_SEC_RE = re.compile(
    r'(?m)^([A-ZÀÈÉÌÒÙÄ][A-ZÀÈÉÌÒÙÄ \-–/&]{2,28})\s*$'
)


def detect_sections(pages_text: list[str], max_section_num: int = MAX_SECTION_NUM) -> list[tuple[int, str, int]]:
    """
    Scansiona il testo di ogni pagina cercando due tipi di intestazione:

    1. Numerata: riga con solo '<numero> <TITOLO MAIUSCOLO>'.
       Es: "3  RESPONSABILITÀ ED AZIONI – OPERAZIONI NORMALI"

    2. Non numerata: riga con solo testo maiuscolo (max 30 char).
       Es: "DEFINIZIONI", "RIFERIMENTI".
       Consente di estrarre sezioni prive di numero ma ricche di contenuto
       utile per il RAG, come le definizioni dei termini chiave.

    In entrambi i casi tiene l'ultima occorrenza (corpo > indice):
    nell'indice i titoli appaiono con "....." o numeri di pagina dopo,
    quindi NON corrispondono ai pattern e non vengono catturati lì —
    viene catturata solo l'occorrenza nel corpo del documento.

    Le sezioni senza contenuto sufficiente (INDICE, acronimi, ecc.) vengono poi
    scartate da is_informative.

    Restituisce [(numero, titolo, indice_pagina)] ordinato per pagina. I titoli
    non numerati ricevono un numero convenzionale (1000+pagina) per il solo
    ordinamento, non visibile nel corpus finale.
    """
    numbered:   dict[int, tuple[int, str, int]] = {}
    unnumbered: dict[str, tuple[int, str, int]] = {}

    for page_idx, page_text in enumerate(pages_text):

        # Sezioni numerate
        for m in SEC_RE.finditer(page_text):
            num   = int(m.group(1))
            title = m.group(2).strip()
            if num > max_section_num or len(title) < 3:
                continue
            numbered[num] = (num, title, page_idx)

        # Sezioni non numerate
        for m in UNNUMBERED_SEC_RE.finditer(page_text):
            title = m.group(1).strip()
            if len(title) < 4:
                continue
            # Ultima occorrenza vince (la prima è nell'indice con ".....",
            # che non matcha il pattern — quindi qui siamo già nel corpo)
            unnumbered[title] = (1000 + page_idx, title, page_idx)

    combined = list(numbered.values()) + list(unnumbered.values())
    return sorted(combined, key=lambda x: x[2])


# ===========================================================================
# PASSO 4: Filtro sezioni informative
# ===========================================================================

def is_informative(text: str) -> bool:
    """True se la sezione ha contenuto sufficiente per il forge di SD-RAG."""
    words = text.split()
    if len(words) < MIN_WORDS:
        return False
    if (text.count('...') * 3) / len(text) > DOTS_RATIO_MAX:
        return False
    non_empty = [l for l in text.split('\n') if l.strip()]
    if not non_empty:
        return False
    # Escludi righe pipe dal denominatore: hanno ~3-5 parole per riga e
    # abbasserebbero la media penalizzando sezioni procedurali a struttura tabellare
    narrative_lines = [l for l in non_empty if ' | ' not in l]
    if not narrative_lines:
        return True  # sezione puramente tabellare con abbastanza parole
    return len(words) / len(narrative_lines) >= AVG_WORDS_MIN


# ===========================================================================
# PASSO 5: Costruzione corpus
# ===========================================================================

def _strip_trailing_nums(text: str) -> str:
    """Rimuove righe vuote e numeri isolati in coda al testo."""
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
    Assembla ogni sezione per indici di pagina:
    · prima pagina dal titolo in poi
    · pagine intermedie intere
    · tail: testo prima del titolo successivo sul suo stesso foglio
    Cleanup finale: numeri in coda e bleed da sezioni brevi adiacenti.
    """
    corpus: dict[str, str] = {}
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

        # Cleanup finale
        section_text = _strip_trailing_nums(section_text)
        if next_title:
            leak_re = re.compile(r'\n\d*\s*\n?' + re.escape(next_title))
            m = leak_re.search(section_text)
            if m:
                section_text = _strip_trailing_nums(section_text[:m.start()])

        # Unisci le frasi spezzate ai confini di pagina: _join_continuations
        # gira per pagina durante la pulizia, ma i confini tra pagine
        # consecutive vengono assemblati in build_corpus — una seconda
        # passata sul testo finale cattura i casi rimasti.
        section_text = _join_continuations(section_text)

        if is_informative(section_text):
            corpus[title] = section_text

    return corpus


# ===========================================================================
# PASSO 6: Salvataggio
# ===========================================================================

def save_corpus(corpus: dict[str, str], output_path: str) -> None:
    """Salva il corpus JSON con riga vuota tra ogni sezione."""
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
        description='Estrae testo da un PDF strutturato e produce un corpus JSON per SD-RAG.'
    )
    parser.add_argument('pdf', help='Percorso del file PDF da elaborare')
    parser.add_argument('output', help='Percorso del file JSON di output')
    parser.add_argument(
        '--max-sections', type=int, default=MAX_SECTION_NUM,
        help=f'Numero massimo di sezione principale (default: {MAX_SECTION_NUM})'
    )
    args = parser.parse_args()

    print('1. Estrazione testo...')
    pages_text = extract_pages_text(args.pdf)
    table_pages = sum(1 for p in pages_text if ' | ' in p)
    print(f'   Pagine: {len(pages_text)}  |  con tabelle: {table_pages}')

    print('2. Pulizia testo...')
    cleaned_pages = clean_content_pages(pages_text)
    print(f'   Caratteri totali: {sum(len(p) for p in cleaned_pages)}')

    print('3. Rilevamento sezioni (su testo pulito)...')
    sections = detect_sections(cleaned_pages, max_section_num=args.max_sections)
    print(f'   Sezioni rilevate: {len(sections)}')
    for num, title, pg in sections:
        print(f'      SEC {num:2d}  pag={pg:3d}  "{title[:55]}"')

    print('4. Costruzione corpus...')
    corpus = build_corpus(cleaned_pages, sections)
    print(f'   Mantenute: {len(corpus)}  |  Scartate: {len(sections) - len(corpus)}')
    for title, body in corpus.items():
        print(f'      OK  "{title[:55]}"  ({len(body.split())} parole)')

    print('5. Salvataggio corpus...')
    save_corpus(corpus, args.output)
    print(f'   Salvato in: {args.output}')
    print('OK')