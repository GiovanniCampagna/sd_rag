"""
extract_pdf_base.py
===================
Estrazione PDF BASELINE (ingenua): dump del testo grezzo di tutte le pagine
in un'unica voce di corpus. Nessuna pulizia, nessun sezionamento, nessuna
gestione delle tabelle. Conservato come riferimento/baseline rispetto alla
pipeline strutturata di extract_pdf.py.

Uso:
    python scripts/extract_pdf_base.py <input.pdf> <output.json> [--key NOME]
"""

import json
import argparse

import fitz  # PyMuPDF


def extract_raw(pdf_path: str) -> tuple[str, int]:
    """Concatena il testo grezzo di tutte le pagine. Ritorna (testo, n_pagine)."""
    doc = fitz.open(pdf_path)
    text = ''.join(page.get_text() for page in doc)
    return text, len(doc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Estrazione PDF baseline (dump grezzo monolitico) per SD-RAG.'
    )
    parser.add_argument('pdf', help='Percorso del file PDF da elaborare')
    parser.add_argument('output', help='Percorso del file JSON di output')
    parser.add_argument(
        '--key', default='documento_prof',
        help='Chiave del corpus sotto cui salvare il testo (default: documento_prof)'
    )
    args = parser.parse_args()

    text, n_pages = extract_raw(args.pdf)
    corpus = {args.key: text}

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f'OK - {n_pages} pagine estratte')
    print(f'Caratteri totali: {len(text)}')
    print('Anteprima primi 500 caratteri:')
    print(text[:500])
