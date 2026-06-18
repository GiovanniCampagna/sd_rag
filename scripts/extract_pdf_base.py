"""
extract_pdf_base.py
===================
BASELINE (naive) PDF extraction: dumps the raw text of all pages into a single
corpus entry. No cleaning, no sectioning, no table handling. Kept as a
reference/baseline against the structured pipeline of extract_pdf.py.

Usage:
    python scripts/extract_pdf_base.py <input.pdf> <output.json> [--key NAME]
"""

import json
import argparse

import fitz  # PyMuPDF


def extract_raw(pdf_path: str) -> tuple[str, int]:
    """Concatenate the raw text of all pages. Returns (text, n_pages)."""
    doc = fitz.open(pdf_path)
    text = ''.join(page.get_text() for page in doc)
    return text, len(doc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Baseline PDF extraction (monolithic raw dump) for SD-RAG.'
    )
    parser.add_argument('pdf', help='Path of the PDF file to process')
    parser.add_argument('output', help='Path of the output JSON file')
    parser.add_argument(
        '--key', default='documento_prof',
        help='Corpus key under which to save the text (default: documento_prof)'
    )
    args = parser.parse_args()

    text, n_pages = extract_raw(args.pdf)
    corpus = {args.key: text}

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f'OK - {n_pages} pages extracted')
    print(f'Total characters: {len(text)}')
    print('Preview of the first 500 characters:')
    print(text[:500])
