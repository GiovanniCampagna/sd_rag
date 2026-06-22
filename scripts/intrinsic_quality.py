"""
INTRINSIC chunk-quality metrics (Step 1): they measure the chunking "refinement" by
looking at the chunks themselves, WITHOUT a query and WITHOUT LLM generation -- so
without the noise introduced by the small model (which flattens end-to-end metrics).

Indicators reported, per method:
  - #chunk, words/chunk, tokens/chunk : granularity and size (context cost).
  - rows_intact   : fraction of the corpus table rows preserved WHOLE inside a chunk.
                    It is adaptive's distinctive trait: it never splits a row (it only
                    splits in the last cell when oversize).
  - lists_intact  : fraction of the corpus list items preserved WHOLE.
  - clean_bounds  : fraction of chunks that start and end on a clean boundary
                    (sentence/row/item). WARNING: it rewards LARGE chunks, so it
                    favours semantic; do NOT use it alone as pro-adaptive evidence.
  - coherence     : mean cosine between adjacent sentences inside the chunk. Same
                    warning: it favours large/prose chunks.

Reading for the thesis: structural fidelity (rows/lists intact) is adaptive's strength,
especially when read TOGETHER with compactness (tokens/chunk): maximum fidelity at the
minimum context cost. clean_bounds/coherence are reported only for completeness and
honesty (they go against adaptive).

Usage:
  env/Scripts/python.exe scripts/intrinsic_quality.py \
      --corpus datasets/corpus.json --ai configs/ai.config.nomic.only.json
"""
import sys, os, json, re, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np

from ai import AI
from make_chunks import make_chunks
from make_chunks_semantic import make_chunks_semantic
from make_chunks_structural import make_chunks_structural, _split_sentences
from make_chunks_structural_semantic import make_chunks_structural_semantic
from make_chunks_adaptive import make_chunks_adaptive

_MARKER_START = re.compile(r'^\s*(?:[a-zA-Z]\)|[•\-–—*→]|\d)')
_LIST_ITEM = re.compile(r'^\s*(?:[a-zA-Z]\)|[•\-–—*→])\s+\S')

def methods(ai):
    return {
        'baseline':   lambda t: make_chunks(t, 60, 0),
        'semantic':   lambda t: make_chunks_semantic(t, ai=ai, max_words=300),
        'structural': lambda t: make_chunks_structural(t, max_words=200, min_words=25),
        'struct-sem': lambda t: make_chunks_structural_semantic(t, ai=ai, max_words=200, min_words=25),
        'adaptive':   lambda t: make_chunks_adaptive(t, ai=ai, max_words=200, min_words=25,
                                                     tabular_max_rows=3, tabular_max_words=200),
    }

def clean_start(c):
    s = c.lstrip()
    return bool(s) and (s[0].isupper() or bool(_MARKER_START.match(s)) or s[0] in '“"')
def clean_end(c):
    s = c.rstrip()
    if not s: return False
    if s[-1] in '.!?:;)”"': return True
    return ' | ' in s.split('\n')[-1]   # table row = atomic unit

def coherence(c, ai):
    sents = [x for x in _split_sentences(c) if x.strip()]
    if len(sents) < 2: return None
    emb = ai.e.embed(sents)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return float(np.mean([emb[i] @ emb[i+1] for i in range(len(sents)-1)]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--ai', default='configs/ai.config.nomic.only.json')
    ap.add_argument('--out', default='results/tier1_intrinsic.json',
                    help='where to save the results (JSON + a .md table alongside)')
    args = ap.parse_args()
    ai = AI(*json.loads(open(args.ai).read()))
    corpus = json.load(open(args.corpus, encoding='utf-8'))
    texts = [v if isinstance(v, str) else v.get('text', '') for v in corpus.values()]

    # "real" structural units present in the corpus
    rows, items = set(), set()
    for t in texts:
        for line in t.split('\n'):
            ls = line.strip()
            if ' | ' in ls: rows.add(ls)
            if _LIST_ITEM.match(line): items.add(ls)

    # Whitespace-robust matching. A chunker may keep a row's CONTENT whole but
    # reformat its spacing (e.g. adaptive's pipe-row reassembly collapses runs of
    # spaces); a raw substring test would then report a false "split". We compare
    # on whitespace-normalised text on both sides so only REAL fragmentation counts.
    _ws = lambda s: ' '.join(s.split())
    rows_n = [(_ws(r), r) for r in rows]
    items_n = [(_ws(i), i) for i in items]

    print(f'\nUnità nel corpus: {len(rows)} righe-tabella, {len(items)} voci di lista\n')
    hdr = (f'{"metodo":11} {"#chunk":>6} {"parole":>7} {"token":>6}  '
           f'{"righe_intatte":>13} {"liste_intatte":>13}  {"clean_bounds":>12} {"coerenza":>9}')
    print(hdr); print('-'*len(hdr))
    out_methods = {}
    for m, fn in methods(ai).items():
        nch=0; w=[]; tok=[]; both=[]; coh=[]; rseen=set(); iseen=set()
        for t in texts:
            chunks = fn(t)
            nch += len(chunks)
            for c in chunks:
                w.append(len(c.split())); tok.append(ai.e.count_tokens(c))
                both.append(clean_start(c) and clean_end(c))
                x = coherence(c, ai)
                if x is not None: coh.append(x)
                cn = _ws(c)
                for rn, r in rows_n:
                    if rn in cn: rseen.add(r)
                for itn, it in items_n:
                    if itn in cn: iseen.add(it)
        out_methods[m] = {
            'n_chunks': nch,
            'avg_words': round(float(np.mean(w)), 1),
            'avg_tokens': round(float(np.mean(tok)), 1),
            'rows_intact': round(len(rseen)/len(rows), 3) if rows else 0.0,
            'lists_intact': round(len(iseen)/len(items), 3) if items else 0.0,
            'clean_bounds': round(float(np.mean(both)), 3),
            'coherence': round(float(np.mean(coh)), 3),
        }
        d = out_methods[m]
        print(f'{m:11} {nch:>6} {d["avg_words"]:>7.0f} {d["avg_tokens"]:>6.0f}  '
              f'{d["rows_intact"]:>13.3f} {d["lists_intact"]:>13.3f}  '
              f'{d["clean_bounds"]:>12.3f} {d["coherence"]:>9.3f}')

    print('\nNota: righe/liste_intatte = fedeltà strutturale (più alto = meglio, pro-adaptive).')
    print('      clean_bounds/coerenza premiano i chunk grandi (pro-semantic): solo per onestà.')

    md = ['# Tier 1 — Fedeltà strutturale dei chunk',
          f'Corpus: `{args.corpus}` — {len(rows)} righe-tabella, {len(items)} voci di lista.', '',
          '| metodo | #chunk | parole/chunk | token/chunk | righe intatte | liste intatte | clean_bounds | coerenza |',
          '|---|---|---|---|---|---|---|---|']
    for m, d in out_methods.items():
        md.append(f'| {m} | {d["n_chunks"]} | {d["avg_words"]:.0f} | {d["avg_tokens"]:.0f} | '
                  f'{d["rows_intact"]:.3f} | {d["lists_intact"]:.3f} | {d["clean_bounds"]:.3f} | {d["coherence"]:.3f} |')
    md += ['', '_righe/liste intatte = fedeltà strutturale (pro-adaptive). '
           'clean_bounds/coerenza premiano i chunk grandi (pro-semantic): solo per onestà._']
    save_results(args.out, {
        'metric': 'tier1_intrinsic',
        'corpus': args.corpus,
        'n_table_rows': len(rows),
        'n_list_items': len(items),
        'methods': out_methods,
    }, '\n'.join(md) + '\n')


def save_results(out_path, data, md_text):
    """Save the results as JSON and, alongside, a readable .md table."""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    md_path = os.path.splitext(out_path)[0] + '.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_text)
    print(f'\nSalvato:\n  {out_path}\n  {md_path}')


if __name__ == '__main__':
    main()
