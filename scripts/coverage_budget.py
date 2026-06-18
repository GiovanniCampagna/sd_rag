"""
Metrica "coverage@budget" (Livello A, locale) per confrontare i metodi di chunking
sullo step 1 della tesi: "minimo contesto, miglior completezza".

Idea (equa, NON viziata dalla dimensione del chunk): per ogni query si recuperano i
chunk per similarità coseno con la query finché non si riempie un BUDGET di token
fisso; poi si misura quanta parte della gold-answer è coperta da quel contesto.
A parità di budget il confronto è onesto: il metodo migliore copre di più la
risposta con lo stesso numero di token (curva coverage(B) più alta / satura prima).

Due misure di copertura, entrambe locali (solo embedding nomic, niente generazione):
  - cov  : copertura semantica = media, sulle frasi della gold, del max coseno con
           le frasi del contesto (0..1, threshold-free). "Quanto del contenuto della
           risposta è presente nel contesto."
  - val  : value-recall = frazione dei valori distintivi della gold (numeri/codici/
           telefoni/frequenze) presenti nel contesto. Mirata ai dati puntuali.

NB: confronto a budget di TOKEN fisso → neutralizza il bias pro-chunk-piccoli che
aveva affossato la vecchia 'tokeff'. Tokenizzazione con lo stesso modello per tutti
i metodi (coerente). Retrieval per-test-case (come la pipeline non-merge).

Uso:
  env/Scripts/python.exe scripts/coverage_budget.py \
      --testset datasets/testset.json --ai configs/ai.config.nomic.only.json
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

BUDGETS = [100, 200, 400, 800]   # budget di token di contesto
SIM_THR = 0.65                   # soglia per la copertura "hard" (riportata a parte)

# ---- metodi di chunking, stessi default della pipeline ----
def _baseline(text, ai):            return make_chunks(text, size=60, overlap=0)
def _semantic(text, ai):            return make_chunks_semantic(text, ai=ai, max_words=300)
def _structural(text, ai):          return make_chunks_structural(text, max_words=200, min_words=25)
def _struct_sem(text, ai):          return make_chunks_structural_semantic(text, ai=ai, max_words=200, min_words=25)
def _adaptive(text, ai):            return make_chunks_adaptive(text, ai=ai, max_words=200, min_words=25,
                                                                tabular_max_rows=3, tabular_max_words=200,
                                                                inject_section_heading=True)
METHODS = {
    'baseline':    _baseline,
    'semantic':    _semantic,
    'structural':  _structural,
    'struct-sem':  _struct_sem,
    'adaptive':    _adaptive,
}

_VAL = re.compile(r'\+?\d[\d._/().\- ]*\d|\b[A-Z]{2,}\d{1,}\b')
def _norm(v): return re.sub(r'[^0-9A-Za-z]', '', v).upper()
def gold_values(text):
    return {_norm(v) for v in _VAL.findall(text) if len(_norm(v)) >= 4}

def cosine_matrix(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A @ B.T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--testset', required=True)
    ap.add_argument('--ai', default='configs/ai.config.nomic.only.json')
    ap.add_argument('--out', default='results/tier2_coverage.json',
                    help='dove salvare i risultati (JSON + tabella .md affianco)')
    args = ap.parse_args()

    ai = AI(*json.loads(open(args.ai).read()))
    testset = json.load(open(args.testset, encoding='utf-8'))

    # accumulatori: metodo -> budget -> liste per query
    cov = {m: {b: [] for b in BUDGETS} for m in METHODS}
    valr = {m: {b: [] for b in BUDGETS} for m in METHODS}
    n_chunks = {m: [] for m in METHODS}
    chunk_tok = {m: [] for m in METHODS}
    n_val_queries = 0

    for m, fn in METHODS.items():
        for tc in testset:
            text = tc['text']
            chunks = fn(text, ai)
            if not chunks:
                continue
            n_chunks[m].append(len(chunks))
            ctok = [ai.e.count_tokens(c) for c in chunks]
            chunk_tok[m].extend(ctok)
            cemb = ai.e.embed(chunks)

            for qa in tc['qa_pairs']:
                gold = qa['unredacted_answer_text']
                q_emb = ai.e.embed(qa['redacted_question_text'])[0]
                # ranking chunk per similarità con la query
                sims = (cemb / (np.linalg.norm(cemb, axis=1, keepdims=True) + 1e-9)) @ \
                       (q_emb / (np.linalg.norm(q_emb) + 1e-9))
                order = list(np.argsort(-sims))

                gsents = _split_sentences(gold)
                gemb = ai.e.embed(gsents) if gsents else None
                gvals = {v for v in gold_values(gold) if v in _norm(text)}
                if m == list(METHODS)[0]:
                    n_val_queries += 1 if gvals else 0

                for b in BUDGETS:
                    picked, tok = [], 0
                    for idx in order:
                        if tok >= b and picked:
                            break
                        picked.append(chunks[idx]); tok += ctok[idx]
                    context = '\n'.join(picked)
                    # copertura semantica (soft, threshold-free)
                    if gemb is not None:
                        csents = _split_sentences(context)
                        if csents:
                            M = cosine_matrix(gemb, ai.e.embed(csents))
                            cov[m][b].append(float(M.max(axis=1).mean()))
                    # value-recall
                    if gvals:
                        cn = _norm(context)
                        valr[m][b].append(sum(1 for g in gvals if g in cn) / len(gvals))

    # ---- report ----
    print(f'\nQuery totali: {sum(len(tc["qa_pairs"]) for tc in testset)} '
          f'(con valori puntuali: {n_val_queries})')
    print('Budget di token di contesto:', BUDGETS, '\n')

    print('=== coverage semantica della gold-answer (media max-coseno frasi) ===')
    hdr = f'{"metodo":11} {"#chunk":>7} {"tok/chunk":>9}  ' + '  '.join(f'B={b:<4}' for b in BUDGETS)
    print(hdr); print('-' * len(hdr))
    for m in METHODS:
        row = f'{m:11} {np.mean(n_chunks[m]):>7.1f} {np.mean(chunk_tok[m]):>9.1f}  '
        row += '  '.join(f'{np.mean(cov[m][b]):>5.3f}' for b in BUDGETS)
        print(row)

    print('\n=== value-recall (valori distintivi della gold presenti nel contesto) ===')
    print(hdr); print('-' * len(hdr))
    for m in METHODS:
        row = f'{m:11} {np.mean(n_chunks[m]):>7.1f} {np.mean(chunk_tok[m]):>9.1f}  '
        row += '  '.join(f'{np.mean(valr[m][b]) if valr[m][b] else 0:>5.3f}' for b in BUDGETS)
        print(row)

    # ---- salvataggio ----
    out_methods = {}
    for m in METHODS:
        out_methods[m] = {
            'n_chunks': round(float(np.mean(n_chunks[m])), 1),
            'tok_per_chunk': round(float(np.mean(chunk_tok[m])), 1),
            'coverage': {str(b): round(float(np.mean(cov[m][b])), 3) for b in BUDGETS},
            'value_recall': {str(b): round(float(np.mean(valr[m][b])) if valr[m][b] else 0.0, 3) for b in BUDGETS},
        }
    def _tbl(title, key):
        lines = [f'**{title}**', '',
                 '| metodo | #chunk | tok/chunk | ' + ' | '.join(f'B={b}' for b in BUDGETS) + ' |',
                 '|' + '---|'*(3+len(BUDGETS))]
        for m, d in out_methods.items():
            lines.append(f"| {m} | {d['n_chunks']:.1f} | {d['tok_per_chunk']:.0f} | "
                         + ' | '.join(f"{d[key][str(b)]:.3f}" for b in BUDGETS) + ' |')
        return '\n'.join(lines)
    md = (f'# Tier 2 — Copertura @ budget di token\n'
          f'Testset: `{args.testset}` — {sum(len(tc["qa_pairs"]) for tc in testset)} query '
          f'(con valori puntuali: {n_val_queries}). Budget: {BUDGETS} token.\n\n'
          + _tbl('Copertura semantica della gold-answer', 'coverage') + '\n\n'
          + _tbl('Value-recall (valori puntuali)', 'value_recall') + '\n')
    save_results(args.out, {
        'metric': 'tier2_coverage_budget',
        'testset': args.testset,
        'n_queries': sum(len(tc['qa_pairs']) for tc in testset),
        'n_value_queries': n_val_queries,
        'budgets': BUDGETS,
        'methods': out_methods,
    }, md)


def save_results(out_path, data, md_text):
    """Salva i risultati come JSON e, affianco, una tabella .md leggibile."""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    md_path = os.path.splitext(out_path)[0] + '.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_text)
    print(f'\nSalvato:\n  {out_path}\n  {md_path}')


if __name__ == '__main__':
    main()
