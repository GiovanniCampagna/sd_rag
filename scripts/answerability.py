"""
Tier 2 — Answerability a rango (top-k). Sostituisce il vecchio coverage@budget.

Perche' non il budget di token: riempire un budget con PIU' chunk lascia che i metodi
deboli compensino (tanti chunk piccoli "coprono" quanto un buon chunk solo), e la
copertura semantica soft satura (nomic da' coseno ~0.6-0.8 anche fuori bersaglio). Non
misura quindi la QUALITA' del chunk, ma la recall su molti chunk.

Qui si guarda chi mette l'informazione COMPLETA in un singolo chunk ben rankato, con
metriche parametro-free (niente soglia, niente budget):

  - valori@k : % di domande (con valori puntuali) i cui valori distintivi della gold sono
               TUTTI nei primi k chunk per coseno con la query. k=1 = "becca subito".
  - MRR      : media di 1/(rango del primo chunk che completa i valori).
  - cov@1    : copertura semantica della gold col SOLO top-1 chunk (max-coseno medio).
  - tok_top1, tok/chunk : costo in token (per leggere il risultato come fedelta'/token).

Uso:
  env/Scripts/python.exe scripts/answerability.py --testset datasets/testset.json \
      --out results/tier2_isp.json
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

KS = [1, 2, 3]   # rank cut-offs

# ---- chunking methods, stessi default della pipeline ----
def _baseline(text, ai):    return make_chunks(text, size=60, overlap=0)
def _semantic(text, ai):    return make_chunks_semantic(text, ai=ai, max_words=300)
def _structural(text, ai):  return make_chunks_structural(text, max_words=200, min_words=25)
def _struct_sem(text, ai):  return make_chunks_structural_semantic(text, ai=ai, max_words=200, min_words=25)
def _adaptive(text, ai):    return make_chunks_adaptive(text, ai=ai, max_words=200, min_words=25,
                                                        tabular_max_rows=3, tabular_max_words=200,
                                                        inject_section_heading=True)
METHODS = {'baseline': _baseline, 'semantic': _semantic, 'structural': _structural,
           'struct-sem': _struct_sem, 'adaptive': _adaptive}

_VAL = re.compile(r'\+?\d[\d._/().\- ]*\d|\b[A-Z]{2,}\d{1,}\b')
def _norm(v): return re.sub(r'[^0-9A-Za-z]', '', v).upper()
def gold_values(text):
    return {_norm(v) for v in _VAL.findall(text) if len(_norm(v)) >= 4}

def cosine_matrix(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A @ B.T

mean = lambda L: (sum(L) / len(L)) if L else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--testset', required=True)
    ap.add_argument('--ai', default='configs/ai.config.nomic.only.json')
    ap.add_argument('--out', default='results/tier2_answerability.json',
                    help='dove salvare i risultati (JSON + tabella .md a fianco)')
    args = ap.parse_args()
    ai = AI(*json.loads(open(args.ai).read()))
    testset = json.load(open(args.testset, encoding='utf-8'))

    vhit = {m: {k: [] for k in KS} for m in METHODS}
    mrr = {m: [] for m in METHODS}
    cov1 = {m: [] for m in METHODS}
    tok1 = {m: [] for m in METHODS}
    chunktok = {m: [] for m in METHODS}
    nchunks = {m: [] for m in METHODS}
    nvq = 0

    for m, fn in METHODS.items():
        for tc in testset:
            text = tc['text']
            chunks = fn(text, ai)
            if not chunks:
                continue
            nchunks[m].append(len(chunks))
            ctok = [ai.e.count_tokens(c) for c in chunks]
            chunktok[m].extend(ctok)
            cnorm = [_norm(c) for c in chunks]
            cemb = ai.e.embed(chunks)

            for qa in tc['qa_pairs']:
                gold = qa['unredacted_answer_text']
                q_emb = ai.e.embed(qa['redacted_question_text'])[0]
                sims = (cemb / (np.linalg.norm(cemb, axis=1, keepdims=True) + 1e-9)) @ \
                       (q_emb / (np.linalg.norm(q_emb) + 1e-9))
                order = list(np.argsort(-sims))
                tok1[m].append(ctok[order[0]])

                # answerability sui valori puntuali
                gvals = {v for v in gold_values(gold) if v in _norm(text)}
                if gvals:
                    if m == list(METHODS)[0]:
                        nvq += 1
                    seen, rank = set(), 0
                    for k_i, idx in enumerate(order, 1):
                        seen |= {v for v in gvals if v in cnorm[idx]}
                        if seen == gvals:
                            rank = k_i
                            break
                    mrr[m].append(1.0 / rank if rank else 0.0)
                    for k in KS:
                        vhit[m][k].append(1.0 if (rank and rank <= k) else 0.0)

                # copertura del solo top-1 (tutte le query)
                gs = _split_sentences(gold)
                if gs:
                    gemb = ai.e.embed(gs)
                    cs = _split_sentences(chunks[order[0]])
                    if cs:
                        cov1[m].append(float(cosine_matrix(gemb, ai.e.embed(cs)).max(axis=1).mean()))

    # ---- report ----
    nq = sum(len(tc['qa_pairs']) for tc in testset)
    print(f'\nTestset: {args.testset} — {nq} query (con valori puntuali: {nvq})\n')
    hdr = (f'{"metodo":11} {"n.chunk":>7} {"tok/chunk":>9} {"tok top-1":>9}  '
           f'{"valori@1":>8} {"valori@2":>8} {"valori@3":>8} {"MRR":>5}  {"cov@1":>6}')
    print(hdr); print('-' * len(hdr))
    for m in METHODS:
        print(f'{m:11} {mean(nchunks[m]):>7.1f} {mean(chunktok[m]):>9.1f} {mean(tok1[m]):>9.1f}  '
              f'{mean(vhit[m][1]):>8.3f} {mean(vhit[m][2]):>8.3f} {mean(vhit[m][3]):>8.3f} '
              f'{mean(mrr[m]):>5.3f}  {mean(cov1[m]):>6.3f}')
    print('\nvalori@k = % domande con TUTTI i valori della risposta nei primi k chunk (k=1 = subito).')
    print('MRR = media di 1/rango del 1° chunk completo. cov@1 = copertura della gold col solo top-1.')

    # ---- save ----
    out = {}
    for m in METHODS:
        out[m] = {
            'n_chunks': round(mean(nchunks[m]), 1),
            'tok_per_chunk': round(mean(chunktok[m]), 1),
            'tok_top1': round(mean(tok1[m]), 1),
            'valori_at': {str(k): round(mean(vhit[m][k]), 3) for k in KS},
            'mrr': round(mean(mrr[m]), 3),
            'cov_top1': round(mean(cov1[m]), 3),
        }
    lines = ['# Tier 2 — Answerability a rango (top-k)',
             f'Testset: `{args.testset}` — {nq} query (con valori puntuali: {nvq}).', '',
             '| metodo | n.chunk | tok/chunk | tok top-1 | valori@1 | valori@2 | valori@3 | MRR | cov@1 |',
             '|---|---|---|---|---|---|---|---|---|']
    for m, dct in out.items():
        lines.append(f"| {m} | {dct['n_chunks']:.1f} | {dct['tok_per_chunk']:.0f} | {dct['tok_top1']:.0f} | "
                     f"{dct['valori_at']['1']:.3f} | {dct['valori_at']['2']:.3f} | {dct['valori_at']['3']:.3f} | "
                     f"{dct['mrr']:.3f} | {dct['cov_top1']:.3f} |")
    lines += ['', '_valori@k = % domande con tutti i valori della risposta nei primi k chunk recuperati. '
              'MRR = media di 1/rango del 1° chunk completo. cov@1 = copertura della gold col solo top-1._']
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'metric': 'tier2_answerability', 'testset': args.testset,
                   'n_queries': nq, 'n_value_queries': nvq, 'methods': out}, f, ensure_ascii=False, indent=2)
    with open(os.path.splitext(args.out)[0] + '.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nSalvato: {args.out}')


if __name__ == '__main__':
    main()
