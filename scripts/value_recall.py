"""
Tier 2 — Recall dei valori (top-k). Sostituisce il vecchio coverage@budget.

Perche' non il budget di token: riempire un budget con PIU' chunk lascia che i metodi
deboli compensino (tanti chunk piccoli "coprono" quanto un buon chunk solo), e la
copertura semantica "soft" (max-coseno medio) satura — nomic da' coseno ~0.6-0.8 anche
fuori bersaglio. Non misura la QUALITA' del chunk, ma la recall su molti chunk. (Per lo
stesso motivo NON si riporta una cov@1: sarebbe la stessa misura satura, ristretta al
top-1, e premia i frammenti piccoli e super-aderenti alla query.)

Metriche, parametro-free (niente soglia, niente budget), sui valori puntuali:
  - valori@k : % di domande i cui valori distintivi della gold sono TUTTI nei primi k
               chunk per coseno con la query. k=1 = "becca subito".
Accanto, la dimensione/costo: parole/chunk, token/chunk, n. chunk.

Uso:
  env/Scripts/python.exe scripts/value_recall.py --testset datasets/testset.json \
      --out results/tier2_isp.json
"""
import sys, os, json, re, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np

from ai import AI
from make_chunks import make_chunks
from make_chunks_semantic import make_chunks_semantic
from make_chunks_structural import make_chunks_structural
from make_chunks_structural_semantic import make_chunks_structural_semantic
from make_chunks_adaptive import make_chunks_adaptive
from build_kg import build_kg
from kg_ontology import get_ontology
from kg_retrieve import _seed_entities, _navigate, _candidate_chunk_ids, _gloss
from ai import get_similarity_scores

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

mean = lambda L: (sum(L) / len(L)) if L else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--testset', required=True)
    ap.add_argument('--ai', default='configs/ai.config.nomic.only.json')
    ap.add_argument('--out', default=f'tier2.{time.time()}.tmp.json',
                    help='dove salvare i risultati (JSON + tabella .md a fianco); default: nella '
                         'cartella corrente col timestamp, come run/eval/interrogate')
    ap.add_argument('--merge', action=argparse.BooleanOptionalAction, default=False,
                    help='indice UNICO su tutte le sezioni (retrieval realistico, coi distrattori '
                         'cross-sezione) invece che per-sezione; ogni query ranka contro il pool intero')
    ap.add_argument('--kg', action=argparse.BooleanOptionalAction, default=False,
                    help='aggiunge la variante "adaptive+kg": stessi chunk adaptive, ma il ranking '
                         'passa dal pool del grafo (seed -> naviga -> chunk delle entita\'), con '
                         'fallback al ranking globale se il pool e\' vuoto — come kg_mode=auto')
    args = ap.parse_args()
    ai = AI(*json.loads(open(args.ai).read()))
    testset = json.load(open(args.testset, encoding='utf-8'))

    vhit = {m: {k: [] for k in KS} for m in METHODS}
    chunkw = {m: [] for m in METHODS}
    chunktok = {m: [] for m in METHODS}
    nchunks = {m: [] for m in METHODS}
    nvq = 0

    for m, fn in METHODS.items():
        # A "unit" = (chunks, queries). No-merge: one unit per section (la query ranka
        # solo tra i chunk della sua sezione). Merge: un unico pool di tutti i chunk del
        # corpus e TUTTE le query rankano contro quello (distrattori cross-sezione).
        if args.merge:
            pool = [c for tc in testset for c in fn(tc['text'], ai)]
            units = [(pool, [(qa, tc['text']) for tc in testset for qa in tc['qa_pairs']])]
        else:
            units = [(fn(tc['text'], ai), [(qa, tc['text']) for qa in tc['qa_pairs']]) for tc in testset]

        for chunks, queries in units:
            if not chunks:
                continue
            nchunks[m].append(len(chunks))
            chunkw[m].extend(len(c.split()) for c in chunks)
            chunktok[m].extend(ai.e.count_tokens(c) for c in chunks)
            cnorm = [_norm(c) for c in chunks]
            cemb = ai.e.embed(chunks)

            for qa, text in queries:
                gold = qa['unredacted_answer_text']
                gvals = {v for v in gold_values(gold) if v in _norm(text)}
                if not gvals:
                    continue
                if m == list(METHODS)[0]:
                    nvq += 1
                q_emb = ai.e.embed(qa['redacted_question_text'])[0]
                sims = (cemb / (np.linalg.norm(cemb, axis=1, keepdims=True) + 1e-9)) @ \
                       (q_emb / (np.linalg.norm(q_emb) + 1e-9))
                order = list(np.argsort(-sims))

                seen, rank = set(), 0
                for k_i, idx in enumerate(order, 1):
                    seen |= {v for v in gvals if v in cnorm[idx]}
                    if seen == gvals:
                        rank = k_i
                        break
                for k in KS:
                    vhit[m][k].append(1.0 if (rank and rank <= k) else 0.0)

    # ---- variante KG: stessi chunk adaptive, ranking attraverso il pool del grafo ----
    if args.kg:
        m = 'adaptive+kg'
        vhit[m] = {k: [] for k in KS}
        chunkw[m], chunktok[m], nchunks[m] = [], [], []
        fn = METHODS['adaptive']
        if args.merge:
            pool = [c for tc in testset for c in fn(tc['text'], ai)]
            units = [(pool, [(qa, tc['text']) for tc in testset for qa in tc['qa_pairs']])]
        else:
            units = [(fn(tc['text'], ai), [(qa, tc['text']) for qa in tc['qa_pairs']]) for tc in testset]

        for chunks, queries in units:
            if not chunks:
                continue
            nchunks[m].append(len(chunks))
            chunkw[m].extend(len(c.split()) for c in chunks)
            chunktok[m].extend(ai.e.count_tokens(c) for c in chunks)
            cnorm = [_norm(c) for c in chunks]
            cemb = ai.e.embed(chunks)
            nodes = [{'ID': str(i), 'text': c, 'part_num': i, 'doc_id': 'doc'}
                     for i, c in enumerate(chunks)]
            kg = build_kg(ai, nodes, cemb, get_ontology())

            for qa, text in queries:
                gold = qa['unredacted_answer_text']
                gvals = {v for v in gold_values(gold) if v in _norm(text)}
                if not gvals:
                    continue
                q_emb = ai.e.embed(qa['redacted_question_text'])[0]
                sims = (cemb / (np.linalg.norm(cemb, axis=1, keepdims=True) + 1e-9)) @ \
                       (q_emb / (np.linalg.norm(q_emb) + 1e-9))
                order = list(np.argsort(-sims))

                # come kg_mode=auto: seed -> naviga -> TARGET (entita' piu' pertinente
                # alla query tra le raggiunte) -> pool = SOLO i chunk del target
                seeds = _seed_entities(ai, qa['redacted_question_text'], kg, 3)
                visited = _navigate(kg, seeds, 1)
                cand = set()
                if visited:
                    reached = [kg['entities'][e] for e in visited]
                    ge = ai.e.embed([_gloss(e) for e in reached])
                    tgt = get_similarity_scores(q_emb, reached, ge, limit=1)[0][0]['ID']
                    cand = _candidate_chunk_ids(kg, {tgt})
                order_used = [i for i in order if str(i) in cand] if cand else order

                seen, rank = set(), 0
                for k_i, idx in enumerate(order_used, 1):
                    seen |= {v for v in gvals if v in cnorm[idx]}
                    if seen == gvals:
                        rank = k_i
                        break
                for k in KS:
                    vhit[m][k].append(1.0 if (rank and rank <= k) else 0.0)

    # ---- report ----
    nq = sum(len(tc['qa_pairs']) for tc in testset)
    print(f'\nTestset: {args.testset} [{"MERGE: indice unico" if args.merge else "no-merge: per-sezione"}] '
          f'— {nq} query (con valori puntuali: {nvq})\n')
    hdr = (f'{"metodo":11} {"n.chunk":>7} {"parole/chunk":>12} {"token/chunk":>11}  '
           f'{"val@1":>6} {"val@2":>6} {"val@3":>6}')
    print(hdr); print('-' * len(hdr))
    for m in vhit:
        print(f'{m:11} {mean(nchunks[m]):>7.1f} {mean(chunkw[m]):>12.0f} {mean(chunktok[m]):>11.0f}  '
              f'{mean(vhit[m][1]):>6.3f} {mean(vhit[m][2]):>6.3f} {mean(vhit[m][3]):>6.3f}')
    print('\nval@k = % domande con TUTTI i valori della risposta nei primi k chunk recuperati (k=1 = subito).')
    print('parole/token-per-chunk = dimensione media del chunk (costo del contesto).')

    # ---- save ----
    out = {}
    for m in vhit:
        out[m] = {
            'n_chunks': round(mean(nchunks[m]), 1),
            'words_per_chunk': round(mean(chunkw[m]), 1),
            'tok_per_chunk': round(mean(chunktok[m]), 1),
            'valori_at': {str(k): round(mean(vhit[m][k]), 3) for k in KS},
        }
    lines = ['# Tier 2 — Recall dei valori (top-k)',
             f'Testset: `{args.testset}` — {nq} query (con valori puntuali: {nvq}).', '',
             '| metodo | n° chunk | parole/chunk | token/chunk | valori nel 1° | valori nei primi 2 | valori nei primi 3 |',
             '|---|---|---|---|---|---|---|']
    for m, dct in out.items():
        lines.append(f"| {m} | {dct['n_chunks']:.1f} | {dct['words_per_chunk']:.0f} | {dct['tok_per_chunk']:.0f} | "
                     f"{dct['valori_at']['1']:.3f} | {dct['valori_at']['2']:.3f} | {dct['valori_at']['3']:.3f} |")
    lines += ['', '_valori nel 1°/nei primi k = % domande con tutti i valori distintivi della risposta nei primi '
              'k chunk recuperati per coseno con la query. parole/token-per-chunk = dimensione media del chunk '
              '(costo del contesto)._']
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'metric': 'tier2_value_recall', 'testset': args.testset, 'merge': args.merge,
                   'n_queries': nq, 'n_value_queries': nvq, 'methods': out}, f, ensure_ascii=False, indent=2)
    with open(os.path.splitext(args.out)[0] + '.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nSalvato: {args.out}')


if __name__ == '__main__':
    main()
