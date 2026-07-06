"""
Knowledge-graph retrieval (the query-time half of the hybrid approach).

Flow (matches the diagram: query -> entity -> graph -> neighbouring chunks):
  1. SEED    : entities most relevant to the query (cosine query<->gloss + alias in query).
  2. NAVIGATE: BFS over the entity->entity skeleton up to `hops` neighbours.
  3. COLLECT : gather the chunk_ids anchored to every visited entity.
  4. RANK    : score those chunks by cosine similarity to the query.

Modes:
  off  : pure vector search over all chunks (handled by the caller, not here).
  auto : restrict to the chunks anchored to the navigated entities, then RE-RANK that pool by
             final = cosine(query, chunk) + alpha * (graph_bonus / max_bonus)
         where graph_bonus rewards a chunk for sharing the query's *specific* (high-IDF)
         entities, decayed by graph distance. This lets a low-cosine but strongly
         entity-linked chunk (e.g. the VVF contacts) rise into the top-k. Falls back to plain
         search when the graph adds nothing (empty pool, or no specific entity overlap) —
         "se trovo bene, altrimenti il solito metodo".
"""

import logging
import math
from collections import deque, defaultdict
from typing import Dict, List, Set, Tuple

from ai import AI, get_similarity_scores
from build_kg import _alias_matches, ALIAS_MIN_CHARS
from schemas import ChunkNode, KGData, SDRAGIndex

log = logging.getLogger(__name__)


def _seed_entities(ai: AI, query: str, kg: KGData, n_seeds: int) -> List[str]:
    """Return ids of the n_seeds entities most relevant to the query."""
    entities = list(kg['entities'].values())
    if not entities:
        return []

    gloss_texts = [f"{e['label']}. {e['gloss']} ({', '.join(e['aliases'][:6])})" for e in entities]
    gloss_embeddings = ai.e.embed(gloss_texts)
    query_embedding = ai.e.embed(query)
    ranked = get_similarity_scores(query_embedding, entities, gloss_embeddings, limit=len(entities))
    seed_ids: List[str] = [e['ID'] for e, _ in ranked[:n_seeds]]

    # Symbolic seeding: any entity whose alias appears literally in the query is
    # always included (graph entry point even when embeddings disagree). Uses the
    # same word-boundary matcher as the index-time anchoring: a plain substring
    # test would fire e.g. on the alias 'cos' inside "cosa"/"così".
    query_lower = query.lower()
    for e in entities:
        if e['ID'] in seed_ids:
            continue
        aliases = [a for a in e['aliases'] if len(a.replace(' ', '')) >= ALIAS_MIN_CHARS]
        if any(_alias_matches(a, query_lower) for a in aliases):
            seed_ids.append(e['ID'])

    return seed_ids


def _navigate(kg: KGData, seed_ids: List[str], hops: int) -> Dict[str, int]:
    """BFS over entity edges (undirected) up to `hops` from the seeds.
    Returns {entity_id: hop distance from the nearest seed} (seeds = distance 0)."""
    adjacency: Dict[str, Set[str]] = {eid: set() for eid in kg['entities']}
    for edge in kg['entity_edges']:
        a, b = edge['src_entity_id'], edge['dst_entity_id']
        if a in adjacency and b in adjacency:
            adjacency[a].add(b)
            adjacency[b].add(a)

    dist: Dict[str, int] = {}
    frontier = deque((eid, 0) for eid in seed_ids if eid in adjacency)
    while frontier:
        eid, depth = frontier.popleft()
        if eid in dist:
            continue
        dist[eid] = depth
        if depth < hops:
            for nbr in adjacency[eid]:
                if nbr not in dist:
                    frontier.append((nbr, depth + 1))
    return dist


def _candidate_chunk_ids(kg: KGData, entity_ids: Set[str]) -> Set[str]:
    return {e['chunk_id'] for e in kg['chunk_entity_edges'] if e['entity_id'] in entity_ids}


def kg_retrieve_chunks(
    ai: AI,
    query: str,
    index: SDRAGIndex,
    limit: int,
    hops: int,
    n_seeds: int,
    alpha: float = 0.10,
) -> List[Tuple[ChunkNode, float]]:
    """Graph-driven chunk retrieval. Returns chunks_and_scores like retrieve_chunks."""

    kg = index.get('kg')
    all_chunks = list(index['chunks'].values())
    all_embeddings = ai.e.embed([c['text'] for c in all_chunks])
    query_embedding = ai.e.embed(query)

    # Global cosine ranking is always computed: it is the 'auto' fallback and the
    # base score the re-ranker adds the graph bonus on top of.
    global_ranked = get_similarity_scores(query_embedding, all_chunks, all_embeddings, limit=len(all_chunks))

    if not kg:
        log.warning('kg_retrieve_chunks called but index has no KG; using global vector search.')
        return global_ranked[:limit]

    seeds = _seed_entities(ai, query, kg, n_seeds)
    visited = _navigate(kg, seeds, hops)          # entity_id -> hop distance from nearest seed
    candidate_ids = _candidate_chunk_ids(kg, set(visited))
    log.info(
        f'KG retrieval: seeds={seeds} | visited={sorted(visited)} | '
        f'{len(candidate_ids)} candidate chunks'
    )

    if not candidate_ids:
        log.info('KG auto: empty candidate set -> plain search (off).')
        return global_ranked[:limit]

    # --- RE-RANKER:  final = cosine + alpha * (graph_bonus / max_bonus) -------
    cos_of = {c['ID']: s for c, s in global_ranked}

    # subjects (alias-anchored entities) per chunk + entity specificity (IDF)
    subjects_of: Dict[str, Set[str]] = defaultdict(set)
    entity_chunks: Dict[str, Set[str]] = defaultdict(set)
    for e in kg['chunk_entity_edges']:
        if e['method'] == 'alias':
            subjects_of[e['chunk_id']].add(e['entity_id'])
            entity_chunks[e['entity_id']].add(e['chunk_id'])
    n_total = len(all_chunks)
    idf = {ent: math.log(n_total / len(chs)) for ent, chs in entity_chunks.items() if chs}

    def graph_bonus(chunk_id: str) -> float:
        # reward chunks sharing the query's *specific* (high-IDF) entities; decay by graph distance
        return sum(
            idf.get(ent, 0.0) * (0.5 ** visited[ent])
            for ent in subjects_of.get(chunk_id, ())
            if ent in visited
        )

    pool = [c for c in all_chunks if c['ID'] in candidate_ids]
    bonuses = {c['ID']: graph_bonus(c['ID']) for c in pool}
    max_bonus = max(bonuses.values(), default=0.0)

    if max_bonus <= 0.0:
        # the query maps only to ubiquitous entities (or none) -> graph adds nothing
        log.info('KG auto: no specific entity overlap -> plain search (off).')
        return global_ranked[:limit]

    scored = sorted(
        ((c, cos_of.get(c['ID'], 0.0) + alpha * (bonuses[c['ID']] / max_bonus)) for c in pool),
        key=lambda x: x[1], reverse=True,
    )
    top = scored[:limit]
    log.info(
        'KG auto rerank top: ' + ' | '.join(
            f"{c['ID'][:6]} cos={cos_of.get(c['ID'], 0.0):.3f} bonus={bonuses[c['ID']]:.2f} -> {s:.3f}"
            for c, s in top
        )
    )
    return top
