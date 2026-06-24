"""
Knowledge-graph retrieval (the query-time half of the hybrid approach).

Flow (matches the diagram: query -> entity -> graph -> neighbouring chunks):
  1. SEED    : entities most relevant to the query (cosine query<->gloss + alias in query).
  2. NAVIGATE: BFS over the entity->entity skeleton up to `hops` neighbours.
  3. COLLECT : gather the chunk_ids anchored to every visited entity.
  4. RANK    : score those chunks by cosine similarity to the query.

Modes:
  soft : rank over (candidate chunks UNION global top-k)  -> robust, never starves.
  hard : rank over candidate chunks only (avio/SPARQL-style hard pre-filter); falls
         back to global vector search only if the graph yields nothing.
  auto : try the graph (hard) and KEEP it only if it "found well" — i.e. the best
         graph chunk is within kg_fallback_tol (cosine) of the best global chunk;
         otherwise use the plain vector search. "se trovo bene, altrimenti il solito metodo."
"""

import logging
from collections import deque
from typing import Dict, List, Set, Tuple

from ai import AI, get_similarity_scores
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
    # always included (graph entry point even when embeddings disagree).
    query_lower = query.lower()
    for e in entities:
        if e['ID'] in seed_ids:
            continue
        if any(a.lower() in query_lower for a in e['aliases'] if len(a) >= 3):
            seed_ids.append(e['ID'])

    return seed_ids


def _navigate(kg: KGData, seed_ids: List[str], hops: int) -> Set[str]:
    """BFS over entity edges (undirected) up to `hops` from the seeds."""
    adjacency: Dict[str, Set[str]] = {eid: set() for eid in kg['entities']}
    for edge in kg['entity_edges']:
        a, b = edge['src_entity_id'], edge['dst_entity_id']
        if a in adjacency and b in adjacency:
            adjacency[a].add(b)
            adjacency[b].add(a)

    visited: Set[str] = set()
    frontier = deque((eid, 0) for eid in seed_ids if eid in adjacency)
    while frontier:
        eid, depth = frontier.popleft()
        if eid in visited:
            continue
        visited.add(eid)
        if depth < hops:
            for nbr in adjacency[eid]:
                if nbr not in visited:
                    frontier.append((nbr, depth + 1))
    return visited


def _candidate_chunk_ids(kg: KGData, entity_ids: Set[str]) -> Set[str]:
    return {e['chunk_id'] for e in kg['chunk_entity_edges'] if e['entity_id'] in entity_ids}


def kg_retrieve_chunks(
    ai: AI,
    query: str,
    index: SDRAGIndex,
    limit: int,
    mode: str,
    hops: int,
    n_seeds: int,
    soft_global_k: int,
    fallback_tol: float = 0.05,
) -> List[Tuple[ChunkNode, float]]:
    """Graph-driven chunk retrieval. Returns chunks_and_scores like retrieve_chunks."""

    kg = index.get('kg')
    all_chunks = list(index['chunks'].values())
    all_embeddings = ai.e.embed([c['text'] for c in all_chunks])
    query_embedding = ai.e.embed(query)

    # Global ranking is always computed: soft-mode union partner / hard/auto fallback.
    global_ranked = get_similarity_scores(query_embedding, all_chunks, all_embeddings, limit=len(all_chunks))

    if not kg:
        log.warning('kg_retrieve_chunks called but index has no KG; using global vector search.')
        return global_ranked[:limit]

    seeds = _seed_entities(ai, query, kg, n_seeds)
    visited = _navigate(kg, seeds, hops)
    candidate_ids = _candidate_chunk_ids(kg, visited)
    log.info(
        f'KG retrieval: seeds={seeds} | visited={sorted(visited)} | '
        f'{len(candidate_ids)} candidate chunks | mode={mode}'
    )

    def rank_pool(pool_ids: Set[str]) -> List[Tuple[ChunkNode, float]]:
        pool = [c for c in all_chunks if c['ID'] in pool_ids]
        if not pool:
            return []
        pool_embeddings = ai.e.embed([c['text'] for c in pool])
        return get_similarity_scores(query_embedding, pool, pool_embeddings, limit=limit)

    if mode == 'soft':
        global_top_ids = {c['ID'] for c, _ in global_ranked[:soft_global_k]}
        return rank_pool(set(candidate_ids) | global_top_ids) or global_ranked[:limit]

    if mode == 'hard':
        if not candidate_ids:
            log.info('KG hard mode: empty candidate set, falling back to global search.')
            return global_ranked[:limit]
        return rank_pool(set(candidate_ids))

    if mode == 'auto':
        # "se trovo bene, altrimenti il solito metodo": trust the graph only if its
        # best chunk is about as relevant as the overall best chunk.
        if not candidate_ids:
            log.info('KG auto: empty candidate set -> plain search.')
            return global_ranked[:limit]
        kg_ranked = rank_pool(set(candidate_ids))
        kg_best = kg_ranked[0][1] if kg_ranked else -1.0
        global_best = global_ranked[0][1] if global_ranked else 0.0
        if kg_best >= global_best - fallback_tol:
            log.info(f'KG auto: graph kept (kg_best={kg_best:.3f} >= global_best={global_best:.3f} - {fallback_tol}).')
            return kg_ranked
        log.info(f'KG auto: graph missed (kg_best={kg_best:.3f} < global_best={global_best:.3f} - {fallback_tol}) -> plain search.')
        return global_ranked[:limit]

    # safety: behave as global
    return global_ranked[:limit]
