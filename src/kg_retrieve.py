"""
Knowledge-graph retrieval (the query-time half of the hybrid approach).

Flow (matches the reference diagram: query -> entity -> graph -> nearby chunks):
  1. SEED    : entities most relevant to the query (cosine query<->gloss + alias in query).
  2. NAVIGATE: BFS over the entity->entity skeleton up to `hops` neighbours.
  3. COLLECT : gather the chunk_ids anchored to every visited entity (the candidate pool).
  4. RANK    : score the pool by cosine similarity to the query, exactly like the plain
               retrieval — the graph only decides WHICH chunks compete, not how they score.

Modes:
  off  : pure vector search over all chunks (handled by the caller, not here).
  auto : restrict to the graph pool, rank it by cosine; fall back to plain search over
         all chunks when the query anchors no entity (empty pool) —
         "se trovo bene, altrimenti il solito metodo".

NOTE: a graph-bonus re-ranker (final = cosine + alpha * IDF-weighted entity overlap)
was implemented and measured (git 0661a6b), then deliberately reverted: the chosen
design keeps the scoring identical to the plain pipeline downstream of the pool.
"""

import logging
from collections import deque
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
    hops: int,
    n_seeds: int,
) -> List[Tuple[ChunkNode, float]]:
    """Graph-driven chunk retrieval. Returns chunks_and_scores like retrieve_chunks."""

    kg = index.get('kg')
    all_chunks = list(index['chunks'].values())
    all_embeddings = ai.e.embed([c['text'] for c in all_chunks])
    query_embedding = ai.e.embed(query)

    if not kg:
        log.warning('kg_retrieve_chunks called but index has no KG; using global vector search.')
        return get_similarity_scores(query_embedding, all_chunks, all_embeddings, limit=limit)

    seeds = _seed_entities(ai, query, kg, n_seeds)
    visited = _navigate(kg, seeds, hops)
    candidate_ids = _candidate_chunk_ids(kg, visited)
    log.info(
        f'KG retrieval: seeds={seeds} | visited={sorted(visited)} | '
        f'{len(candidate_ids)} candidate chunks'
    )

    if not candidate_ids:
        # "se non c'e' corrispondenza passa a off": plain search over all chunks
        log.info('KG auto: empty candidate set -> plain search (off).')
        return get_similarity_scores(query_embedding, all_chunks, all_embeddings, limit=limit)

    # The graph only restricted the pool; scoring is plain cosine, as in retrieve_chunks.
    pool = [c for c in all_chunks if c['ID'] in candidate_ids]
    pool_embeddings = ai.e.embed([c['text'] for c in pool])
    return get_similarity_scores(query_embedding, pool, pool_embeddings, limit=limit)
