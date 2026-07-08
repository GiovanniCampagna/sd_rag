"""
Knowledge-graph retrieval (the query-time half of the hybrid approach).

Flow (matches the reference diagram: query -> entity -> graph -> the topic's chunks):
  1. SEED : the QUERY'S entities (alias literally in the query + top-n by cosine
            query<->gloss). Multi-fact queries legitimately have several entities.
  2. POOL : chunks anchored to the query's entities ONLY. Navigated neighbours do NOT
            contribute chunks — pooling every visited entity's chunks was measured to
            blow the pool up towards the whole corpus (55/58).
  3. NAVIGATE (fallback only): if the query's entities anchor no chunk, walk `hops`
            steps to find the topic (e.g. a query naming only the aircraft can reach
            refuelling_event) and pool from the reached entities.
  4. RANK : cosine similarity to the query over the pool, exactly like plain retrieval.

Modes:
  off  : pure vector search over all chunks (handled by the caller, not here).
  auto : the flow above; falls back to plain search over all chunks when no entity
         (even after navigation) anchors any chunk —
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


def _gloss(e) -> str:
    return f"{e['label']}. {e['gloss']} ({', '.join(e['aliases'][:6])})"


def _seed_entities(ai: AI, query: str, kg: KGData, n_seeds: int) -> List[str]:
    """Return ids of the n_seeds entities most relevant to the query."""
    entities = list(kg['entities'].values())
    if not entities:
        return []

    gloss_texts = [_gloss(e) for e in entities]
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

    # POOL: chunks of the QUERY'S entities only (all of them: a multi-fact query
    # legitimately spans several entities). Navigated neighbours do not contribute.
    candidate_ids = _candidate_chunk_ids(kg, set(seeds))
    used = 'seeds'

    if not candidate_ids and seeds:
        # Topic-finding fallback: the query's entities anchor nothing, walk the graph
        # to reach the topic (e.g. aircraft -> refuelling_event).
        visited = _navigate(kg, seeds, hops)
        candidate_ids = _candidate_chunk_ids(kg, visited)
        used = f'navigated({sorted(visited)})'

    log.info(f'KG retrieval: seeds={seeds} | pool from {used} | {len(candidate_ids)} candidate chunks')

    if not candidate_ids:
        # "se non c'e' corrispondenza passa a off": plain search over all chunks
        log.info('KG auto: no anchored entity -> plain search (off).')
        return get_similarity_scores(query_embedding, all_chunks, all_embeddings, limit=limit)

    # The graph only restricted the pool; scoring is plain cosine, as in retrieve_chunks.
    pool = [c for c in all_chunks if c['ID'] in candidate_ids]
    pool_embeddings = ai.e.embed([c['text'] for c in pool])
    return get_similarity_scores(query_embedding, pool, pool_embeddings, limit=limit)
