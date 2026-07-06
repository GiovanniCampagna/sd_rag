"""
Build the in-memory knowledge graph (HYBRID approach) and attach it to the index.

Produces (schemas.KGData):
  1. entities            : one EntityNode per AISS ontology concept.
  2. entity_edges        : entity->entity edges from the ontology relation skeleton.
  3. chunk_entity_edges  : the "anchoring" — which chunk mentions which entity.

It also writes chunk['subjects'] = [entity ids matched literally in the chunk].
This is the professor's [Soggetto: ...] tag, materialised as structured metadata
(kept out of the generation text so it does not pollute answers).

Anchoring uses two signals, mirroring how build_index.py binds constraints:
  - 'alias' : the entity surface form occurs literally in the chunk (symbolic, score 1.0).
  - 'embed' : top-k chunks by cosine(entity gloss, chunk) above embed_min_score (neural).
"""

import re
import logging
from typing import Dict, List

import numpy as np

from ai import AI, get_similarity_scores
from kg_ontology import Ontology
from schemas import ChunkNode, EntityNode, EntityRelationEdge, ChunkEntityEdge, KGData

log = logging.getLogger(__name__)

# Aliases shorter than this (spaces excluded) are skipped by the symbolic matcher.
# Shared by index-time anchoring (here) and query-time seeding (kg_retrieve).
ALIAS_MIN_CHARS = 3


def _alias_matches(alias: str, text_lower: str) -> bool:
    """Substring match for multi-word aliases; word-boundary match for single
    short/numeric tokens (so '304' does not match inside '13045')."""
    alias = alias.strip().lower()
    if not alias:
        return False
    if ' ' in alias or '/' in alias or '-' in alias:
        return alias in text_lower
    return re.search(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', text_lower) is not None


def build_kg(
    ai: AI,
    chunks: List[ChunkNode],
    chunk_embeddings: np.ndarray,
    ontology: Ontology,
    bind_limit: int = 8,
    alias_min_chars: int = ALIAS_MIN_CHARS,
    embed_min_score: float = 0.5,
    prune_orphan_entities: bool = True,
) -> KGData:

    log.info('Building knowledge graph (hybrid, AISS-derived).')

    # 1. ENTITIES ----------------------------------------------------------
    entities: Dict[str, EntityNode] = {}
    for c in ontology['concepts']:
        entities[c['id']] = {
            'ID': c['id'],
            'label': c['label'],
            'type': c['type'],
            'aliases': c['aliases'],
            'gloss': c['gloss'],
        }

    chunk_entity_edges: List[ChunkEntityEdge] = []

    # 2a. alias (symbolic) match  +  materialise chunk['subjects']
    chunk_texts_lower = [c['text'].lower() for c in chunks]
    chunk_subjects: Dict[str, List[str]] = {c['ID']: [] for c in chunks}
    for entity in entities.values():
        aliases = [a for a in entity['aliases'] if len(a.replace(' ', '')) >= alias_min_chars]
        for chunk, text_lower in zip(chunks, chunk_texts_lower):
            if any(_alias_matches(a, text_lower) for a in aliases):
                chunk_entity_edges.append({
                    'chunk_id': chunk['ID'],
                    'entity_id': entity['ID'],
                    'score': 1.0,
                    'method': 'alias',
                })
                chunk_subjects[chunk['ID']].append(entity['ID'])

    # write the [Soggetto: ...] tag as structured metadata on each chunk
    for chunk in chunks:
        chunk['subjects'] = sorted(chunk_subjects[chunk['ID']])

    # 2b. embedding (neural) match — top bind_limit chunks per entity gloss, but only
    # above embed_min_score: without a threshold every entity always anchors its
    # bind_limit "least bad" chunks, even entities foreign to the corpus, and the
    # candidate pool degenerates towards the whole document.
    for entity in entities.values():
        gloss_text = f"{entity['label']}. {entity['gloss']} ({', '.join(entity['aliases'][:6])})"
        gloss_embedding = ai.e.embed(gloss_text)
        chunks_and_scores = get_similarity_scores(gloss_embedding, chunks, chunk_embeddings, limit=bind_limit)
        for chunk, score in chunks_and_scores:
            if score < embed_min_score:
                continue
            chunk_entity_edges.append({
                'chunk_id': chunk['ID'],
                'entity_id': entity['ID'],
                'score': float(score),
                'method': 'embed',
            })

    # 3. ENTITY -> ENTITY EDGES (ontology relation skeleton) ---------------
    entity_edges: List[EntityRelationEdge] = []
    for src, rel, dst in ontology['relations']:
        if src in entities and dst in entities:
            entity_edges.append({'src_entity_id': src, 'dst_entity_id': dst, 'relation': rel})
        else:
            # Every concept still exists at this point (pruning runs later), so an
            # unknown id here can only be a typo in the ontology's _RELATIONS.
            log.warning(f'Dropped ontology relation ({src}, {rel}, {dst}): unknown concept id.')

    # 4. optional pruning of entities that anchor no chunk -----------------
    if prune_orphan_entities:
        # Only ALIAS edges count as anchoring: embed edges are similarity top-k,
        # so they say nothing about whether the entity is actually mentioned in
        # this corpus (an entity foreign to the document must be pruned).
        anchored = {e['entity_id'] for e in chunk_entity_edges if e['method'] == 'alias'}
        orphans = [eid for eid in entities if eid not in anchored]
        for eid in orphans:
            del entities[eid]
        entity_edges = [
            e for e in entity_edges
            if e['src_entity_id'] in entities and e['dst_entity_id'] in entities
        ]
        chunk_entity_edges = [e for e in chunk_entity_edges if e['entity_id'] in entities]
        if orphans:
            log.info(f'Pruned {len(orphans)} orphan entities: {orphans}')

    log.info(
        f'KG built: {len(entities)} entities, {len(entity_edges)} entity-edges, '
        f'{len(chunk_entity_edges)} chunk-entity edges '
        f'({sum(1 for e in chunk_entity_edges if e["method"] == "alias")} alias, '
        f'{sum(1 for e in chunk_entity_edges if e["method"] == "embed")} embed).'
    )

    return {
        'entities': entities,
        'entity_edges': entity_edges,
        'chunk_entity_edges': chunk_entity_edges,
    }
