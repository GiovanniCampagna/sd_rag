from typing import Dict, List
from make_chunks import make_chunks
from make_chunks_semantic import make_chunks_semantic
from make_chunks_structural import make_chunks_structural
from make_chunks_structural_semantic import make_chunks_structural_semantic
from make_chunks_adaptive import make_chunks_adaptive
from uuid import uuid4
from ai import get_similarity_scores, AI
from schemas import *
import logging
 
 
log = logging.getLogger(__name__)
 
def build_index(
    ai: AI,
    doc_texts: List[str], 
    sensitive_categories: List[str], 
    config: BuildSDRAGIndexConfig,
) -> SDRAGIndex:
 
    log.info('Building index.')
 
    chunks: Dict[str, ChunkNode] = {}
    docs: Dict[str, DocNode] = {}
    constraints: Dict[str, ConstraintNode] = {}
    constraint_edges: List[ChunkConstraintEdge] = []
 
    # Read chunking method from config.
    # Default: 'word' → original behaviour, no breaking change.
    chunk_method = config.get('chunk_method', 'word')
 
    for doc_text in doc_texts:
 
        doc_id = str(uuid4())
 
        docs[doc_id] = {
            'ID': doc_id,
            'text': doc_text,
        }
 
        if chunk_method == 'semantic':
            chunk_texts = make_chunks_semantic(
                doc_text,
                ai=ai,
                similarity_threshold=config.get('similarity_threshold', 0.45),
                max_words=config.get('max_words', 300),
                min_sentences=config.get('min_sentences', 1),
                max_sentences=config.get('max_sentences', 8),
            )
            log.info(
                f'Semantic chunking: {len(chunk_texts)} chunks '
                f'(threshold={config.get("similarity_threshold", 0.45)}, '
                f'max_words={config.get("max_words", 300)}), '
                f'max_sentences={config.get("max_sentences", 8)})'
            )
        elif chunk_method == 'structural':
            chunk_texts = make_chunks_structural(
                doc_text,
                max_words=config.get('max_words', 200),
                min_words=config.get('min_words', 25),
            )
            log.info(
                f'Structural chunking: {len(chunk_texts)} chunks '
                f'(max_words={config.get("max_words", 200)}, '
                f'min_words={config.get("min_words", 25)})'
            )
        elif chunk_method == 'structural_semantic':
            chunk_texts = make_chunks_structural_semantic(
                doc_text,
                ai=ai,
                max_words=config.get('max_words', 200),
                min_words=config.get('min_words', 25),
                similarity_threshold=config.get('similarity_threshold', 0.45),
                min_sentences=config.get('min_sentences', 1),
                max_sentences=config.get('max_sentences', 8),
            )
            log.info(
                f'Structural-semantic chunking: {len(chunk_texts)} chunks '
                f'(max_words={config.get("max_words", 200)}, '
                f'min_words={config.get("min_words", 25)}, '
                f'threshold={config.get("similarity_threshold", 0.45)}, '
                f'max_sentences={config.get("max_sentences", 8)})'
            )
        elif chunk_method == 'adaptive':
            chunk_texts = make_chunks_adaptive(
                doc_text,
                ai=ai,
                max_words=config.get('max_words', 200),
                min_words=config.get('min_words', 25),
                similarity_threshold=config.get('similarity_threshold', 0.45),
                min_sentences=config.get('min_sentences', 1),
                max_sentences=config.get('max_sentences', 8),
                tabular_max_rows=config.get('tabular_max_rows', 3),
                tabular_max_words=config.get('tabular_max_words', 200),
                inject_section_heading=config.get('inject_section_heading', True),
                heading_like_max_words=config.get('heading_like_max_words', 10),
                indented_list_density=config.get('indented_list_density', 0.70),
            )
            log.info(
                f'Adaptive chunking: {len(chunk_texts)} chunks '
                f'(max_words={config.get("max_words", 200)}, '
                f'min_words={config.get("min_words", 25)}, '
                f'threshold={config.get("similarity_threshold", 0.45)}, '
                f'max_sentences={config.get("max_sentences", 8)}, '
                f'tabular_max_rows={config.get("tabular_max_rows", 3)}, '
                f'tabular_max_words={config.get("tabular_max_words", 200)}, '
                f'inject_heading={config.get("inject_section_heading", True)})'
            )
        else:
            chunk_texts = make_chunks(
                doc_text,
                config['chunk_size_words'],
                config['chunk_overlap_words'],
            )
            log.info(
                f'Word chunking: {len(chunk_texts)} chunks '
                f'(size={config["chunk_size_words"]}, '
                f'overlap={config["chunk_overlap_words"]})'
            )
 
        for part_num, chunk_text in enumerate(chunk_texts):
 
            chunk_id = str(uuid4())
            chunks[chunk_id] = {
                'ID': chunk_id,
                'text': chunk_text,
                'part_num': part_num,
                'doc_id': doc_id,
            }
        
    log.info(f'Inserted {len(docs)} documents into index.')
    log.info(f'Inserted {len(chunks)} chunks into index.')
 
    chunks_list = list(chunks.values())
    chunk_embeddings = ai.e.embed([c['text'] for c in chunks_list])
    
    for category in sensitive_categories:
 
        category_id = str(uuid4())
        constraints[category_id] = {
            'ID': category_id,
            'sensitive_category': category,
            'action': 'remove',
        }
 
        category_embedding = ai.e.embed(category)
        chunks_and_scores = get_similarity_scores(
            category_embedding,
            chunks_list,
            chunk_embeddings,
            limit=config['bind_limit'],
        )
 
        for chunk, score in chunks_and_scores:
 
            constraint_edges.append({
                'chunk_id': chunk['ID'],
                'constraint_id': category_id,
                'score': score,
            })
    
    log.info(f'Inserted {len(constraints)} constraints into index.')
    log.info(f'Inserted {len(constraint_edges)} constraints-chunk edges.')
    log.info('Done building index.')
 
    return {
        'chunk_constraint_edges': constraint_edges,
        'chunks': chunks,
        'constraints': constraints,
        'docs': docs,
    }