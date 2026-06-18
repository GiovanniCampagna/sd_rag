from typing import List, Optional, Tuple
from uuid import uuid4
from ai import get_similarity_scores, cosine_sim, AI
from elapsed_time import timeit
from generate_redacted_answer_monolith import generate_redacted_answer_monolith
from generate_answer import generate_answer
from redact import redact_extract, redact_rephrase
from prepare_context import prepare_context
from schemas  import *
from get_prompt_leak_query import get_prompt_leak_query
import logging


log = logging.getLogger(__name__)


def retrieve_chunks(
    ai: AI,
    query: str,
    index: SDRAGIndex,
    limit: int,
) -> List[Tuple[ChunkNode, float]]:

    chunks = list(index['chunks'].values())
    chunk_embeddings = ai.e.embed([c['text'] for c in chunks])
    query_embedding = ai.e.embed(query)
    chunks_and_scores = get_similarity_scores(
        query_embedding,
        chunks,
        chunk_embeddings,
        limit,
    )
    return chunks_and_scores

def get_answer(
    ai: AI,
    index: SDRAGIndex,
    config: QueryConfig,
    query: str,
    query_id: str = '',

    injection: Optional[PromptLeakConfig] = None,

) -> Answer:

    # Prepare injected query
    injection = injection or {'inject': False}
    query = get_prompt_leak_query(query, injection)

    chunks_and_scores, time_retrieve_chunks_ms = timeit(
        retrieve_chunks,
        ai,
        query,
        index,
        config['limit_chunks'],
    )

    chunks = [c for c, _ in chunks_and_scores]
    context = prepare_context(chunks)

    constraints, time_retrieve_constraints_ms = timeit(
        retrieve_constraints,
        ai, 
        query, 
        chunks_and_scores, 
        index, 
        context,
        config['limit_constraints'],
        config['rerank_constraints'],
    )
    
    match config['generate']:

        case 'plain':

            redacted_context = ''
            time_redact_ms = 0
            answer_text, time_answer_ms = timeit(
                generate_answer,
                ai,
                context,
                query,
            )
        
        case 'mute':

            redacted_context = ''
            time_redact_ms = 0
            time_answer_ms = 0
            answer_text = ''
        
        case 'monolith':

            redacted_context = ''
            time_redact_ms = 0
            answer_text, time_answer_ms = timeit(
                generate_redacted_answer_monolith, 
                ai=ai,
                query=query, 
                context=context, 
                constraints=[get_constraint_text(c) for c in constraints],
            )
        
        case 'extract':

            redacted_context, time_redact_ms = timeit(
                redact_extract,
                ai=ai,
                context=context,
                constraints=[get_constraint_text(c) for c in constraints],
                batch_size=config['constraints_batch_size'],
            )

            answer_text, time_answer_ms = timeit(
                generate_answer,
                ai,
                redacted_context,
                query,
            )
        
        case 'rephrase':

            redacted_context, time_redact_ms = timeit(
                redact_rephrase,
                ai=ai,
                context=context,
                constraints=[get_constraint_text(c) for c in constraints],
                batch_size=config['constraints_batch_size'],
            )

            answer_text, time_answer_ms = timeit(
                generate_answer,
                ai,
                redacted_context,
                query,
            )
    
    return {
        'answer_id': str(uuid4()),
        'query_id': query_id,
        'query': query,
        'answer_text': answer_text,
        'retrieved_chunks': chunks,
        'retrieved_constraints': constraints,
        'time_answer_ms': time_answer_ms,
        'time_redact_ms': time_redact_ms,
        'time_retrieve_chunks_ms': time_retrieve_chunks_ms,
        'time_retrieve_constraints_ms': time_retrieve_constraints_ms,
        'model_names': ai.model_names(),
        'original_context': context,
        'redacted_context': redacted_context,
        **config,
        **injection,
    }

def get_constraint_text(constraint: ConstraintNode) -> str:

    return constraint['action'] + ' ' + constraint['sensitive_category']


def retrieve_constraints(
    ai: AI, 
    query: str, 
    chunks_and_scores: List[Tuple[ChunkNode, float]], 
    index: SDRAGIndex,
    context: str,
    limit: int,
    rerank: List[ConstraintRerankMethod] | ConstraintRerankMethod,
) -> List[ConstraintNode]:

    # Get the constraints attached to the retrieved chunks (candidate constraints)
    retrieved_chunks = [x for x, _ in chunks_and_scores]
    retrieved_chunk_ids = [x['ID'] for x in retrieved_chunks]
    chunk_scores = [s for _, s in chunks_and_scores]
    relevant_edges = [e for e in index['chunk_constraint_edges'] if e['chunk_id'] in retrieved_chunk_ids]
    candidate_constraint_ids = {e['constraint_id'] for e in relevant_edges}
    constraints = [index['constraints'][cid] for cid in candidate_constraint_ids]

    # Get mapping from chunk ID to constraint scores
    chunk_id_to_constraint_scores = {c['ID']: {e['constraint_id'] : e['score'] for e in index['chunk_constraint_edges'] if e['chunk_id'] == c['ID']} for c in retrieved_chunks}

    consider_doc_importance = rerank == 'wavg'
    # Rank the constraints by average similarity to retrieved docs.
    weights = chunk_scores if consider_doc_importance else [1] * len(chunk_scores) 
    constraint_id_to_scores = {constr['ID'] : [s * chunk_id_to_constraint_scores[chunk['ID']].get(constr['ID'], 0) for s, chunk in zip(weights, retrieved_chunks)] for constr in constraints}


    rerank_criteria: List[ConstraintRerankMethod] = rerank if isinstance(rerank, list) else [rerank]
    rankings: List[List[ConstraintNode]] = []

    for criterion in rerank_criteria:

        rankings.append(rerank_constraints(
            ai, 
            constraints, 
            criterion, 
            context, 
            constraint_id_to_scores, 
            query,
        ))

    from get_rrf_ranks import get_rrf_ranks
    final_ranking = get_rrf_ranks(rankings, limit)

    return final_ranking


def rerank_constraints(
    ai: AI,
    constraints: List[ConstraintNode], 
    rerank: ConstraintRerankMethod,
    context: str,
    constraint_id_to_scores: Dict[str, List[float]],
    query: str,
) -> List[ConstraintNode]:

    avg = lambda xs: sum(xs) / len(xs)

    def get_score(constraint: ConstraintNode):

        match rerank:
            
            case 'concat':
                ectx = ai.e.embed(context)[0]
                ecns = ai.e.embed(constraint['sensitive_category'])[0]
                return cosine_sim(ectx, ecns)
            case 'query':
                ectx = ai.e.embed(query)[0]
                ecns = ai.e.embed(constraint['sensitive_category'])[0]
                return cosine_sim(ectx, ecns)
            case 'max':
                scores = constraint_id_to_scores[constraint['ID']]
                return max(scores)
            case 'wavg' | 'avg':
                scores = constraint_id_to_scores[constraint['ID']]
                return avg(scores)
            
            # TODO: also use proximity of chunks to infer possible related constraints!!!!! But 
            # in that case re-ranking has to be done in a way that it is not guaranteed
            # that constraints most similar to chunks/query are always favored?
            # Intuitive idea: constraints attached to nearby (non retrieved) chunks
            # should also be taken into consideration.

    constraints_and_scores = [(c, get_score(c)) for c in constraints]     
    reranked_constraints_and_scores = sorted(constraints_and_scores, key=lambda e: e[1], reverse=True)
    reranked_constraints = [x for x, _ in reranked_constraints_and_scores]
    return reranked_constraints
