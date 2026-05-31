from typing import List, Optional, TypeVar

V = TypeVar("V")

def get_rrf_ranks(
    rankings: List[List[V]], 
    limit: Optional[int] = None, 
    k: int = 60,
) -> List[V]:
    
    """
    - Rerank each document d by RRF(d)
    - RRF(d) = Σ(r ∈ R) 1 / (k + r(d))
    - d is a document
    - R is the set of rankers (retrievers)
    - k is a constant (typically 60)
    - r(d) is the rank of document d in ranker r
    """

    # Parameter k must be positive
    assert k > 0

    # Defensively deduplicate each ranking
    rankings = [[item for i, item in enumerate(ranking) if item not in ranking[:i]] for ranking in rankings]
    # Get all of the items in a flat list
    items = [item for ranking in rankings for item in ranking]
    # Deduplicate the items
    items = [item for i, item in enumerate(items) if item not in items[:i]]
    # Initialize emtpy scores
    scores: List[float] = [0] * len(items)

    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[items.index(item)] += 1 / (k + rank)
    
    # Note: tie-breaking dependns on order of received rankings
    items_and_scores = sorted(zip(items, scores), key=lambda e: e[1], reverse=True)
    items_and_scores = items_and_scores[:limit]
    items = [e for e, _ in items_and_scores]
    return items
