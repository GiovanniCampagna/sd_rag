from difflib import SequenceMatcher
from typing import Tuple


def get_best_match(
    query: str, 
    corpus: str, 
    step: int=4, 
    flex: int=3, 
    case_sensitive: bool=False,
) -> Tuple[int, int, float, str]:

    """
    Return best matching substring of corpus.
    """

    # In case lowercased, need to keep original to return match
    original_corpus = corpus

    if not case_sensitive:
        query = query.lower()
        corpus = corpus.lower()

    qlen = len(query)
    clen = len(corpus)

    assert qlen and clen

    if flex >= qlen / 2:
        flex = 3

    def match(a, b):
        return SequenceMatcher(None, a, b).ratio()

    # Initial coarse scan
    best_ratio = 0.0
    best_pos = 0

    for pos in range(0, max(clen - qlen + 1, 1), step):
        window = corpus[pos:pos + qlen]
        ratio = match(query, window)
        if ratio > best_ratio:
            best_ratio = ratio
            best_pos = pos

    # Fine-tune left/right boundaries
    best_left = best_pos
    best_right = best_pos + qlen

    for f in range(-flex, flex + 1):
        left = max(0, best_pos + f)
        for g in range(-flex, flex + 1):
            right = min(clen, best_pos + qlen + g)
            if right > left:
                ratio = match(query, corpus[left:right])
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_left = left
                    best_right = right
    
    return best_left, best_right, best_ratio, original_corpus[best_left: best_right]
