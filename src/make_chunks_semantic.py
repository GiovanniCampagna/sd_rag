import re
import numpy as np
from typing import List, Set, TYPE_CHECKING

from make_chunks_structural import _split_sentences as _split_sentences_base

if TYPE_CHECKING:
    from ai import AI


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def _word_count(s: str) -> int:
    return len(s.split())


def _split_sentences(text: str) -> List[str]:
    """
    Wrapper around _split_sentences_base (make_chunks_structural) that adds the
    empty-string '' paragraph sentinels required by _semantic_subsplit to treat
    paragraph breaks as hard semantic cuts.
    """
    result: List[str] = []
    for para in re.split(r'\n{2,}', text.strip()):
        para = para.strip()
        if not para:
            continue
        result.extend(_split_sentences_base(para))
        result.append('')
    while result and result[-1] == '':
        result.pop()
    return result


def _semantic_subsplit(
    text: str,
    ai: 'AI',
    max_words: int,
    similarity_threshold: float,
    min_sentences: int,
    max_sentences: int,
) -> List[str]:
    """
    Core semantic splitter: embed sentences, cut where cosine similarity drops.
    Does NOT add overlap — the caller handles that.
    Used by make_chunks_semantic, make_chunks_structural_semantic, and make_chunks_adaptive.
    """
    def _cap(chs: List[str]) -> List[str]:
        # Split chunks above max_words when the sentence splitter cannot divide them
        # (e.g. very long sentences caused by numbers/abbreviations like '3.3.5',
        # 'par. 4.2'). Protects structural_semantic and adaptive, which call this
        # function directly (make_chunks_semantic already has an outer cap).
        if max_words <= 0:
            return chs
        out: List[str] = []
        for c in chs:
            w = c.split()
            if len(w) <= max_words:
                out.append(c)
            else:
                for s in range(0, len(w), max_words):
                    out.append(' '.join(w[s:s + max_words]))
        return out

    tokens = _split_sentences(text)
    if not tokens:
        return [text.strip()] if text.strip() else []

    sentences: List[str] = []
    para_cuts: Set[int] = set()

    for t in tokens:
        if t == '':
            if sentences:
                para_cuts.add(len(sentences) - 1)
        else:
            sentences.append(t)

    if not sentences:
        return [text.strip()] if text.strip() else []
    if len(sentences) == 1:
        return _cap([sentences[0]])

    embeddings = ai.e.embed(sentences)
    sims = [_cosine_sim(embeddings[i], embeddings[i + 1]) for i in range(len(sentences) - 1)]

    cut_after = [False] * len(sentences)
    cur_sents = 1
    cur_words = _word_count(sentences[0])

    for i, sim in enumerate(sims):
        next_words = cur_words + _word_count(sentences[i + 1])
        if (i in para_cuts
                or (sim < similarity_threshold and cur_sents >= min_sentences)
                or cur_sents >= max_sentences
                or (next_words > max_words and cur_sents >= min_sentences)):
            cut_after[i] = True
            cur_sents = 1
            cur_words = _word_count(sentences[i + 1])
        else:
            cur_sents += 1
            cur_words  = next_words

    chunks: List[str] = []
    group: List[str] = []
    for i, sent in enumerate(sentences):
        group.append(sent)
        if cut_after[i]:
            chunks.append(' '.join(group))
            group = []
    if group:
        chunks.append(' '.join(group))

    return _cap(chunks if chunks else [text.strip()])


def make_chunks_semantic(
    text: str,
    ai: 'AI',
    similarity_threshold: float = 0.45,
    min_sentences: int = 1,
    max_sentences: int = 8,
    max_words: int = 300,
) -> List[str]:
    """
    Split text into semantically coherent chunks by detecting topic changes
    (drops in cosine similarity between consecutive sentence embeddings).

    Parameters
    ----------
    similarity_threshold : cosine similarity below which a boundary is inserted (default 0.45)
    min_sentences        : minimum sentences per chunk before a cut is allowed (default 1)
    max_sentences        : force a cut after this many sentences (default 8)
    max_words            : hard word-count cap; 0 to disable (default 300)
    """
    if not text or not text.strip():
        return []

    chunks = _semantic_subsplit(text, ai, max_words, similarity_threshold, min_sentences, max_sentences)

    # No overlap: if two chunks are separated it is because of a similarity drop
    # (topic change); re-pasting the previous sentence would only blur the new
    # chunk's embedding and hurt retrieval.

    # Hard word-count cap. Redundant safety net: _semantic_subsplit already caps to
    # max_words via _cap, so this re-cap is normally a no-op; kept as a guard in case
    # the upstream cap is ever disabled.
    if max_words > 0:
        capped: List[str] = []
        for chunk in chunks:
            words = chunk.split()
            if len(words) <= max_words:
                capped.append(chunk)
            else:
                for start in range(0, len(words), max_words):
                    capped.append(' '.join(words[start:start + max_words]))
        return capped

    return chunks
