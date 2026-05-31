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
    Wrapper di _split_sentences_base (make_chunks_structural) che aggiunge
    i sentinel '' di paragrafo richiesti da _semantic_subsplit per trattare
    i parabreak come hard cut semantici.
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
        return [sentences[0]]

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

    return chunks if chunks else [text.strip()]


def make_chunks_semantic(
    text: str,
    ai: 'AI',
    similarity_threshold: float = 0.45,
    min_sentences: int = 1,
    max_sentences: int = 8,
    overlap_sentences: int = 1,
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
    overlap_sentences    : sentences from chunk N prepended to chunk N+1 (default 1)
    max_words            : hard word-count cap; 0 to disable (default 300)
    """
    if not text or not text.strip():
        return []

    chunks = _semantic_subsplit(text, ai, max_words, similarity_threshold, min_sentences, max_sentences)

    # Sentence-level overlap (prefix taken from chunks[i-1], not overlapped[-1])
    if overlap_sentences > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_sents = [s for s in _split_sentences_base(chunks[i - 1]) if s]
            n = min(overlap_sentences, len(prev_sents))
            prefix = ' '.join(prev_sents[-n:]) if n else ''
            overlapped.append((prefix + ' ' + chunks[i]).strip() if prefix else chunks[i])
        chunks = overlapped

    # Hard word-count cap (fallback for text the sentence splitter can't break)
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
