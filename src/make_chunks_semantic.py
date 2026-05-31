import re
import numpy as np
from typing import List, TYPE_CHECKING
 
if TYPE_CHECKING:
    from ai import AI
 
 
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
 
 
def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences, handling:
    - PDF soft line-wraps (single \n inside a paragraph)
    - Paragraph breaks (\n\n or more)
    - Abbreviations like S.p.A., art. 3.4 that should NOT trigger a split
 
    Returns a flat list of sentence strings with paragraph-boundary
    sentinels (empty string '') so the caller knows where hard cuts are.
    """
    text = text.strip()
    # Protect paragraph breaks
    text = re.sub(r'\n{2,}', '\x00', text)
    # Collapse soft line-wraps into spaces
    text = text.replace('\n', ' ')
    # Restore paragraph breaks
    text = text.replace('\x00', '\n\n')
 
    # Sentence boundary: .!? followed by whitespace + uppercase/digit.
    # Avoids splitting on "S.p.A.", "art. 3", "rev. 0 Agg 19".
    sentence_re = re.compile(r'(?<=[.!?])\s+(?=[A-ZÀÈÌÒÙÁÉÍÓÚ\d\"])')
 
    result: List[str] = []
    for para in text.split('\n\n'):
        para = para.strip()
        if not para:
            continue
        sentences = sentence_re.split(para)
        sentences = [s.strip() for s in sentences if s.strip()]
        result.extend(sentences)
        result.append('')   # paragraph-boundary sentinel
 
    # Remove trailing sentinel
    while result and result[-1] == '':
        result.pop()
 
    return result
 
 
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
    Split text into semantically coherent chunks by detecting topic changes,
    measured as a drop in cosine similarity between consecutive sentence
    embeddings.
 
    This is genuinely non-standard chunking: boundaries are determined by
    content, not by a fixed size parameter.
 
    Parameters
    ----------
    text : str
        Raw document text (paragraphs separated by blank lines).
    ai : AI
        The SD-RAG AI object. Uses ai.e.embed() — the same nomic-embed-text
        model already used by the rest of the pipeline.
    similarity_threshold : float
        Cosine similarity below which a sentence boundary becomes a chunk
        boundary. Lower = fewer, larger chunks. Higher = more, smaller chunks.
        Typical range: 0.35 – 0.60. Default: 0.45.
    min_sentences : int
        Never cut a chunk shorter than this many sentences, even if similarity
        drops. Prevents degenerate single-sentence chunks. Default: 1.
    max_sentences : int
        Force a cut after this many sentences regardless of similarity.
        Guards against run-on chunks on low-variability text. Default: 8.
    overlap_sentences : int
        Sentences from the end of chunk N prepended to chunk N+1.
        Keeps context across boundaries. Set 0 to disable. Default: 1.
    max_words : int
        Hard word-count cap applied after all other chunking logic.
        Chunks exceeding this limit are split every max_words words regardless
        of sentence boundaries. Acts as a last-resort guardrail for text that
        the sentence splitter cannot break (e.g. bullet lists or tables
        extracted from PDF without punctuation). Set 0 to disable. Default: 300.
 
    Returns
    -------
    List[str]
        Chunk strings ready to embed and index.
    """
 
    # --- 1. Split into sentences, noting paragraph hard-cut positions ---
    sentences = _split_sentences(text)
    if not sentences:
        return []
 
    real_sentences: List[str] = []
    hard_cuts: set[int] = set()   # indices of real sentences that start a new paragraph
 
    for s in sentences:
        if s == '':
            # next real sentence will start a new paragraph
            hard_cuts.add(len(real_sentences))
        else:
            real_sentences.append(s)
 
    if not real_sentences:
        return []
 
    # --- 2. Embed all sentences in one batched call ---
    # ai.e.embed() returns np.ndarray of shape (N, dim) with internal cache.
    embeddings = ai.e.embed(real_sentences)   # shape: (N, dim)
 
    # --- 3. Compute cosine similarity between consecutive sentences ---
    sims: List[float] = []
    for i in range(len(real_sentences) - 1):
        sims.append(_cosine_sim(embeddings[i], embeddings[i + 1]))
 
    # --- 4. Decide cut points ---
    # Cut between sentence i and i+1 when ANY of:
    #   a) paragraph hard-cut (sentence i+1 starts a new paragraph)
    #   b) similarity drop below threshold AND chunk has >= min_sentences
    #   c) chunk has reached max_sentences (hard upper limit)
    cut_after: List[bool] = [False] * len(real_sentences)
    current_chunk_size = 1
 
    for i in range(len(sims)):
        current_chunk_size += 1
        is_para_break = (i + 1) in hard_cuts
        is_sim_break  = sims[i] < similarity_threshold and current_chunk_size >= min_sentences
        is_max_break  = current_chunk_size >= max_sentences
 
        if is_para_break or is_sim_break or is_max_break:
            cut_after[i] = True
            current_chunk_size = 0
 
    # --- 5. Group sentences into raw chunks ---
    raw_chunks: List[List[str]] = []
    current: List[str] = []
    for i, sentence in enumerate(real_sentences):
        current.append(sentence)
        if cut_after[i]:
            raw_chunks.append(current)
            current = []
    if current:
        raw_chunks.append(current)
 
    # --- 6. Apply sentence-level overlap ---
    if overlap_sentences > 0 and len(raw_chunks) > 1:
        overlapped: List[List[str]] = [raw_chunks[0]]
        for i in range(1, len(raw_chunks)):
            prev = overlapped[-1]
            n = min(overlap_sentences, len(prev))
            overlapped.append(prev[-n:] + raw_chunks[i])
        joined = [' '.join(chunk) for chunk in overlapped]
    else:
        joined = [' '.join(chunk) for chunk in raw_chunks]

    # --- 7. Hard word-count cap (last-resort guardrail) ---
    # Splits any chunk exceeding max_words on word boundaries.
    # max_sentences cannot catch blocks the sentence splitter sees as a
    # single sentence (bullet lists, tables without punctuation in PDFs).
    # Set max_words=0 to disable.
    if max_words > 0:
        return _apply_word_cap(joined, max_words)
    return joined


def _apply_word_cap(chunks: List[str], max_words: int) -> List[str]:
    """
    Last-resort guardrail: split any chunk exceeding max_words on word
    boundaries. Called after overlap so the cap is applied to final output.
    Handles text the sentence splitter cannot break (bullet lists, tables
    extracted from PDF without sentence-ending punctuation).
    """
    if max_words <= 0:
        return chunks
    capped: List[str] = []
    for chunk in chunks:
        words = chunk.split()
        if len(words) <= max_words:
            capped.append(chunk)
        else:
            for start in range(0, len(words), max_words):
                capped.append(' '.join(words[start:start + max_words]))
    return capped