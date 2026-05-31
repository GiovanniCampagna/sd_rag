import re

def make_chunks(text:str, size:int, overlap:int=0):
    """
    Split text into chunks under `size` words.
    
    Rules:
    - Do not split sentences unless a sentence exceeds size.
    - Long sentences are split by phrase punctuation (, ; :)
    - If a phrase is still too long, it is kept intact.
    - Optional word overlap between consecutive chunks.
    """

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        
        if not sentence:
            continue

        if word_count(sentence) <= size:
            if word_count(current_chunk) + word_count(sentence) <= size:
                current_chunk = (current_chunk + " " + sentence).strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        else:
            # Split long sentence into phrases
            phrases = re.split(r'(?<=[,;:])\s+', sentence)

            for phrase in phrases:
                if word_count(phrase) <= size:
                    if word_count(current_chunk) + word_count(phrase) <= size:
                        current_chunk = (current_chunk + " " + phrase).strip()
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = phrase
                else:
                    # Phrase still too long → force as its own chunk
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                    chunks.append(phrase.strip())

    if current_chunk:
        chunks.append(current_chunk)

    # Apply overlap (prefix taken from chunks[i-1], not overlapped[-1],
    # to prevent cascading prefix growth across chunks)
    if overlap > 0 and chunks:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prefix = last_words(chunks[i - 1], overlap)
            overlapped.append((prefix + " " + chunks[i]).strip())
        return overlapped

    return chunks

def word_count(s: str):
    return len(s.split())

def last_words(s: str, n: int):
    words = s.split()
    return " ".join(words[-n:]) if n > 0 else ""
