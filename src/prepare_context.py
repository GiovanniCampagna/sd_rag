from typing import Dict, List
from schemas import ChunkNode

def prepare_context(chunks: List[ChunkNode]) -> str:

    # Group the chunks by document of provenance
    doc_to_chunks: Dict[str, List[ChunkNode]] = {}
    context = ''

    for chunk in chunks:
        
        if chunk['doc_id'] not in doc_to_chunks:
            doc_to_chunks[chunk['doc_id']] = []
        
        doc_to_chunks[chunk['doc_id']].append(chunk)
    
    for doc_id, chunks in doc_to_chunks.items():
        # Sort the chunks by their part_num (order of appearence)
        doc_to_chunks[doc_id] = sorted(doc_to_chunks[doc_id], key=lambda e: e['part_num'])

    for i, (doc_id, chunks) in enumerate(doc_to_chunks.items()):
        
        for chunk in chunks:
            # heading = f'Document {i} (part {chunk["part_num"]})'
            # context += heading + '\n\n'
            context += chunk['text'] + '\n\n'
    
    return context

