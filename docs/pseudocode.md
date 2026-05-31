
- Write pseudocode and time complexity analysis for
    - constraint binding (and index construction in general)
    - constraint retrieval
    - constraint application

## Index Construction Algorithm


```
fun build_graph:

    takes:
        - texts: list of string documents
        - constraints: list of strings
        - K: how many chunks to bind to each constraint
    
    # Create document and chunk nodes
    for each text in texts:

        doc = createDocument(text)
        documentNodes += doc
        
        for chunk in makeChunks(text):
            chunkNodes += createChunk(doc, chunk)
    
    # Compute the text embedding of each chunk node
    for each chunk in chunkNodes:

        computeEmbedding(chunk)
    
    # Attach each constraint to the K most relevant chunks
    for each constraint in constraints:

        constraintNodes += createConstraint(constraint)
        e = computeEmbedding(constraint)
        
        for chunk, score in findSimilarityScores(chunkNodes, e, K):
            edges += createEdge(constraint, chunk, weight=score)
    
    return {
        edges,
        chunkNodes,
        documentNodes,
        constraintNodes,
    }
```

## Answering Algorithm

TODO

## Constraint Retrieval Algorithm

TODO


## Constraint Application Algorithm

- NB: an important efficiency benefit of decoupling redaction from answer generation is being able to chunk the constraints and apply them in batches.

### Extractive

TODO

### Periphrastic

TODO

