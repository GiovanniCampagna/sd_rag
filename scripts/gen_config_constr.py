#!env/bin/python
import json

base =  {
    "label": "",
    "generate": "mute",
    "bind_limit": 30,

    "chunk_size_words": 60,
    "chunk_overlap_words": 0,
    "constraints_batch_size": 3,
    "limit_chunks": 3,
    "limit_constraints": 3,
}

results = []

for k in range(1, 11):

    ciao = {**base, 'limit_constraints': k}

    for method in ['wavg', 'query', 'avg', 'concat', 'max', ['query', 'avg'], ['query','concat']]:
        
        results.append({**ciao, 'rerank_constraints': method})

print(json.dumps(results))