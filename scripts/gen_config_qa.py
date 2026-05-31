#!env/bin/python
import json

base =  {
    "label": "",
    "bind_limit": 30,
    "chunk_size_words": 60,
    "chunk_overlap_words": 0,
    "constraints_batch_size": 3,
    "limit_chunks": 3,
    "limit_constraints": 3,
    "rerank_constraints": "query",
    # "inject_max_query_percent": 1.1,
    "inject_max_query_percent": 0.9,
    "inject_repeat": 2,
}

results = []

for generate in ['monolith', 'rephrase', 'extract']:

    ciao = {**base, 'generate': generate}

    for inject in [True, False]:
    
        results.append({**ciao, 'inject': inject})

print(json.dumps(results))