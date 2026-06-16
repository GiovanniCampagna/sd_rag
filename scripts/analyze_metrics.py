#!env/bin/python
import pandas as pd
import argparse

def get_mean_avg_precision_at_k_constraint_retrieval(df: pd.DataFrame):
    # Group by limit_constraints (K), method, label and take mean of AP over the queries

    df['rerank_constraints'] = df.rerank_constraints.map(lambda x: str(x))

    x = df.groupby(['rerank_constraints', 'limit_constraints'])[[
        'constraint_retrieval_precision',
    ]].mean()

    return x

def get_average_recall_at_k_constraint_retrieval(df: pd.DataFrame):
    # Group by limit_constraints (K), method (label) and take mean of recall@k over the queries

    df['rerank_constraints'] = df.rerank_constraints.map(lambda x: str(x))

    x = df.groupby(['rerank_constraints', 'limit_constraints'])[[
        'constraint_retrieval_recall',
    ]].mean()

    return x

def get_performance(df: pd.DataFrame):

    # Maybe also LLM
    return df.groupby(['generate'])[[
        'time_retrieve_chunks_ms',
        'time_retrieve_constraints_ms',
        'time_redact_ms',
        'time_answer_ms',
    ]].mean()

    # raise NotImplementedError()
    # Group by label and/or generate and compute mean of 
    # time_retrieve_chunks_ms, time_retrieve_constraints_ms, time_redact_ms, time_answer_ms

def compare_asr_interrogation(df: pd.DataFrame):

    # return df.groupby(['generate']).qed.mean()
    return df.groupby([
        'generate', 
        'inject',
    ]).qed.mean()

def compare_priv_complet_score(df: pd.DataFrame):

    df['completeness_score_judge_llm'] = df['completeness_score_judge_llm'].map(lambda x: x if isinstance(x, float) else 0)

    return df.groupby(['generate', 'inject'])[[
        'privacy_score', 
        'completeness_score_judge_llm', 
        'completeness_score',
    ]].mean()

def compare_priv_score(df: pd.DataFrame):

    return df.groupby(['generate', 'inject'])[[
        'privacy_score',
    ]].mean()

def compare_full_by_method(df: pd.DataFrame):
    # Vista completa per METODO di chunking (label): tutte le metriche affiancate.
    # Token/parole = colonne di COSTO (nessun rapporto). gap_gen = coverage - completeness
    # = fatti nel contesto ma non nella risposta (quanto il modello piccolo "spreca").
    import numpy as np

    if 'completeness_score_judge_llm' in df.columns:
        df['completeness_score_judge_llm'] = df['completeness_score_judge_llm'].map(
            lambda x: x if isinstance(x, (int, float)) else np.nan)
    if 'original_context' in df.columns:
        df['ctx_words'] = df['original_context'].map(
            lambda s: len(str(s).split()) if isinstance(s, str) else np.nan)

    group = 'label' if 'label' in df.columns else 'chunk_method'
    cols = [c for c in [
        'completeness_score_judge_llm',   # qualità (giudice LLM)
        'completeness_score',             # qualità (embedding/fatti)
        'coverage_score',                 # retrieval: fatti-gold nel contesto
        'original_context_tokens',        # costo: token contesto recuperato
        'answer_context_tokens',          # costo: token passati al generatore
        'ctx_words',                      # costo: parole di contesto
        'privacy_score',
        'constraint_retrieval_precision',
        'constraint_retrieval_recall',
    ] if c in df.columns]

    table = df.groupby(group)[cols].mean()
    if 'coverage_score' in table.columns and 'completeness_score' in table.columns:
        table['gap_gen'] = table['coverage_score'] - table['completeness_score']
    return table


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'results',
        type=str,
    )

    parser.add_argument(
        '--metrics',
        nargs='+',
        type=str,
        required=True,
    )

    args = parser.parse_args()
    df = pd.read_json(args.results, orient='records')

    name_to_fun = {
        'prec': get_mean_avg_precision_at_k_constraint_retrieval,
        'rec': get_average_recall_at_k_constraint_retrieval,
        'perf': get_performance,
        'asr': compare_asr_interrogation,
        'privcomp': compare_priv_complet_score,
        'priv': compare_priv_score,
        'full': compare_full_by_method,
    }

    for metric in args.metrics:
        
        get_table = name_to_fun[metric]
        print(metric)

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(get_table(df).to_string())
            # print(get_table(df).to_markdown())
            
        print('='*8)

