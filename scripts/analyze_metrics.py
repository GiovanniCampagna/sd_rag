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

def compare_chunking_token_efficiency(df: pd.DataFrame):

    # Efficienza informativa per metodo di chunking: a parita' di completeness,
    # vince chi usa meno token di contesto ("minimo contesto, miglior completezza").
    # original_context_tokens isola il costo del chunking dalla redazione privacy.

    if 'original_context_tokens' not in df.columns:
        raise SystemExit(
            "Colonna 'original_context_tokens' assente: rilancia la pipeline "
            "(modalita' run) per produrre i conteggi token."
        )

    # Raggruppa per metodo di chunking. Usa 'label' se presente, altrimenti
    # ripiega su chunk_method / chunk_size_words.
    group_keys = [k for k in ['label', 'chunk_method'] if k in df.columns] or ['generate']

    if 'completeness_score_judge_llm' in df.columns:
        df['completeness_score_judge_llm'] = df['completeness_score_judge_llm'].map(
            lambda x: x if isinstance(x, float) else 0
        )

    value_cols = [c for c in [
        'original_context_tokens',
        'answer_context_tokens',
        'completeness_score',
        'completeness_score_judge_llm',
    ] if c in df.columns]

    table = df.groupby(group_keys)[value_cols].mean()

    # Efficienza: punti di completeness per 1000 token di contesto.
    if 'completeness_score' in table.columns:
        table['completeness_per_1k_tokens'] = (
            table['completeness_score'] / table['original_context_tokens'] * 1000
        )

    return table.sort_values('original_context_tokens')


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
        'tokeff': compare_chunking_token_efficiency,
    }

    for metric in args.metrics:
        
        get_table = name_to_fun[metric]
        print(metric)

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(get_table(df).to_string())
            # print(get_table(df).to_markdown())
            
        print('='*8)

