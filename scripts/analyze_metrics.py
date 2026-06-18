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


def compare_by_chunking_method(df: pd.DataFrame):
    """
    Confronto end-to-end PER METODO DI CHUNKING (raggruppa per 'label'): media di
    token di contesto, completeness (embedding e giudice) e privacy. Da usare coi
    risultati di configs.chunking.compare.json. Il metodo migliore e' quello con la
    completeness piu' alta al MINOR numero di context_tokens (Livello B della metrica).
    Usa solo le colonne presenti (l'eval puo' aver calcolato un sottoinsieme).
    """
    if 'completeness_score_judge_llm' in df.columns:
        df['completeness_score_judge_llm'] = df['completeness_score_judge_llm'].map(
            lambda x: x if isinstance(x, float) else 0
        )
    cols = [c for c in (
        'context_tokens',
        'redacted_context_tokens',
        'completeness_score',
        'completeness_score_judge_llm',
        'privacy_score',
    ) if c in df.columns]
    group = ['label'] + [g for g in ('generate', 'inject') if g in df.columns]
    return df.groupby(group)[cols].mean()


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
        'bychunk': compare_by_chunking_method,
    }

    for metric in args.metrics:
        
        get_table = name_to_fun[metric]
        print(metric)

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(get_table(df).to_string())
            # print(get_table(df).to_markdown())
            
        print('='*8)

