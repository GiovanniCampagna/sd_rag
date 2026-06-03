from ai import AI

def generate_answer(
    ai: AI,
    context: str,
    query: str,
) -> str:

    prompt_template = (
        "Answer the following question according to the following context "
        "to the best of your ability, in one single short sentence. "
        "Answer in the same language as the question.\n\n"
        "Question:\n {query}\n\n"
        "Context:\n {context}\n\n"
        "Answer:"
    )

    prompt = prompt_template.format(
        context=context,
        query=query,
    )

    # --- DEBUG temporaneo: individua cosa gonfia il prompt ---
    import sys
    _ctx_tok = ai.g.count_tokens(context)
    _q_tok = ai.g.count_tokens(query)
    if _ctx_tok > 1500 or _q_tok > 1500:
        print(f"[DEBUG] context={_ctx_tok}tok ({len(context)}char) "
              f"query={_q_tok}tok ({len(query)}char)", file=sys.stderr)
        print(f"[DEBUG] context head: {context[:200]!r}", file=sys.stderr)
        print(f"[DEBUG] context tail: {context[-200:]!r}", file=sys.stderr)
    # --- fine DEBUG ---

    # La risposta richiesta e' "one single short sentence": bastano pochi token.
    # NB: impostare max_tokens = count_tokens(context) faceva sforare il context
    # window dei modelli locali (prompt_tokens + max_tokens > n_ctx). Gemini
    # ignora max_tokens, quindi il problema emergeva solo con llama.cpp.
    answer = ai.g.generate(
        prompt,
        max_tokens=256,
    )

    return answer
