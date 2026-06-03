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

    # La risposta richiesta e' "one single short sentence": bastano pochi token.
    # NB: impostare max_tokens = count_tokens(context) faceva sforare il context
    # window dei modelli locali (prompt_tokens + max_tokens > n_ctx). Gemini
    # ignora max_tokens, quindi il problema emergeva solo con llama.cpp.
    answer = ai.g.generate(
        prompt,
        max_tokens=256,
    )

    return answer
