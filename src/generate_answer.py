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

    import logging; logging.getLogger(__name__).info(f"[DEBUG2] prompt_tokens={ai.g.count_tokens(prompt)}")

    # The requested answer is "one single short sentence": a few tokens are enough.
    # NB: setting max_tokens = count_tokens(context) overflowed the context window
    # of local models (prompt_tokens + max_tokens > n_ctx). Gemini ignores
    # max_tokens, so the problem only showed up with llama.cpp.
    answer = ai.g.generate(
        prompt,
        max_tokens=256,
    )

    return answer
