from ai import AI

def generate_answer(
    ai: AI,  
    context: str, 
    query: str,
) -> str:

    prompt_template = (
        "Answer the following question according to the following context " +
        "to the best of your ability, in one single short sentence. " +
        # "If no answer is possible, answer “I am unable to answer this "
        # "question.” instead.\n\n" +
        "Question:\n {query}\n\n" +
        "Context:\n {context}\n\n" +
        "Answer:"
    )

    prompt = prompt_template.format(
        context=context,
        query=query,
    )

    answer = ai.g.generate(
        prompt,
        max_tokens=ai.g.count_tokens(context),
    )

    return answer
