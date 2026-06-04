from typing import List
from ai import AI, ChatMessage


def generate_redacted_answer_monolith(
    ai: AI, 
    context: str, 
    query: str, 
    constraints: List[str],
) -> str:

    '''
    Adapted from: https://aclanthology.org/2024.nlpaics-1.25.pdf
    '''

    if ai.g.supports_system_prompt():

        prompt: List[ChatMessage] | str = [
            {
                "role":"system",
                "content": ( 
                    f"Answer a question using the following story:\n\n{context}\n\n" +
                    'The answer must obey the following constraint(s):\n' +
                    '\n'.join(['- ' + c for c in constraints]) + '\n\n'
                )
            },
            {
                "role": "user",
                "content": (
                    "Answer the question to the best of your ability using " +
                    "a single sentence and without violating the provided " +
                    "constraint(s). Answer in the same language as the question. " +
                    "If no answer is possible, answer " +
                    "“I am unable to answer this question.” instead.\n" +
                    f"Question: {query}\n\n"
                )
            }
        ]

    else:

        prompt = (
            f'Answer a question using the following story:\n\n{context}\n\n' +
            'The answer must obey the following constraint(s):\n' +
            '\n'.join(['- ' + c for c in constraints]) + '\n\n' +
            "Answer the question to the best of your ability using " +
            "a single sentence and without violating the provided " +
            "constraint(s). Answer in the same language as the question. " +
            "If no answer is possible, answer " +
            "“I am unable to answer this question.” instead.\n" +
            f"Question: {query}\n\n"
        )


    answer_text = ai.g.generate(
        prompt, 
        max_tokens = ai.g.count_tokens(context), # avoid infinite loop issue
    )

    return answer_text
