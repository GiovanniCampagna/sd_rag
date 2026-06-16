from ai import AI
from typing import List
from parse_markdown_list import parse_markdown_list
import os


def extract_facts(
    ai: AI, 
    text: str, 
) -> List[str]:

    folder, _ = os.path.split(__file__)
    prompt_path = os.path.join(folder, 'atomic-facts-prompt.md')
    prompt_template = open(prompt_path).read()
    prompt = prompt_template.format(text=text)

    # Floor su max_tokens: con testi corti (es. gold di 6 parole) 2*count_tokens
    # era ~24 token, troppo pochi per emettere la lista di fatti → output vuoto.
    facts = ai.g.generate_and_check(
        prompt,
        max_tokens=max(256, 3 * ai.g.count_tokens(text)),
        is_valid=parse_markdown_list,
    )

    return facts
