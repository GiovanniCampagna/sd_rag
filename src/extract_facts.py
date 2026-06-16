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

    # La generazione e' stocastica (temperature 0.2 di default): su alcuni gold
    # qwen ~1 volta su 3 collassa su un solo '-' -> lista vuota. null_if_empty=True
    # rende quell'output non valido, cosi' generate_and_check fa il retry con
    # invalidate=True (nuovo campionamento) fino a max_attempts, invece di
    # accettare [] e far crashare l'assert a valle.
    facts = ai.g.generate_and_check(
        prompt,
        max_tokens=2 * ai.g.count_tokens(text),
        is_valid=lambda raw: parse_markdown_list(raw, null_if_empty=True),
    )

    return facts
