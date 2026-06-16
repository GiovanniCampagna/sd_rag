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

    facts = ai.g.generate_and_check(
        prompt,
        max_tokens=2 * ai.g.count_tokens(text),
        is_valid=parse_markdown_list,
    )

    return facts
