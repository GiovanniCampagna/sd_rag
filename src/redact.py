import re
from typing import List
from ai import AI
import logging
from parse_markdown_list import parse_markdown_list
from get_best_match import get_best_match
import os


log = logging.getLogger(__name__)


def redact_rephrase(
    ai: AI,
    context: str, 
    constraints: List[str],
    batch_size: int,
) -> str:

    batches = get_batches(constraints, batch_size)
    redacted_context = context

    # Load the prompt template
    folder, _ = os.path.split(__file__)
    prompt_path = os.path.join(folder, 'rephrase-prompt.md')
    prompt_template = open(prompt_path).read()

    for batch in batches:

        prompt = prompt_template.format(
            passage=redacted_context, 
            constraints='\n'.join(['- ' + constraint for constraint in batch]),
        )
        redacted_context = ai.g.generate(
            prompt,
            max_tokens=2*ai.g.count_tokens(prompt),
        )
    
    return redacted_context

def get_batches(items: List[str], max_batch_size: int):
    return [items[i: i+max_batch_size] for i in range(0, len(items), max_batch_size)]


def get_sensitive_phrases(
    ai: AI,
    context: str, 
    constraints: List[str],
    batch_size: int,
) -> List[str]:

    batches = get_batches(constraints, batch_size)
    phrases: List[str] = []

    # Load the prompt template
    folder, _ = os.path.split(__file__)
    prompt_path = os.path.join(folder, 'extract-prompt.md')
    prompt_template = open(prompt_path).read()

    for batch in batches:

        prompt = prompt_template.format(
            constraints='\n'.join(['- ' + constraint for constraint in batch]),
            passage=context,
        )

        new_phrases = ai.g.generate_and_check_or_return_none(
            prompt, 
            is_valid=parse_markdown_list,
            max_tokens=ai.g.count_tokens(prompt),
        )

        if not new_phrases:
            continue

        phrases += new_phrases

    return phrases

def redact_extract(
    ai: AI,
    context: str, 
    constraints: List[str],
    batch_size: int,
    placeholder: str = '<REDACTED>',
) -> str:

    phrases = get_sensitive_phrases(
        ai, 
        context, 
        constraints, 
        batch_size,
    )

    # Replace every sensitive phrase span
    for phrase in phrases:

        # Skip empty or too-short phrases: get_best_match requires qlen>0 and a
        # tiny/garbage match would blow up the replace.
        if not phrase or len(phrase.strip()) < 4:
            continue

        # LLM-quoted strings aren't perfect, find the best match
        _, _, ratio, best_match = get_best_match(phrase, context)

        # Guards against degenerate replaces (they caused exploding contexts with
        # local models that return imprecise phrases):
        #  - empty or too-short best_match -> the replace self-amplifies
        #  - low-quality match -> unreliable, better not to redact
        #  - best_match already containing the placeholder -> cascading re-substitution
        if (not best_match
                or len(best_match) < 4
                or ratio < 0.6
                or placeholder in best_match):
            continue

        # Replace all instances of the best match
        context = context.replace(best_match, placeholder)

    # Deduplicate placeholders if they appear in sequence.
    pattern = r'(?:'+placeholder+r'\s+){1,}'+placeholder
    context = re.sub(pattern, placeholder, context)

    return context

