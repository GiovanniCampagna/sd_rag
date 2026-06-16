from typing import List, Optional
import re

def parse_markdown_list(text: str, null_if_empty: bool = False) -> Optional[List[str]]:

    lines = text.splitlines()
    # Discard empty lines
    lines = [x for x in lines if x.strip()]
    # Discard lines that are not part of the list
    bullet_point_pattern = r'^\s*([-*]|\d+[\.\)])\s+'
    lines_and_bullets = [(line, re.match(bullet_point_pattern, line)) for line in lines]
    lines_and_bullets = [(line, bull) for line, bull in lines_and_bullets if bull]
    # Remove the bullet point from each item
    lines = [line[bull.end():] for line, bull in lines_and_bullets]
    # Remove optional quotes
    lines = [x.strip('"\'') for x in lines]
    lines = [x for x in lines if x.strip()]
    # Se non ci sono elementi, ritorna None: così generate_and_check (is_valid)
    # considera l'output non valido e fa il retry con invalidate=True, invece di
    # accettare una lista vuota. Serve perché la generazione e' stocastica
    # (temperature 0.2) e ~1 volta su 3 qwen collassa su un solo '-'.
    if null_if_empty and not lines:
        return None

    return lines

