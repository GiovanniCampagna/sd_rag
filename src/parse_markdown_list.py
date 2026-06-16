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
    # # If no items, return none
    # if null_if_empty and not lines:
    #     return None

    return lines

