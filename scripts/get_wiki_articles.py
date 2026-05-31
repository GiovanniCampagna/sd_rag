#!env/bin/python
from typing import Dict, List
import wikipediaapi
import argparse
import shelve
import json
import os

def url_to_page_name(url: str) -> str:

    prefix = 'https://en.wikipedia.org/wiki/'
    page_name = url.lstrip(prefix)
    return page_name

def download_pages(urls: List[str], wiki, cache) -> Dict[str, str]:

    results = {}

    for url in urls:

        if url not in cache:
            page_name = url_to_page_name(url)
            page = wiki.page(page_name)
            text = page.text
            cache[url] = text
        
        results[url] = cache[url]
    
    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--project-string',
        type=str,
        default='secret.wiki-project-string.env',
        help='Path to a file with project string, e.g.: "YourProjectName (youremail@domain.tld)".'
    )

    parser.add_argument(
        '--lang',
        type=str,
        default='en',
    )

    parser.add_argument(
        '--cache',
        type=str,
        default='wiki.cache.tmp.pkl',
    )

    parser.add_argument(
        'pages',
        type=str,
        help='Path to file with a URL on each line.',
    )

    args = parser.parse_args()
    urls = open(args.pages).read().splitlines()
    urls = [url.strip() for url in urls]
    urls = [url for url in urls]
    cache = shelve.open(args.cache)

    if os.path.isfile(args.project_string):
        with open(args.project_string) as f:
            project_string = f.read()
    else:
        project_string = args.project_string

    wiki = wikipediaapi.Wikipedia(project_string, args.lang)
    d = download_pages(urls, wiki, cache)
    print(json.dumps(d))

