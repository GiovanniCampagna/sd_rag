import json
import os
import logging

log = logging.getLogger(__name__)

def save_json(obj, path: str):

    with open(path, 'w+', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        log.info(f'Saved results to {path}.')

    if os.path.exists('/content/drive/MyDrive/'):
        push_to_gdrive(path)

def push_to_gdrive(path: str):

    result = os.popen(f'cp "{path}" "/content/drive/MyDrive/"').read()
    log.info(f'Push to g-drive: "{result}"')

