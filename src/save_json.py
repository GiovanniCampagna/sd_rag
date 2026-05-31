import json
import os
import logging

log = logging.getLogger(__name__)

def save_json(obj, path: str):

    with open(path, 'w+', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        log.info(f'Saved results to {path}.')

    # Salva su Google Drive solo se la cartella di output e' configurata
    # (su Colab impostare: os.environ['GDRIVE_OUT_DIR'] = '/content/drive/MyDrive/sd_rag_data')
    base = os.getenv('GDRIVE_OUT_DIR')
    if base and os.path.isdir(base):
        push_to_gdrive(path, base)

def push_to_gdrive(path: str, base: str):

    # Sceglie la sottocartella in base al tipo di file:
    # run.* -> run/ , eval.* -> eval/ , interrogate.* -> interrogate/
    name = os.path.basename(path)
    kind = name.split('.')[0] or 'altro'
    dest_dir = os.path.join(base, kind)
    os.makedirs(dest_dir, exist_ok=True)

    result = os.popen(f'cp "{path}" "{dest_dir}/"').read()
    log.info(f'Push to g-drive: "{dest_dir}" "{result}"')
