import os

def load_env(path: str):

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            value = value.strip('"')
            value = value.strip('\'')
            os.environ[key] = value
