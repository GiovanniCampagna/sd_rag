# SD-RAG

This is an experimental implementation of the Selective-Disclosure Retrieval Augmented Generation (SD-RAG) framework introduced in this [paper](https://arxiv.org/abs/2601.11199).

## To install dependencies

Create and activate a virtual environment:

```sh
python -m venv env
source env/bin/activate
```

Then, if on CPU:

```sh
pip install -r requirements/cpu.txt 
```

Otherwise, if on GPU:

```sh
pip install -r requirements/gpu.txt 
```

Unfortunately, this requires CUDA to be installed globally (because of LLaMA C++). It generally works with Google Colab. Also check and change the CUDA version of the library (in [requirements/gpu.txt](requirements/gpu.txt)) to ensure it's the one compatible with your system.

## To execute

You can run the main entry-point like so:

```sh
./src/main.py
```

The entry-point allows you to:

- `forge` test sets.
- `run` tests.
- `eval` test results.
- `interrogate` pipeline via dynamic questions to test resilience to disclosure.

Depending on your goal, you may have to provide:

- `--ai` config (e.g. [ai.config.gemini.json](./configs/ai.config.gemini.json))
- `--secrets` key-value env file for API keys.
- `--configs` list for tests (e.g. [./configs/settings.json](./configs/settings.json))
- `--corpus` iterable with article texts (e.g. [./datasets/wiki.raw.json](./datasets/wiki.raw.json)).

For more options, please refer to the `--help` option.

For more config format examples, please refer to the [configs/](configs/) folder. 

Most of the code is in the [src/](src/) folder.

## Corpus extraction (reproducibility)

The professional corpus (`datasets/corpus_prof.json`) is regenerated from the
source PDF with the structured extraction pipeline (table-aware, section-aware,
self-cleaning):

```sh
python scripts/extract_pdf.py <path/to/document.pdf> datasets/corpus_prof.json
```

A naive baseline extraction (raw text dump, one corpus entry, no cleaning) is
kept for comparison:

```sh
python scripts/extract_pdf_base.py <path/to/document.pdf> datasets/corpus_prof_raw.json
```

Both scripts read and write UTF-8 explicitly, so accented Italian characters
are preserved regardless of the platform default code page.

## Other scripts

You can find other helper scripts in the [scripts/](scripts/) folder:

- [analyze_metrics.py](./scripts/analyze_metrics.py)
- [get_wiki_articles.py](./scripts/get_wiki_articles.py)

## Debugging

To debug LLM output while running tests, you can set the "verbose_tokens" flag to true for the specific model you want to read the logs of in the AI config file.

## Links to Model files (GGUF)

- https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
- https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
- https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf
- https://huggingface.co/nomic-ai/nomic-embed-text-v1-GGUF/resolve/main/nomic-embed-text-v1.f32.gguf

