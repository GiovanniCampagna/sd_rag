import ctypes
import hashlib
import json
import os
import shelve
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Hashable, Iterator, List, Literal, MutableMapping, Optional, Sequence, Tuple, TypedDict, cast
import numpy as np
from google import genai
from llama_cpp import ChatCompletionRequestMessage, Llama, LlamaGrammar, llama_log_set
from typing_extensions import NotRequired, TypeVar
import logging
import sys


log = logging.getLogger(__name__)

"""
This is a compact library that implements LLM response caching, logging, text chunking,
vector index search, and a single interface for multi-LLM management with 2 backends.
"""

# silence llamacpp logs -----------------------------------------------------------------------------------
log_callback = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)(
    lambda l, m, u: None
)
llama_log_set(log_callback, ctypes.c_void_p())


class BaseLLMConfig(TypedDict):
    name: str
    cache: "CacheConfig"
    # provider-specific options for model init
    opts: NotRequired[Dict[str, Any]]
    # provider-specific options for inference
    inference_opts: NotRequired[Dict[str, Any]]
    default: NotRequired[bool]
    max_attempts: NotRequired[int]
    # Print each token to stderr
    verbose_tokens: NotRequired[bool]

    supports_system_prompt: NotRequired[bool]

class LlamaConfig(BaseLLMConfig):
    gguf_path: str
    chat_format: NotRequired[str]
    n_ctx: NotRequired[int]
    embed: NotRequired[bool]

class GeminiConfig(BaseLLMConfig):
    google_api_key: NotRequired[str]
    model_name: str

class GeminiColabConfig(BaseLLMConfig):
    colab: bool
    model_name: NotRequired[str]

LLMConfig = LlamaConfig | GeminiConfig | GeminiColabConfig

class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str

class CacheConfig(TypedDict):
    path: NotRequired[str]
    "Path to cache file."
    readonly: NotRequired[bool]
    "Loads cache from file (if exists, else empty init) and only updates it in-memory."
    keep_open: NotRequired[bool]
    "Keep persistent cache file open, faster but not guaranteed to release file."

def get_cache(config: CacheConfig) -> MutableMapping:
    readonly = "readonly" in config and config["readonly"]
    keep_open = "keep_open" in config and config["keep_open"]

    if readonly:
        # There is no path, so user wants empty dict
        if "path" not in config:
            return {}

        # Otherwise readable file must already exist
        assert os.path.isfile(config["path"]), "Readonly cache path not found!"

        # Return copy of data in readonly file
        with shelve.open(config["path"], flag="r") as s:
            return dict(s)

    # User wants a persistent dict, so now path string must be provided
    assert "path" in config, "Missing cache file path!"

    if keep_open:
        return shelve.open(config["path"])
    else:
        return SafePersistentDict(config["path"])


@dataclass
class SafePersistentDict(dict):
    path: str
    T = TypeVar("T")

    def do_to_cache(self, foo: Callable[[MutableMapping], T]) -> T:
        with shelve.open(self.path) as cache:
            return foo(cache)

    def __setitem__(self, key: str, value):
        return self.do_to_cache(lambda x: x.__setitem__(key, value))

    def __getitem__(self, key: str):
        return self.do_to_cache(lambda x: x[key])

    def __contains__(self, key: object):
        return self.do_to_cache(lambda x: key in x)

    def __repr__(self) -> str:
        return self.do_to_cache(lambda x: str(dict(x)))

    def update(self, kwargs) -> None: # pyright:ignore
        self.do_to_cache(lambda x: x.update(**kwargs))

def get_hash(text: str) -> str:
    hash_obj = hashlib.md5(text.encode("utf-8"))
    digest = hash_obj.hexdigest()
    return str(uuid.UUID(hex=digest))

def count_tokens_llama(model: Llama, text: str) -> int:
    return len(model.tokenize(text.encode("utf-8")))

def count_tokens_google(client: genai.Client, model_name: str, text: str) -> int:
    response = client.models.count_tokens(
        model=model_name,
        contents=text,
    )

    token_count = response.total_tokens
    assert token_count
    return token_count


def embed_llama(model: Llama, texts: List[str]) -> np.ndarray:
    """
    There is a bug in the current version of llama-cpp-python:
    https://github.com/abetlen/llama-cpp-python/issues/2051
    """

    embeddings = model.embed(texts)
    embeddings = np.array(embeddings, dtype="float32")
    return embeddings


def get_stream_llama(
    model: Llama,
    messages: str | List[ChatMessage],
    grammar: str | None = None,
    verbose_tokens: bool = False,
    **kwargs,
) -> Iterator[str]:
    messages = (
        [{"content": messages, "role": "user"}]
        if isinstance(messages, str)
        else messages
    )

    stream = model.create_chat_completion(
        cast(List[ChatCompletionRequestMessage], messages),
        grammar=LlamaGrammar.from_string(grammar) if grammar else None,
        stream=True,
        **kwargs,
    )

    assert isinstance(stream, Iterator)

    for resp in stream:
        delta = resp["choices"][0]["delta"]

        if "content" in delta:
            content = delta["content"]
            if content is None:
                continue

            if verbose_tokens:
                print(content, end='', sep='', flush=True, file=sys.stderr)

            yield content
        else:
            yield ""


def generate_google(
    client: genai.Client,
    model_name: str,
    messages: str | List[ChatMessage],
    **kwargs,
) -> str:
    messages = str(messages)

    response = client.models.generate_content(
        model=model_name,
        contents=messages,
    )

    text = response.text
    assert text
    return text


def parse_json(raw: str) -> Optional[Any]:
    if "```json" in raw:
        s = raw.index("```json") + len("```json")
        e = raw.index("```", s)
        raw = raw[s:e]

    try:
        data = json.loads(raw)
        return data
    except:
        return None


def cosine_sim_strings(ai: "AI", a: str, b: str) -> float:
    return cosine_sim(ai.e.embed(a)[0], ai.e.embed(b)[0])


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_embedding_dim_llama(model: Llama) -> int:
    return embed_llama(model, ["hello"]).shape[1]


class LLM:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.cache = get_cache(config["cache"])

    def log(self, *args: object, **kwargs):
        if "log" in self.config:
            self.config["log"](*args, **kwargs)

    def count_tokens(self, text: str, invalidate: bool = False) -> int:
        key = get_hash("<COUNT-TOKENS>" + text)

        if key not in self.cache or invalidate:
            self.cache[key] = self._count_tokens(text)

        return self.cache[key]

    G = TypeVar("G", default=str)

    def generate_and_check_or_return_none(
        self,
        messages: str | List[ChatMessage],
        grammar: str | None = None,
        invalidate: bool = False,
        is_valid: Callable[[str], Optional[G]] = lambda _: _,
        max_attempts: Optional[int] = None,
        **kwargs,
    ) -> Optional[G]:
        
        max_attempts = max_attempts or (self.config['max_attempts'] if 'max_attempts' in self.config else None) or 1

        for i in range(max_attempts):
            
            raw = self.generate(
                messages=messages, 
                grammar=grammar, 
                invalidate=invalidate, 
                **kwargs,
            )

            x = is_valid(raw)

            if x is not None:
                return x
            else:
                log.info(f"Invalid LLM output, retrying {i}")
                invalidate = True
        
        log.info(f'Failed to validate LLM output: {raw}') # pyright:ignore
        return None

    def generate_and_check(
        self,
        messages: str | List[ChatMessage],
        grammar: str | None = None,
        invalidate: bool = False,
        is_valid: Callable[[str], Optional[G]] = lambda _: _,
        max_attempts: Optional[int] = None,
        **kwargs,
    ) -> G:

        output = self.generate_and_check_or_return_none(
            messages=messages,
            grammar=grammar,
            invalidate=invalidate,
            is_valid=is_valid,
            max_attempts=max_attempts,
            **kwargs,
        )

        if output is not None:
            return output
        
        raise Exception("Max attempts exceeded.")

        # max_attempts = max_attempts or (self.config['max_attempts'] if 'max_attempts' in self.config else None) or 1

        # for i in range(max_attempts):
            
        #     raw = self.generate(
        #         messages=messages, 
        #         grammar=grammar, 
        #         invalidate=invalidate, 
        #         **kwargs,
        #     )

        #     x = is_valid(raw)

        #     if x is not None:
        #         return x
        #     else:
        #         log.info(f"Invalid LLM output, retrying {i}")
        #         invalidate = True

        # raise Exception("Max attempts exceeded.")

    def generate(
        self,
        messages: str | List[ChatMessage],
        grammar: str | None = None,
        invalidate: bool = False,
        **kwargs,
    ) -> str:

        log.info(
            'LLM: recevied prompt="""'
            + str(messages).encode("unicode_escape").decode("utf-8")[:40]
            + "..."
            + '"""'
        )

        key = get_hash("<GENERATE>" + str(messages) + str(grammar) + str(kwargs))

        if key not in self.cache or invalidate:
            log.info("LLM cache miss or invalidate.")

            self.cache[key] = self._generate(
                messages=messages,
                grammar=grammar,
                **kwargs,
            )

        return self.cache[key]

    # T = TypeVar('T')
    # def embed_smoothed(
    #     self,
    #     elements: Sequence[T],
    #     get_neighbors: Callable[[T], List[T]],
    #     stringify: Callable[[T], str],
    #     embed_batch_size: int = 1,
    #     invalidate: bool = False,
    # ) -> 'VectorStore':

    #     # Create a collective hash use it as key, if key not found, recompute all, otherwise return all

    #     key = get_hash(str(sorted(elements))) # pyright:ignore

    #     if key in self.cache:
    #         return self.cache[key]

    #     strings = [stringify(unit) for unit in elements]
    #     embeddings = self.embed(
    #         strings,
    #         embed_batch_size=embed_batch_size,
    #         invalidate=invalidate,
    #     )

    #     store = VectorStore(elements, embeddings)

    #     # Embedding smoothing -------------------
    #     alpha = 0.6
    #     beta = 1 - alpha
    #     for i, unit in enumerate(elements, start=1):
    #         neighbors = get_neighbors(unit)
    #         neighbor_embeddings = store.getmany(neighbors)
    #         unit_embedding = store[unit]
    #         avg_neighbor_embedding = np.average(neighbor_embeddings, axis=0)
    #         store[unit] = alpha * unit_embedding + beta * avg_neighbor_embedding
    #         log.info(f'Smoothing embedding {i} / {len(elements)}')

    #     # return store.embeddings
    #     return store

    def embed(
        self,
        texts: List[str] | str,
        embed_batch_size: int = 1,
        invalidate: bool = False,
    ) -> np.ndarray:
        # Use a hashed key: dbm backends (e.g. ndbm on Linux) have a limit on the
        # key length, and chunk texts can exceed it causing a KeyError. The hash is
        # short and of fixed length.
        def ckey(text: str) -> str:
            return get_hash("<EMBED>" + text)

        # Calculate the embeddings that are missing from the lookup table
        texts = [texts] if isinstance(texts, str) else texts
        to_be_computed = (
            texts if invalidate else [x for x in texts if ckey(x) not in self.cache]
        )
        batches = [
            to_be_computed[i : i + embed_batch_size]
            for i in range(0, len(to_be_computed), embed_batch_size)
        ]

        for i, batch in enumerate(batches, start=1):
            embeddings = self._embed(batch)
            self.cache.update({ckey(t): e for t, e in zip(batch, embeddings)})
            log.info(f"Embeding batches {i} / {len(batches)}")

        # Return the appropriate embeddings from the lookup table
        results = [self.cache[ckey(x)] for x in texts]
        results = np.array(results, dtype="float32")
        return results

    def _embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError()

    def get_embedding_dim(self) -> int:
        raise NotImplementedError()

    def _count_tokens(self, text: str) -> int:
        raise NotImplementedError()

    def _generate(
        self,
        messages: str | List[ChatMessage],
        grammar: str | None = None,
        **kwargs,
    ) -> str:
        raise NotImplementedError()

    def get_stream(
        self,
        messages: str | List[ChatMessage],
        grammar: str | None = None,
        **kwargs,
    ) -> Iterator[str]:
        raise Exception()

    
    def supports_system_prompt(self) -> bool:
        return self.config['supports_system_prompt'] if 'supports_system_prompt' in self.config else False


@dataclass
class LlamaLLM(LLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.model = Llama(
            model_path=os.path.expanduser(config["gguf_path"]),
            chat_format=config["chat_format"] if "chat_format" in config else None,
            verbose=config.get("opts", {}).get("verbose", False),
            n_ctx=config["n_ctx"] if "n_ctx" in config else 5_000,
            embedding=config["embed"] if "embed" in config else False,
            **(self.config["opts"] if "opts" in self.config else {}),
        )

    def _count_tokens(self, text: str) -> int:
        return count_tokens_llama(self.model, text)

    def _embed(self, texts: List[str]) -> np.ndarray:
        return embed_llama(self.model, texts)

    def get_embedding_dim(self) -> int:
        return get_embedding_dim_llama(self.model)

    def _generate(
        self,
        messages: str | List[ChatMessage],
        grammar: str | None = None,
        **kwargs,
    ) -> str:

        stream = get_stream_llama(
            model=self.model,
            messages=messages,
            grammar=grammar,
            verbose_tokens=self.config['verbose_tokens'] if 'verbose_tokens' in self.config else False,
            **{
                **(self.config['inference_opts'] if 'inference_opts' in self.config else {}),
                **kwargs,
            },
        )

        accumulator = ""

        for token in stream:
            accumulator += token

        return accumulator

    # def get_stream(
    #     self,
    #     messages: str | List[ChatMessage],
    #     grammar: str | None = None,
    #     **kwargs,
    # ) -> Iterator[str]:

    #     return get_stream_llama(
    #         model=self.model,
    #         messages=messages,
    #         grammar=grammar,
    #         **kwargs,
    #     )


@dataclass
class GeminiLLM(LLM):
    def __init__(self, config: GeminiConfig):
        super().__init__(config)
        self.config = config

        api_key = config["google_api_key"] if "google_api_key" in config else None
        api_key = os.getenv("google_api_key") if not api_key else api_key
        assert api_key, "No Google API key provided."

        self.client = genai.Client(api_key=api_key)

    def _count_tokens(self, text: str) -> int:
        return count_tokens_google(self.client, self.config["model_name"], text)

    def _generate(
        self,
        messages: str | List[ChatMessage],
        grammar: str | None = None,
        **kwargs,
    ) -> str:
        return generate_google(
            client=self.client,
            model_name=self.config["model_name"],
            messages=messages,
            **kwargs,
        )


def get_llm(config: LLMConfig) -> LLM:

    if "gguf_path" in config:
        return LlamaLLM(config)
    elif 'colab' in config:
        return GeminiColabLLM(config)
    else:
        return GeminiLLM(config)


class AI:
    def __init__(self, *configs: LLMConfig):
        self.create_models(*configs)
        self.setup_defaults()

    def create_models(self, *configs: LLMConfig):
        self.models = {c["name"]: get_llm(c) for c in configs}

    def setup_defaults(self):
        defaults = [
            m
            for m in self.models.values()
            if "default" in m.config and m.config["default"]
        ]
        embeds = [m for m in defaults if "embed" in m.config and m.config["embed"]]
        geners = [
            m for m in defaults if "embed" not in m.config or not m.config["embed"]
        ]
        assert len(embeds) <= 1, "Can only have one default E-LLM."
        assert len(geners) <= 1, "Can only have one default G-LLM."

        if embeds:
            self.models["default-embed"] = embeds[0]

        if geners:
            self.models["default-generate"] = geners[0]

    def get(self, name: str) -> LLM:
        assert name in self.models, f"No such LLM {name}"
        return self.models[name]

    @property
    def e(self) -> LLM:
        "Get the default embedding model."
        return self.get("default-embed")

    @property
    def g(self) -> LLM:
        "Get the default generator model."
        return self.get("default-generate")

    def model_names(self) -> List[str]:
        return list(self.models.keys())
    
    # def get_model_names(self):
    #     {k: v.config[os.name] for k, v in self.models.items()}


# T = TypeVar("T")

# @dataclass
# class BM25Index(Generic[T]):
#     items: Sequence[T]
#     stringify: Optional[Callable[[T], str]] = None

#     def __post_init__(self):
#         strings = [self.stringify(x) if self.stringify else str(x) for x in self.items]
#         tokenized = list(map(self.tokenize, strings))
#         self.bm25 = BM25Okapi(tokenized)

#     def find_with_score(
#         self,
#         query: str,
#         limit: Optional[int],
#         prefilter: Callable[[T], bool] = lambda _: True,
#     ) -> List[Tuple[T, float]]:
#         scores = self.bm25.get_scores(self.tokenize(query))
#         items_with_scores = [(self.items[i], s) for i, s in enumerate(scores)]
#         items_with_scores = [
#             (e, float(s)) for e, s in items_with_scores if prefilter(e)
#         ]
#         sorted_items_with_scores = sorted(
#             items_with_scores, key=lambda e: e[1], reverse=True
#         )
#         results = sorted_items_with_scores[:limit]
#         return results

#     def find(
#         self,
#         query: str,
#         limit: Optional[int],
#         prefilter: Callable[[T], bool] = lambda _: True,
#     ) -> List[T]:
#         items_and_scores = self.find_with_score(query, limit, prefilter)
#         items = [x for x, _ in items_and_scores]
#         return items

#     def tokenize(self, text: str) -> List[str]:
#         return text.lower().split()


W = TypeVar("W")

def rerank_by_query(
    query: np.ndarray,
    candidates: List[W],
    embeddings: np.ndarray,
    limit: int,
) -> List[W]:

    candidates_and_scores = get_similarity_scores(query, candidates, embeddings, limit)
    candidates_and_scores = [c for c, _ in candidates_and_scores]
    return candidates_and_scores


def get_similarity_scores(
    query: np.ndarray,
    candidates: List[W],
    embeddings: np.ndarray,
    limit: int,
) -> List[Tuple[W, float]]:

    assert len(candidates) == len(embeddings)
    # Normalize on copies: the input arrays may be views into buffers shared by the
    # caller (e.g. metrics.py iterates the rows of golden_embeddings and passes each
    # row as `query`, and build_index reuses the same chunk_embeddings for every
    # category). An in-place `/=` would mutate those buffers; we therefore normalize
    # on copies to avoid side effects outside the function.
    query = query / np.linalg.norm(query)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Compute matrix-vector product
    scores = embeddings @ query.T
    # Flatten the scores
    scores = scores.squeeze()
    scores = scores.tolist()

    # Ensure this is a list of floats
    scores = [scores] if isinstance(scores, float) else scores
    assert isinstance(scores, list) and all(isinstance(x, float) for x in scores)

    candidates_and_scores = sorted(zip(candidates, scores), key=lambda e: e[1], reverse=True)
    candidates_and_scores = candidates_and_scores[:limit]
    return candidates_and_scores

V = TypeVar('V', bound=Hashable)

class VectorStore(Generic[V]):

    '''
    A simple vector store for hashable items.
    '''

    def __init__(self, items: Sequence[V], embeddings: np.ndarray) -> None:
        assert len(items) == len(embeddings)
        self.item_to_embedding = {x : embeddings[i] for i, x in enumerate(items)}
        self.embeddings = embeddings

    def __getitem__(self, item: V) -> np.ndarray:
        return self.item_to_embedding[item]

    def getmany(self, units: Sequence[V]) -> np.ndarray:
        return np.asarray([self.item_to_embedding[u] for u in units])

    def __setitem__(self, item: V, value: np.ndarray):
        self.item_to_embedding[item] = value


@dataclass
class GeminiColabLLM(LLM):

    def __init__(self, config: GeminiColabConfig):
        super().__init__(config)
        self.config = config
    
    def _count_tokens(self, text: str) -> int:
        return len(text.split())

    def _generate(
        self,
        messages: str | List[ChatMessage],
        grammar: str | None = None,
        **kwargs,
    ) -> str:

        from google.colab import ai # pyright:ignore

        # model_name = self.config["model_name"] if 'model_name' in self.config else None
        # if model_name:
        #     response = ai.generate_text(messages, model_name=model_name)
        # else:
        #     response = ai.generate_text(messages)
        
        response = ai.generate_text(messages)
        return response
