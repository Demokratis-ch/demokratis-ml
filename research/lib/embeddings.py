import abc
import contextlib
import functools
import pathlib
from collections.abc import Callable, Iterator
from typing import Any, ClassVar

import numpy as np
import openai
import tiktoken

from research.lib import embeddings_cache


class EmbeddingModel(abc.ABC):
    model_name: str
    max_input_tokens: int
    batch_size: int

    @abc.abstractmethod
    def tokenize(self, text: str, truncate: bool = True) -> list[int]: ...

    @abc.abstractmethod
    def embed_batch(self, inputs: list[str] | list[list[int]]) -> np.ndarray: ...


class OpenAIEmbeddingModel(EmbeddingModel):
    MAX_INPUT_TOKENS: ClassVar[dict[str, int]] = {
        "text-embedding-ada-002": 8192,
        "text-embedding-3-small": 8192,
        "text-embedding-3-large": 8192,
    }

    def __init__(self, model_name: str):
        assert model_name.startswith("openai/")
        self.model_name = model_name
        self._model_name_api = model_name[len("openai/") :]
        self.max_input_tokens = self.MAX_INPUT_TOKENS[self._model_name_api]
        self.batch_size = 100
        self._tokenizer = tiktoken.encoding_for_model(self._model_name_api)
        self._client = openai.OpenAI()

    def tokenize(self, text: str, truncate: bool = True) -> list[int]:
        if truncate:
            # First shorten the text, to make the encoding faster
            probable_max_length = 10 * self.max_input_tokens
            text = text[:probable_max_length]
        tokens = self._tokenizer.encode(text)
        if truncate:
            tokens = tokens[: self.max_input_tokens]
        return tokens

    def embed_batch(self, inputs: list[str] | list[list[int]]) -> np.ndarray:
        assert len(inputs) <= self.batch_size
        response = self._client.embeddings.create(
            model=self._model_name_api,
            input=inputs,
            timeout=60,
        )
        return np.array([item.embedding for item in response.data])


def create_embedding_model(model_name: str) -> EmbeddingModel:
    if model_name.startswith("openai/"):
        return OpenAIEmbeddingModel(model_name)
    raise NotImplementedError("Unsupported model name", model_name)


@contextlib.contextmanager
def use_cache(
    model: EmbeddingModel,
    cache_directory: pathlib.Path,
    tqdm: Any | None = None,
) -> Iterator[Callable[[list[str] | list[list[int]]], np.ndarray]]:
    """
    This context manager provides a function that can be used to get embeddings
    for a batch of inputs. The function will use :class:`EmbeddingsCache`.
    At the end of the context, the cache will be saved.

    Example::

        embedding_model = embeddings.create_embedding_model("some/model")
        my_tokens = df["text"].map(embedding_model.tokenize)

        with embeddings.use_cache(embedding_model) as get_embeddings:
            my_embeddings = get_embeddings(my_tokens)
    """
    cache = embeddings_cache.EmbeddingsCache(cache_directory, model.model_name)
    yield functools.partial(
        cache.get_embeddings_batch,
        batched_embedding_function=model.embed_batch,
        batch_size=model.batch_size,
        tqdm=tqdm,
    )
    cache.save()
