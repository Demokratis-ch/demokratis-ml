"""Simple wrappers for embedding models."""

import abc
from typing import Any, ClassVar

import numpy as np
import openai
import tiktoken


class EmbeddingModel(abc.ABC):
    """Abstract class for text embedding models: 3rd party APIs or local models."""

    model_name: str
    max_input_tokens: int
    max_batch_size: int

    @abc.abstractmethod
    def tokenize(self, text: str, truncate: bool = True) -> list[int]:
        """Tokenize the text using the model's tokenizer.

        :param truncate: If True, truncate the text to the maximum input length.
        """

    @abc.abstractmethod
    def embed_batch(self, inputs: list[str] | list[list[int]]) -> np.ndarray:
        """Get embeddings for a batch of texts/tokens."""


class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI embedding model wrapper."""

    _MAX_INPUT_TOKENS: ClassVar[dict[str, int]] = {
        "text-embedding-ada-002": 8192,
        "text-embedding-3-small": 8192,
        "text-embedding-3-large": 8192,
    }

    def __init__(self, model_name: str, client: openai.OpenAI | None = None) -> None:
        """Initialize the OpenAI embedding model wrapper.

        `max_input_tokens` and `max_batch_size` are set based on the chosen model.
        """
        assert model_name.startswith("openai/")
        self.model_name = model_name
        self._model_name_api = model_name[len("openai/") :]
        self.max_input_tokens = self._MAX_INPUT_TOKENS[self._model_name_api]
        # 300k is the maximum number of tokens for a single request
        self.max_batch_size = 300_000 // self.max_input_tokens
        self._tokenizer = tiktoken.encoding_for_model(self._model_name_api)
        self._client = openai.OpenAI() if client is None else client

    def tokenize(self, text: str, truncate: bool = True) -> list[int]:
        """Tokenize the text using tiktoken."""
        if truncate:
            # First shorten the text, to make the encoding faster
            probable_max_length = 10 * self.max_input_tokens
            text = text[:probable_max_length]
        tokens = self._tokenizer.encode(text)
        if truncate:
            tokens = tokens[: self.max_input_tokens]
        return tokens

    def embed_batch(self, inputs: list[str] | list[list[int]]) -> np.ndarray:
        """Call OpenAI API to get embeddings for a batch of texts/tokens."""
        assert len(inputs) <= self.max_batch_size
        response = self._client.embeddings.create(
            model=self._model_name_api,
            input=inputs,
            timeout=60,
        )
        return np.array([item.embedding for item in response.data])


def create_embedding_model(model_name: str, **kwargs: Any) -> EmbeddingModel:
    """Create embedding models with this factory function."""
    if model_name.startswith("openai/"):
        return OpenAIEmbeddingModel(model_name, **kwargs)
    raise NotImplementedError("Unsupported model name", model_name)
