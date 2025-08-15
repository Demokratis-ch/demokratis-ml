"""Reusable Prefect tasks for embedding texts."""

import datetime
import itertools
from collections.abc import Iterable

import numpy as np
import prefect
import prefect.cache_policies
import prefect.tasks

import demokratis_ml.data.embeddings


@prefect.task(
    # This is needed because otherwise Prefect will try to calculate a cache key from `embedding_model`,
    # which is not hashable.
    cache_policy=prefect.cache_policies.NONE,
)
def embed_texts(texts: list[str], embedding_model: demokratis_ml.data.embeddings.EmbeddingModel) -> np.ndarray:
    """Split the provided texts into batches, embed them, and return all embeddings as a single array."""
    futures = embed_batch.map(
        texts=itertools.batched(texts, embedding_model.max_batch_size),
        embedding_model=prefect.unmapped(embedding_model),
    )
    return np.vstack(futures.result())


@prefect.task(
    # Custom function is used so that we can correctly use the unhashable embedding_model parameter
    cache_key_fn=lambda context, parameters: prefect.tasks.task_input_hash(
        context, {"texts": parameters["texts"], "embedding_model_name": parameters["embedding_model"].model_name}
    ),
    cache_expiration=datetime.timedelta(days=7),
    retries=3,
    retry_delay_seconds=[10, 63, 127],
)
def embed_batch(texts: Iterable[str], embedding_model: demokratis_ml.data.embeddings.EmbeddingModel) -> np.ndarray:
    """Single batch embedding, wrapped in a task so that it can be retried if the API call fails."""
    tokens = list(map(embedding_model.tokenize, texts))
    return embedding_model.embed_batch(tokens)
