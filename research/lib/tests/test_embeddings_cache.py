import itertools

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from research.lib.embeddings_cache import EmbeddingsCache


@pytest.fixture
def embeddings_cache(tmp_path, mocker):
    c = EmbeddingsCache(tmp_path, "test_model", read_only=False)
    embedding_fn = mocker.Mock(return_value=np.array([0, 7]))
    c.get_embedding("already embedded", embedding_fn)
    return c


def test_get_embedding(embeddings_cache, mocker):
    embedding_fn = mocker.stub()
    embeddings_cache.get_embedding("test", embedding_fn)
    embeddings_cache.get_embedding("test", embedding_fn)
    embeddings_cache.get_embedding("test", embedding_fn)
    assert embedding_fn.call_count == 1


def test_save_and_load(embeddings_cache, mocker, tmp_path):
    embedding_fn = mocker.Mock(return_value=np.array([88, 99]))
    embeddings_cache.get_embedding("test", embedding_fn)
    assert embedding_fn.call_count == 1
    embeddings_cache.save()

    another_cache = EmbeddingsCache(tmp_path, "test_model", read_only=True)
    another_embedding_fn = mocker.stub()
    another_cache.get_embedding("test", another_embedding_fn)
    assert another_embedding_fn.call_count == 0


@pytest.mark.parametrize("texts", list(itertools.permutations(["test1", "test2", "already embedded"])), ids=repr)
@pytest.mark.parametrize("batch_size", [2, 3, 4])
def test_get_embeddings_batch(embeddings_cache, texts: list[str], batch_size: int):
    expected_embeddings = {
        "test1": np.array([1, 2]),
        "test2": np.array([3, 4]),
        "already embedded": np.array([0, 7]),
    }

    def embedding_fn(uncached_texts):
        # Can't simply use mocker and embedding_fn.assert_called_once_with(["test1", "test2"])
        # because the argument is mutable. Instead, we make the assertion here.
        # See also https://docs.python.org/3/library/unittest.mock-examples.html#coping-with-mutable-arguments
        assert uncached_texts == [text for text in texts if text != "already embedded"]
        return np.array([expected_embeddings[text] for text in uncached_texts])

    embeddings = embeddings_cache.get_embeddings_batch(texts, embedding_fn, batch_size)
    assert_array_equal(embeddings, np.array([expected_embeddings[text] for text in texts]))
