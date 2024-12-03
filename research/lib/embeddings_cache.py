import hashlib
import logging
import pathlib
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pyarrow.parquet


class _NoTqdm:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def set_description(self, *_args, **_kwargs) -> None:
        pass

    def update(self, *_args, **_kwargs) -> None:
        pass

    def __enter__(self) -> "_NoTqdm":
        return self

    def __exit__(self, *_args) -> None:
        pass


class EmbeddingsCache:
    def __init__(self, root_cache_dir: pathlib.Path, embedding_model_name: str) -> None:
        self._file_path = root_cache_dir / f"{embedding_model_name.replace('/', '--')}.parquet"
        if self._file_path.exists():
            t0 = time.monotonic()
            table = pyarrow.parquet.read_table(self._file_path)
            self._cache = {key: table[key].to_numpy() for key in table.column_names}
            read_time = time.monotonic() - t0
            logging.info(
                "Loaded %d embeddings in %.1fs from cache at %s",
                len(self._cache),
                read_time,
                self._file_path,
            )
        else:
            self._cache = {}
            logging.warning("No cache found at %s", self._file_path)
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._cache)

    def _generate_cache_key(self, text_or_tokens: str | list[int]) -> str:
        return hashlib.md5(str(text_or_tokens).encode()).hexdigest()

    def get_embedding(
        self,
        text_or_tokens: str | list[int],
        embedding_function: Callable[[str | list[int]], np.ndarray],
    ) -> np.ndarray:
        cache_key = self._generate_cache_key(text_or_tokens)
        if cache_key in self._cache:
            self.hits += 1
        else:
            self.misses += 1
            self._cache[cache_key] = embedding_function(text_or_tokens)
        return self._cache[cache_key]

    # I don't know how to simplify this function at the moment.
    def get_embeddings_batch[InputType: (str, list[int])](
        self,
        inputs: list[InputType],
        batched_embedding_function: Callable[[list[InputType]], np.ndarray],
        batch_size: int,
        tqdm: Any | None = None,
    ) -> np.ndarray:
        if tqdm is None:
            tqdm = _NoTqdm

        resulting_embeddings = None
        batch_indices: list[int] = []
        batch_inputs: list[InputType] = []
        call_hits = 0
        call_misses = 0

        def initialize_resulting_embeddings(some_embedding: np.ndarray) -> None:
            nonlocal resulting_embeddings
            if resulting_embeddings is None:
                resulting_embeddings = np.zeros((len(inputs), some_embedding.shape[0]))

        def embed_batch() -> None:
            nonlocal resulting_embeddings
            batch_embeddings = batched_embedding_function(batch_inputs)
            initialize_resulting_embeddings(batch_embeddings[0])
            # Cache embeddings and insert them into the result
            for i, batch_input, batch_embedding in zip(batch_indices, batch_inputs, batch_embeddings, strict=False):
                self._cache[self._generate_cache_key(batch_input)] = batch_embedding
                resulting_embeddings[i] = batch_embedding
            # Reset batch
            batch_indices.clear()
            batch_inputs.clear()

        with tqdm(total=len(inputs)) as progress:
            for input_index, input_ in enumerate(inputs):
                progress.set_description(f"Embedding (cached={call_hits}, new={call_misses})")
                cache_key = self._generate_cache_key(input_)
                if cache_key in self._cache:
                    # Cached => insert into resulting embeddings
                    self.hits += 1
                    call_hits += 1
                    cached_embedding = self._cache[cache_key]
                    initialize_resulting_embeddings(cached_embedding)
                    resulting_embeddings[input_index] = cached_embedding
                    progress.update(1)
                    continue

                # Not cached yet => add to batch
                batch_indices.append(input_index)
                batch_inputs.append(input_)
                self.misses += 1
                call_misses += 1
                if len(batch_inputs) == batch_size:
                    # Batch is full => compute embeddings
                    embed_batch()
                    progress.update(batch_size)

        if batch_indices:
            embed_batch()

        return resulting_embeddings

    def save(self) -> None:
        t0 = time.monotonic()
        table = pyarrow.table({key: pyarrow.array(value) for key, value in self._cache.items()})
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        pyarrow.parquet.write_table(table, self._file_path, compression="snappy")
        write_time = time.monotonic() - t0
        logging.info("Saved %d embeddings in %.1fs to cache at %s", len(self._cache), write_time, self._file_path)
