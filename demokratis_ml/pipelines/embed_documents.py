"""Prefect pipeline for embedding document texts; see the `embed_documents` flow for details."""

import datetime
import itertools
import pathlib
from collections.abc import Iterable

import numpy as np
import pandas as pd
import prefect
import prefect.cache_policies
import prefect.logging
import prefect.task_runners
import prefect.tasks

import demokratis_ml.data.embeddings
from demokratis_ml.pipelines import blocks, utils


@prefect.flow(
    # We're not running much in parallel here
    task_runner=prefect.task_runners.ThreadPoolTaskRunner(max_workers=4),
)
@utils.slack_status_report()
def embed_documents(
    consultation_documents_file: str,
    store_dataframes_remotely: bool,
    embedding_model_name: str = "openai/text-embedding-3-large",
    bootstrap_from_previous_output: bool = True,
    only_languages: Iterable[str] | None = ("de",),
) -> pathlib.Path:
    """
    Embed document contents (document_content_plain) and store the embeddings in a dataframe indexed by document_id.

    The beginnings of the documents are used for embedding, with content exceeding the maximum input length truncated.

    Embeddings are stored in a single column "embedding" containing the embeddings as numpy arrays.

    :param consultation_documents_file: An output of the :mod:`preprocess_consultation_documents` flow, containing
        the list of documents to process. The file name is relative to the file system used (selected with
        the ``store_dataframes_remotely`` parameter).
    :param store_dataframes_remotely: If true, read inputs from and store the resulting dataframe in
        Exoscale object storage.
    :param embedding_model_name: The name of the embedding model to use. Currently, only OpenAI models are supported.
        The name must start with "openai/" and be one of the models listed in
        https://platform.openai.com/docs/models (Embeddings).
    :param bootstrap_from_previous_output: If true, the latest existing embeddings dataframe (output of this flow)
        will be found and used as a "cache": the embeddings for the documents in the input file will be computed
        only for the documents that are not already present in the warmup dataframe.
        The resulting dataframe will contain all the documents from the warmup dataframe and the new documents.
    :param only_languages: If set, only documents in the specified languages will be processed. This is to
        save time and resources at a stage where we're only developing the models and don't cover all languages yet.
    """
    logger = prefect.logging.get_run_logger()
    output_dataframe_prefix = f"consultation-documents-embeddings-beginnings-{embedding_model_name.replace('/', '-')}"

    # Choose where to load source dataframes from and where to store the resulting dataframe
    fs_dataframe_storage = utils.get_dataframe_storage(store_dataframes_remotely)

    if bootstrap_from_previous_output:
        # Dispatch the task in parallel with the other read that follows.
        find_latest_output_dataframe_future = find_latest_output_dataframe.submit(
            fs_dataframe_storage, output_dataframe_prefix
        )
    else:
        find_latest_output_dataframe_future = None

    # Load the input dataframe (preprocessed documents)
    df_documents = utils.read_dataframe(
        pathlib.Path(consultation_documents_file),
        columns=["document_id", "document_language", "document_content_plain"],
        fs=fs_dataframe_storage,
    )
    # Filter by language
    if only_languages is not None:
        original_len = len(df_documents)
        only_languages = set(only_languages)
        df_documents = df_documents[df_documents["document_language"].isin(only_languages)]
        logger.info(
            "Filtering documents by languages=%r; keeping %d out of %d (%.1f%%)",
            only_languages,
            len(df_documents),
            original_len,
            len(df_documents) / original_len * 100,
        )
    # Drop documents with empty text (can't be embedded)
    missing_content = df_documents["document_content_plain"].str.strip() == ""
    if missing_content.any():
        logger.warning(
            "Empty document_content_plain for %d documents out of %d (%.1f%%)",
            missing_content.sum(),
            len(df_documents),
            missing_content.mean() * 100,
        )
        df_documents = df_documents[~missing_content]

    # Run the embedding
    if bootstrap_from_previous_output:
        assert find_latest_output_dataframe_future is not None
        df_bootstrap = find_latest_output_dataframe_future.result()
    else:
        df_bootstrap = pd.DataFrame()

    docs_index = df_documents["document_id"]
    df_documents_to_process = df_documents[~docs_index.isin(df_bootstrap.index)]
    # df_documents_to_process = df_documents_to_process.head(100)
    logger.info(
        "Processing %d documents (%d are already in the bootstrap dataframe)",
        len(df_documents_to_process),
        len(df_bootstrap),
    )
    if df_documents_to_process.empty:
        embeddings = []
    else:
        embedding_model = demokratis_ml.data.embeddings.create_embedding_model(
            embedding_model_name,
            client=blocks.OpenAICredentials.load("openai-credentials").get_client(),
        )
        embeddings = embed_texts(df_documents_to_process["document_content_plain"].tolist(), embedding_model)
        logger.info("Newly computed embeddings shape: %s", embeddings.shape)

    df = pd.DataFrame({"embedding": list(embeddings)}, index=df_documents_to_process["document_id"])
    df = pd.concat([df_bootstrap, df], axis=0)
    df.index.name = "document_id"
    assert not df.index.duplicated().any(), "DataFrame index contains duplicates"

    # Store the dataframe
    output_path, _ = utils.store_dataframe(df, output_dataframe_prefix, fs_dataframe_storage)
    return output_path


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
    retry_delay_seconds=[10, 60, 60],
)
def embed_batch(texts: Iterable[str], embedding_model: demokratis_ml.data.embeddings.EmbeddingModel) -> np.ndarray:
    """Single batch embedding, wrapped in a task so that it can be retried if the API call fails."""
    tokens = list(map(embedding_model.tokenize, texts))
    return embedding_model.embed_batch(tokens)


@prefect.task(
    cache_policy=prefect.cache_policies.TASK_SOURCE,
    cache_expiration=datetime.timedelta(hours=1),
)
def find_latest_output_dataframe(
    fs_dataframe_storage: blocks.ExtendedFileSystemType, output_dataframe_prefix: str
) -> pd.DataFrame:
    """Find the latest output of this flow and return the dataframe to be used as a bootstrap cache."""
    logger = prefect.logging.get_run_logger()
    latest_path = utils.find_latest_dataframe(output_dataframe_prefix, fs_dataframe_storage)
    logger.info("Loading latest output from %r", latest_path)
    latest_df = utils.read_dataframe(latest_path, columns=None, fs=fs_dataframe_storage)
    return latest_df


if __name__ == "__main__":
    import sys

    consultation_documents_file = sys.argv[1]
    output_path = embed_documents(
        consultation_documents_file=consultation_documents_file,
        store_dataframes_remotely=False,
        bootstrap_from_previous_output=False,
    )
    print(output_path)
