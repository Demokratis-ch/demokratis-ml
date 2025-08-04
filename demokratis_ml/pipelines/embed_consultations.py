"""Prefect pipeline for embedding consultation attributes; see the `embed_consultations` flow for details."""

import pathlib
from collections.abc import Iterable

import pandas as pd
import prefect
import prefect.cache_policies
import prefect.logging
import prefect.task_runners
import prefect.tasks

import demokratis_ml.data.embeddings
from demokratis_ml.pipelines.lib import blocks, embeddings, utils

DEFAULT_EMBEDDING_MODEL_NAME = "openai/text-embedding-3-large"


def get_output_dataframe_prefix(embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME) -> str:
    """Generate a prefix for the output dataframe; the embedding model name is a part of it."""
    return f"consultation-attributes-embeddings-beginnings-{embedding_model_name.replace('/', '-')}"


@prefect.flow(
    # We're not running much in parallel here
    task_runner=prefect.task_runners.ThreadPoolTaskRunner(max_workers=4),
)
@utils.slack_status_report(":1234:")
def embed_consultations(  # noqa: PLR0913
    consultation_documents_file: str,
    store_dataframes_remotely: bool,
    embed_attributes: tuple[str, ...] = ("consultation_title", "consultation_description", "organisation_name"),
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    bootstrap_from_previous_output: bool = True,
    only_languages: Iterable[str] | None = ("de",),
) -> pathlib.Path:
    """
    Embed consultation attributes and store them in a dataframe indexed by (consultation_identifier, attribute_language, attribute_name).

    The beginnings of the attributes are used for embedding, with content exceeding the maximum input length truncated.

    Embeddings are stored in a single column "embedding" containing the embeddings as numpy arrays.
    Original attribute texts are stored in the "text" column for convenience.

    :param consultation_documents_file: An output of the :mod:`preprocess_consultation_documents` flow, containing
        the list of documents from which consultation attributes will be read (after grouping
        by `consultation_identifier`).
        The file name is relative to the file system used (selected with the ``store_dataframes_remotely`` parameter).
    :param store_dataframes_remotely: If true, read inputs from and store the resulting dataframe in
        Exoscale object storage.
    :param embed_attributes: Consultation attributes to embed.
    :param embedding_model_name: The name of the embedding model to use. Currently, only OpenAI models are supported.
        The name must start with "openai/" and be one of the models listed in
        https://platform.openai.com/docs/models (Embeddings).
    :param bootstrap_from_previous_output: If true, the latest existing embeddings dataframe (output of this flow)
        will be found and used as a "cache": the embeddings for the consultations in the input file will be computed
        only for the consultations that are not already present in the warmup dataframe.
        The resulting dataframe will contain all the consultations from the warmup dataframe and the new consultations.
    :param only_languages: If set, only consultations in the specified languages will be processed. This is to
        save time and resources at a stage where we're only developing the models and don't cover all languages yet.
        Filtering is done by the `document_language` column, before documents are grouped by consultation.
    """  # noqa: E501
    logger = prefect.logging.get_run_logger()
    output_dataframe_prefix = get_output_dataframe_prefix(embedding_model_name)

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
        columns=["consultation_identifier", "document_language", *embed_attributes],
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
    # Group by consultation
    df_consultations = (
        df_documents.rename(columns={"document_language": "attribute_language"})
        .groupby(["consultation_identifier", "attribute_language"], observed=True)
        .agg(dict.fromkeys(embed_attributes, "first"))
    )
    logger.info("Loaded %d consultations", len(df_consultations))
    df_attributes = df_consultations.reset_index().melt(
        id_vars=["consultation_identifier", "attribute_language"],
        value_vars=embed_attributes,
        var_name="attribute_name",
        value_name="attribute_value",
    )
    df_attributes = df_attributes.set_index(["consultation_identifier", "attribute_language", "attribute_name"])
    assert len(df_attributes.columns) == 1, "All other columns should be included in the index"

    # Remove empty attributes
    missing_content = df_attributes["attribute_value"].str.strip() == ""
    if missing_content.any():
        logger.warning(
            "Empty attributes: %d out of %d (%.1f%%)",
            missing_content.sum(),
            len(df_attributes),
            missing_content.mean() * 100,
        )
        df_attributes = df_attributes[~missing_content]

    # Run the embedding
    if bootstrap_from_previous_output:
        assert find_latest_output_dataframe_future is not None
        df_bootstrap = find_latest_output_dataframe_future.result()
    else:
        df_bootstrap = pd.DataFrame()

    df_attributes_to_process = df_attributes[~df_attributes.index.isin(df_bootstrap.index)]
    # df_attributes_to_process = df_attributes_to_process.head(10)  # testing
    logger.info(
        "Processing %d attributes (%d are already in the bootstrap dataframe)",
        len(df_attributes_to_process),
        len(df_bootstrap),
    )
    if df_attributes_to_process.empty:
        text_embeddings = []
    else:
        embedding_model = demokratis_ml.data.embeddings.create_embedding_model(
            embedding_model_name,
            client=blocks.OpenAICredentials.load("openai-credentials").get_client(),
        )
        text_embeddings = embeddings.embed_texts(df_attributes_to_process["attribute_value"].tolist(), embedding_model)
        logger.info("Newly computed embeddings shape: %s", text_embeddings.shape)

    df = pd.DataFrame(
        {"text": df_attributes_to_process["attribute_value"], "embedding": list(text_embeddings)},
        index=df_attributes_to_process.index,
    )
    df = pd.concat([df_bootstrap, df], axis=0)
    assert not df.index.duplicated().any(), "DataFrame index contains duplicates"

    # Store the dataframe
    output_path, _ = utils.store_dataframe(df, output_dataframe_prefix, fs_dataframe_storage)
    return output_path


@prefect.task()
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
    output_path = embed_consultations(
        consultation_documents_file=consultation_documents_file,
        store_dataframes_remotely=False,
        bootstrap_from_previous_output=True,
    )
    print(output_path)
