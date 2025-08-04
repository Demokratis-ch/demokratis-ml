"""A composition of ingestion flows; see the :func:`main_ingestion` flow for details."""

import datetime

import prefect
import prefect.logging

from demokratis_ml.pipelines import (
    embed_consultations,
    embed_documents,
    extract_document_features,
    predict_consultation_topics,
    predict_document_types,
    preprocess_consultation_documents,
    publish_data,
)
from demokratis_ml.pipelines.lib import utils


@prefect.flow(retries=1)
@utils.slack_status_report()
def main_ingestion(publish: bool, store_dataframes_remotely: bool, bootstrap_from_previous_output: bool) -> None:
    """Compose the main ingestion pipeline.

    - preprocess_consultation_documents.preprocess_data
        |-> extract_document_features
        |-> embed_documents
        |-> embed_consultations
    - predict_document_types
    - predict_consultation_topics
    - (if publish=True) publish_data
    """
    logger = prefect.logging.get_run_logger()

    #
    # ================= Preprocessing =================
    #
    consultation_documents_file = preprocess_consultation_documents.preprocess_data(
        store_dataframes_remotely=store_dataframes_remotely,
        bootstrap_extracted_content=bootstrap_from_previous_output,
    )
    logger.info("preprocess_data() -> %r", consultation_documents_file)
    features_file = extract_document_features.extract_document_features(
        consultation_documents_file=str(consultation_documents_file),
        store_dataframes_remotely=store_dataframes_remotely,
        bootstrap_from_previous_output=bootstrap_from_previous_output,
    )
    logger.info("extract_document_features() -> %r", features_file)
    # It would be great to run extract & embed concurrently but Prefect makes that hard:
    # - https://github.com/PrefectHQ/prefect/issues/6689
    # - https://linen.prefect.io/t/18865322/what-is-the-best-practice-for-running-sub-flows-in-parallel-
    document_embedding_model_name = embed_documents.DEFAULT_EMBEDDING_MODEL_NAME
    document_embeddings_file = embed_documents.embed_documents(
        consultation_documents_file=str(consultation_documents_file),
        store_dataframes_remotely=store_dataframes_remotely,
        embedding_model_name=document_embedding_model_name,
        bootstrap_from_previous_output=bootstrap_from_previous_output,
    )
    logger.info("embed_documents() -> %r", document_embeddings_file)

    consultation_embedding_model_name = embed_consultations.DEFAULT_EMBEDDING_MODEL_NAME
    consultation_embeddings_file = embed_consultations.embed_consultations(
        consultation_documents_file=str(consultation_documents_file),
        store_dataframes_remotely=store_dataframes_remotely,
        embedding_model_name=consultation_embedding_model_name,
        bootstrap_from_previous_output=bootstrap_from_previous_output,
    )
    logger.info("embed_consultations() -> %r", consultation_embeddings_file)

    #
    # ================= Inference =================
    #
    data_files_version = datetime.datetime.now(tz=datetime.UTC).date()

    model_output_file = predict_document_types.predict_document_types(
        data_files_version=data_files_version,
        store_dataframes_remotely=store_dataframes_remotely,
        embedding_model_name=document_embedding_model_name,
    )
    logger.info("predict_document_types(%s) -> %r", data_files_version, model_output_file)

    model_output_file = predict_consultation_topics.predict_consultation_topics(
        data_files_version=data_files_version,
        store_dataframes_remotely=store_dataframes_remotely,
        embedding_model_name=consultation_embedding_model_name,
    )
    logger.info("predict_consultation_topics(%s) -> %r", data_files_version, model_output_file)

    if publish:
        files = {
            consultation_documents_file: f"{preprocess_consultation_documents.OUTPUT_DATAFRAME_PREFIX}.parquet",
            features_file: f"{extract_document_features.OUTPUT_DATAFRAME_PREFIX}.parquet",
            document_embeddings_file: f"{embed_documents.get_output_dataframe_prefix(document_embedding_model_name)}.parquet",  # noqa: E501
            consultation_embeddings_file: f"{embed_consultations.get_output_dataframe_prefix(consultation_embedding_model_name)}.parquet",  # noqa: E501
        }
        publish_data.publish_data(use_remote_storage=store_dataframes_remotely, files=files)
        logger.info("publish_data(%r)", files)


if __name__ == "__main__":
    # Command-line entry point for local testing.
    main_ingestion(
        publish=False,
        store_dataframes_remotely=False,
        bootstrap_from_previous_output=True,
    )
