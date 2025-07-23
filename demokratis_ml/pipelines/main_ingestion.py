"""A composition of ingestion flows; see the :func:`main_ingestion` flow for details."""

import datetime

import prefect
import prefect.logging

from demokratis_ml.pipelines import (
    embed_documents,
    extract_document_features,
    predict_document_types,
    preprocess_consultation_documents,
)
from demokratis_ml.pipelines.lib import utils


@prefect.flow()
@utils.slack_status_report()
def main_ingestion(publish: bool, store_dataframes_remotely: bool, bootstrap_from_previous_output: bool) -> None:
    """Compose the main ingestion pipeline.

    - preprocess_consultation_documents.preprocess_data
        |-> extract_document_features.extract_document_features
        |-> embed_documents.embed_documents
    - predict_document_types.predict_document_types
    """
    logger = prefect.logging.get_run_logger()
    consultation_documents_file = preprocess_consultation_documents.preprocess_data(
        publish=publish,
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
    embeddings_file = embed_documents.embed_documents(
        consultation_documents_file=str(consultation_documents_file),
        store_dataframes_remotely=store_dataframes_remotely,
        bootstrap_from_previous_output=bootstrap_from_previous_output,
    )
    logger.info("embed_documents() -> %r", embeddings_file)

    data_files_version = datetime.datetime.now(tz=datetime.UTC).date()
    model_output_file = predict_document_types.predict_document_types(
        data_files_version=data_files_version,
        store_dataframes_remotely=store_dataframes_remotely,
    )
    logger.info("predict_document_types(%s) -> %r", data_files_version, model_output_file)
