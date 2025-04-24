"""A composition of ingestion flows; see the :func:`main_ingestion` flow for details."""

import prefect
import prefect.logging

from demokratis_ml.pipelines import extract_document_features, preprocess_consultation_documents


@prefect.flow()
def main_ingestion(publish: bool, store_dataframes_remotely: bool, bootstrap_from_previous_output: bool) -> None:
    """Compose the main ingestion pipeline.

    preprocess_consultation_documents.preprocess_data -> extract_document_features.extract_document_features
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
