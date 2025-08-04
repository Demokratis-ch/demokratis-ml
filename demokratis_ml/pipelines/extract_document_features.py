"""Prefect pipeline for extracting features from PDF documents; see the `extract_document_features` flow for details."""

import datetime
import os
import pathlib
from collections.abc import Iterable

import pandas as pd
import prefect
import prefect.cache_policies
import prefect.filesystems
import prefect.logging
import prefect.task_runners

from demokratis_ml.pipelines.lib import blocks, pdf_extraction, utils

OUTPUT_DATAFRAME_PREFIX = "consultation-documents-features"
MAX_PDF_PAGES_TO_PROCESS = 50


@prefect.flow(
    # It seems the extraction isn't CPU-bound so we can use a high number of threads per core.
    task_runner=prefect.task_runners.ThreadPoolTaskRunner(max_workers=(os.cpu_count() or 1) * 20),
)
@utils.slack_status_report(":triangular_ruler:")
def extract_document_features(
    consultation_documents_file: str,
    store_dataframes_remotely: bool,
    bootstrap_from_previous_output: bool = True,
    only_languages: Iterable[str] | None = ("de",),
) -> pathlib.Path:
    """
    Extract "visual" document features such as table counts and aspect ratios.

    Only PDF documents are supported. For the full list of features, see :class:`pdf_extraction.ExtendedPDFFeatures`.

    :param consultation_documents_file: An output of the :mod:`preprocess_consultation_documents` flow, containing
        the list of documents to process. The file name is relative to the file system used (selected with
        the ``store_dataframes_remotely`` parameter).
    :param store_dataframes_remotely: If true, read inputs from and store the resulting dataframe in
        Exoscale object storage.
    :param bootstrap_from_previous_output: If true, the latest existing features dataframe (output of this flow)
        will be found and used as a "cache": the features for the documents in the input file will be computed
        only for the documents that are not already present in the warmup dataframe.
        The resulting dataframe will contain all the documents from the warmup dataframe and the new documents.
    :param only_languages: If set, only documents in the specified languages will be processed. This is to
        save time and resources at a stage where we're only developing the models and don't cover all languages yet.
    """
    logger = prefect.logging.get_run_logger()

    # Choose where to load source dataframes from and where to store the resulting dataframe
    fs_dataframe_storage = utils.get_dataframe_storage(store_dataframes_remotely)

    if bootstrap_from_previous_output:
        # Dispatch the task in parallel with the other read that follows.
        find_latest_output_dataframe_future = find_latest_output_dataframe.submit(fs_dataframe_storage)
    else:
        find_latest_output_dataframe_future = None

    # Load the input dataframe (preprocessed documents)
    df_documents = utils.read_dataframe(
        pathlib.Path(consultation_documents_file),
        columns=["document_uuid", "document_language", "stored_file_hash", "stored_file_path", "stored_file_mime_type"],
        fs=fs_dataframe_storage,
    )
    # Drop those where the files are not available
    missing_paths = df_documents["stored_file_path"].isna()
    if missing_paths.any():
        logger.warning(
            "Missing stored_file_path for %d documents out of %d (%.1f%%)",
            missing_paths.sum(),
            len(df_documents),
            missing_paths.mean() * 100,
        )
        df_documents = df_documents[~missing_paths]

    # Drop non-PDF documents
    non_pdf = df_documents["stored_file_mime_type"] != "application/pdf"
    if non_pdf.any():
        logger.warning(
            "Skipping non-PDF documents: %d out of %d (%.1f%%)\n%r",
            non_pdf.sum(),
            len(df_documents),
            non_pdf.mean() * 100,
            df_documents[non_pdf].value_counts("stored_file_mime_type"),
        )
        df_documents = df_documents[~non_pdf]

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

    # Run the extraction
    if bootstrap_from_previous_output:
        assert find_latest_output_dataframe_future is not None
        df_bootstrap = find_latest_output_dataframe_future.result()
    else:
        df_bootstrap = pd.DataFrame()

    docs_index = pd.MultiIndex.from_frame(df_documents[["document_uuid", "stored_file_hash"]])
    df_documents_to_process = df_documents[~docs_index.isin(df_bootstrap.index)]
    logger.info(
        "Processing %d documents (%d are already in the bootstrap dataframe)",
        len(df_documents_to_process),
        len(df_bootstrap),
    )
    futures = extract_pdf_features.map(
        document_uuid=df_documents_to_process["document_uuid"],
        stored_file_hash=df_documents_to_process["stored_file_hash"],
        stored_file_path=df_documents_to_process["stored_file_path"],
    )
    pdf_features = futures.result()
    df = pd.DataFrame.from_dict(
        {
            (document_uuid, stored_file_hash): (features or {})
            for document_uuid, stored_file_hash, features in pdf_features
        },
        orient="index",
    )
    df = pd.concat([df_bootstrap, df], axis=0)
    df.index.names = ["document_uuid", "stored_file_hash"]
    assert not df.index.duplicated().any(), "DataFrame index contains duplicates"

    missing_index = df["count_pages"].isna()
    if missing_index.any():
        logger.warning("Missing data for %d documents", missing_index.sum())
        df = df[~missing_index]

    # Store the dataframe
    output_path, _ = utils.store_dataframe(df, OUTPUT_DATAFRAME_PREFIX, fs_dataframe_storage)
    return output_path


@prefect.task(
    task_run_name="extract_pdf_features({stored_file_path} [{document_uuid},{stored_file_hash}])",
    cache_policy=prefect.cache_policies.TASK_SOURCE + prefect.cache_policies.INPUTS,
    cache_expiration=datetime.timedelta(days=7),
    retries=3,
    retry_delay_seconds=[5, 10, 60],
)
def extract_pdf_features(
    document_uuid: int,
    stored_file_hash: str,
    stored_file_path: str,
) -> tuple[int, str, pdf_extraction.BasicPDFFeatures | pdf_extraction.ExtendedPDFFeatures | None]:
    """
    Retrieve a PDF file from platform file storage and extract features from it.

    document_uuid and stored_file_hash are returned as "primary keys" identifying the output.

    :returns: (document_uuid, stored_file_hash, features|None in case of an error)
    """
    fs_platform_storage = prefect.filesystems.RemoteFileSystem.load("platform-file-storage")
    data = fs_platform_storage.read_path(stored_file_path)
    try:
        features = pdf_extraction.extract_features_from_pdf(data, max_pages_to_process=MAX_PDF_PAGES_TO_PROCESS)
    except (FileNotFoundError, pdf_extraction.PDFExtractionError):
        logger = prefect.logging.get_run_logger()
        logger.exception("Error extracting text from PDF %r", stored_file_path)
        features = None
    return document_uuid, stored_file_hash, features


@prefect.task()
def find_latest_output_dataframe(fs_dataframe_storage: blocks.ExtendedFileSystemType) -> pd.DataFrame:
    """Find the latest output of this flow and return the dataframe to be used as a bootstrap cache."""
    logger = prefect.logging.get_run_logger()
    latest_path = utils.find_latest_dataframe(OUTPUT_DATAFRAME_PREFIX, fs_dataframe_storage)
    logger.info("Loading latest output from %r", latest_path)
    latest_df = utils.read_dataframe(latest_path, columns=None, fs=fs_dataframe_storage)
    return latest_df


if __name__ == "__main__":
    import sys

    consultation_documents_file = sys.argv[1]
    output_path = extract_document_features(
        consultation_documents_file=consultation_documents_file,
        store_dataframes_remotely=False,
        bootstrap_from_previous_output=True,
    )
    print(output_path)
