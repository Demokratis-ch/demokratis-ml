"""Prefect pipeline for preprocessing consultation documents; see the `preprocess_data` flow for details."""

import datetime
import functools
import io
import os
import pathlib
import re
import sys
from typing import cast

import httpx
import huggingface_hub
import lingua
import numpy as np
import pandas as pd
import pandas.api.typing
import pandera
import prefect
import prefect.cache_policies
import prefect.filesystems
import prefect.logging
import prefect.task_runners

from demokratis_ml.data import schemata
from demokratis_ml.pipelines import blocks, pdf_extraction, utils

OUTPUT_DATAFRAME_PREFIX = "consultation-documents-preprocessed"


@prefect.flow(
    # Max concurrency must be set, otherwise document extraction blows up on too many open files.
    task_runner=prefect.task_runners.ThreadPoolTaskRunner(max_workers=(os.cpu_count() or 1) * 2),
)
@pandera.check_types
def preprocess_data(
    publish: bool,
    store_dataframes_remotely: bool,
    bootstrap_extracted_content: bool = True,
) -> pathlib.Path:
    """Retrieve all available consultation documents from the Demokratis API and preprocess them.

    Main steps:
    - Load metadata and contents of consultation documents.
    - Detect the language of cantonal documents (the Demokratis platform is unreliable at detecting them).
    - Optionally bootstrap missing document content by finding a previously preprocessed dataframe and taking
      document_content_plain from there.
    - Extract any missing plain text from Fedlex documents (not provided by the API).
    - Store the resulting dataframe in a Parquet file.

    The dataframe is either stored remotely in Exoscale object storage (S3-compatible) or on the local filesystem,
    depending on the ``store_dataframes_remotely`` parameter.

    Only documents with non-empty content are kept in the final dataframe.

    :param publish: If true, upload the resulting dataframe to our public HuggingFace dataset repository.
    :param store_dataframes_remotely: If true, store the resulting dataframe in Exoscale object storage.
    :param bootstrap_extracted_content: If true, try to find a previously extracted dataframe and use the
        document_content_plain from there to fill in missing content for Fedlex documents.
    """
    # Choose where to store the resulting dataframe
    fs_dataframe_storage = utils.get_dataframe_storage(store_dataframes_remotely)

    # Run the actual preprocessing
    df = create_preprocessed_dataframe(bootstrap_extracted_content=bootstrap_extracted_content)

    # Store the dataframe
    output_path, df_serialized = utils.store_dataframe(df, OUTPUT_DATAFRAME_PREFIX, fs_dataframe_storage)

    # Upload to HuggingFace if requested
    if publish:
        upload_to_huggingface(
            repository_id="demokratis/consultation-documents",
            # No need to include the date in the filename, as the HF dataset is a Git repository.
            file_name=f"{OUTPUT_DATAFRAME_PREFIX}.parquet",
            data=df_serialized,
        )

    return output_path


@prefect.task()
def upload_to_huggingface(repository_id: str, file_name: str, data: bytes) -> None:
    """Upload our resulting preprocessed dataframe to our HuggingFace dataset repository."""
    logger = prefect.logging.get_run_logger()
    logger.info("Uploading to HuggingFace repository %s, file %s", repository_id, file_name)
    hf_token = blocks.HuggingFaceDatasetUploadCredentials.load("huggingface-dataset-upload-credentials").token
    hf_api = huggingface_hub.HfApi(token=hf_token.get_secret_value())
    hf_api.upload_file(
        repo_id=repository_id,
        path_in_repo=file_name,
        path_or_fileobj=data,
        repo_type="dataset",
    )


@prefect.task
@pandera.check_types
def create_preprocessed_dataframe(bootstrap_extracted_content: bool) -> schemata.FullConsultationDocumentV1:
    """Retrieve all available consultation documents from the Demokratis API and preprocess them.

    Main steps:
    - Load metadata and contents of consultation documents.
    - Detect the language of cantonal documents (the Demokratis platform is unreliable at detecting them).
    - Optionally bootstrap missing document content by finding a previously preprocessed dataframe and taking
      document_content_plain from there.
    - Extract any missing plain text from Fedlex documents (not provided by the API).

    Only documents with non-empty content are kept in the final dataframe.

    :param bootstrap_extracted_content: If true, try to find a previously extracted dataframe and use the
        document_content_plain from there to fill in missing content for Fedlex documents.
    """
    logger = prefect.logging.get_run_logger()

    if bootstrap_extracted_content:
        # Dispatch the task in parallel with the API requests that follow.
        previously_extracted_content_future = find_previously_extracted_content.submit()
    else:
        previously_extracted_content_future = None

    # Load raw data from Demokratis API. Takes a few minutes but we do it sequentially to be nice to the API.
    metadata = load_consultation_document_metadata()
    contents = load_consultation_document_contents()
    stored_files = load_consultation_document_stored_files()
    df = metadata.join(contents, on="document_id")
    df = df.join(stored_files, on="latest_stored_file_id")

    # Language is unreliable for cantonal documents => detect it.
    # This assumes that we do have the content of cantonal documents retrieved from the API.
    openparldata_index = df["document_source"] == "openparldata"
    detected_languages = detect_document_language(df.loc[openparldata_index])
    # TODO: log the % difference between detected_languages and df.loc[openparldata_index, "document_language"]
    df.loc[openparldata_index, "document_language"] = detected_languages

    # Extract text from Fedlex documents
    missing_content_index = df["document_source"] == "fedlex"
    assert (
        df.loc[missing_content_index, "document_content_plain"].isna().all()
    ), "Fedlex documents should not have content yet"
    logger.info("Need content for %d Fedlex documents", missing_content_index.sum())
    if bootstrap_extracted_content:
        assert previously_extracted_content_future is not None
        previously_extracted_content = previously_extracted_content_future.result()
        usable_content = df.loc[missing_content_index].join(
            previously_extracted_content, on="document_id", rsuffix="_previous"
        )["document_content_plain_previous"]
        df.loc[usable_content.index, "document_content_plain"] = usable_content
        missing_content_index &= df["document_content_plain"].isna()
        logger.info("After bootstrapping, %d documents still have missing content", missing_content_index.sum())

    extracted_content = extract_document_content(df.loc[missing_content_index])
    df.loc[missing_content_index, "document_content_plain"] = extracted_content

    # Drop documents that still don't have any content
    missing_content = df[df["document_content_plain"].isna()]
    if len(missing_content) > 0:
        logger.warning(
            "Missing content for %d documents (%.1f%%):\n%r",
            len(missing_content),
            100 * len(missing_content) / len(df),
            missing_content.groupby("document_source", observed=False).size(),
        )
        df = df[~df["document_content_plain"].isna()]

    return cast(schemata.FullConsultationDocumentV1, df)  # Pandera validation makes this cast safe


@prefect.task(
    task_run_name="demokratis_api_request({version}/{endpoint})",
    cache_policy=prefect.cache_policies.TASK_SOURCE + prefect.cache_policies.INPUTS,
    cache_expiration=datetime.timedelta(hours=1),
)
def demokratis_api_request(endpoint: str, version: str = "v0.1", timeout: float = 180.0) -> str:
    """Make an authenticated request to the Demokratis API and return the JSON response."""
    credentials = blocks.DemokratisAPICredentials.load("demokratis-api-credentials")
    response = httpx.get(
        f"https://www.demokratis.ch/api/{version}/{endpoint}",
        auth=(credentials.username, credentials.password.get_secret_value()),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.text


@prefect.task
@utils.print_validation_failure_cases()
@pandera.check_output(schemata.ConsultationDocumentMetadataSchemaV1.to_schema(), lazy=True)
def load_consultation_document_metadata() -> pd.DataFrame:
    """Load the metadata of all consultation documents from the Demokratis API.

    Make sure the dataframe types match the schema and drop a few known invalid rows.
    """
    logger = prefect.logging.get_run_logger()
    raw = demokratis_api_request("documents-metadata")
    df = pd.read_json(io.StringIO(raw), dtype={"latest_stored_file_id": "Int64"})
    # Cast nested datetime strings to actual datetimes
    df["consultation_internal_tags"] = df["consultation_internal_tags"].map(
        lambda tags: [{**tag, "created_at": pd.to_datetime(tag["created_at"])} for tag in (tags or [])]
    )

    # === Document source is not provided in the API response but we can infer it from the political body, for now.
    index_federal = df["political_body"] == schemata.FEDERAL_CODE
    df.loc[index_federal, "document_source"] = "fedlex"
    df.loc[~index_federal, "document_source"] = "openparldata"

    # === Infer consultation_topics_label_source from `document_source` and `consultation_internal_tags`.
    topic_review_times = df["consultation_internal_tags"].map(
        functools.partial(_find_internal_tag_date, tag_name="topics_reviewed")
    )
    column = "consultation_topics_label_source"
    manual_index = ~topic_review_times.isna()
    df.loc[manual_index, column] = "manual"
    df.loc[~manual_index & (df["document_source"] == "openparldata"), column] = "openparldata"
    df.loc[~manual_index & (df["document_source"] == "fedlex"), column] = "organisation_rule"
    df[column] = df[column].astype("category")
    logger.info("Topics label source (documents):\n%r", df[column].value_counts())
    logger.info(
        "Topics label source (consultations):\n%r",
        df.groupby("consultation_id").agg({column: "first"}).value_counts(),
    )

    # === Remove document_type labels that are not guaranteed to be correct.
    document_type_review_times = df["consultation_internal_tags"].map(
        functools.partial(_find_internal_tag_date, tag_name="document_types_reviewed")
    )
    df.loc[
        # OpenParlData documents...
        (df["document_source"] == "openparldata")
        # ...that haven't been reviewed...
        & document_type_review_times.isna()
        # ...and have the type that's automatically assigned to every OpenParlData document by default...
        & (df["document_type"] == "VARIOUS_TEXT"),
        # ...should be considered unlabelled:
        "document_type",
    ] = None
    # (In other words: the VARIOUS_TEXT label is only trusted for OpenParlDocuments if the consultation
    # has been marked with the 'document_types_reviewed' internal tag.)
    logger.info(
        "OpenParlData document types:\n%r",
        df.loc[df["document_source"] == "openparldata", "document_type"].value_counts(dropna=False),
    )

    # === Cast to the correct types
    for time_column in ("consultation_start_date", "consultation_end_date", "document_publication_date"):
        df[time_column] = pd.to_datetime(df[time_column])
    for category_column in ("political_body", "document_source", "document_type", "document_language"):
        df[category_column] = df[category_column].astype("category")

    # === Format topics
    topic_separator = re.compile(r"\s*,\s*")

    def topic_mapper(csv: str | None) -> np.ndarray:
        if csv is None:
            csv = ""
        # Split and drop empty strings
        topics = [t_striped for t in topic_separator.split(csv) if (t_striped := t.strip())]
        # Sort to ensure consistent order
        topics = sorted(topics)
        if not topics:
            # Make sure that even the empty array has the correct unicode dtype. Important for Parquet serialisation.
            return np.array([], dtype="U")
        return np.array(topics)

    df["consultation_topics"] = df["consultation_topics"].map(topic_mapper)

    # === Drop known invalid data
    missing_start_date = df["consultation_start_date"].isna()
    if len(missing := df[missing_start_date]) > 13:  # noqa: PLR2004
        logger.warning("Dropping %d documents with missing consultation start date: %r", len(missing), missing)
    df = df[~missing_start_date]
    missing_url = df["document_source_url"] == "#"  # Some OpenParlData documents have this placeholder instead of a URL
    if len(missing := df[missing_url]) > 0:
        logger.warning("Dropping %d documents with missing URL: %r", len(missing), missing)
    df = df[~missing_url]
    # There are a few documents "published" in "1970" (= clearly an invalid timestamp).
    invalid_publication_date = df["document_publication_date"] < df["consultation_start_date"].min()
    if len(invalid := df[invalid_publication_date]) > 0:
        logger.warning(
            "Erasing %d invalid publication dates:\n%r",
            len(invalid),
            invalid[["document_id", "document_source", "document_publication_date"]],
        )
        df.loc[invalid_publication_date, "document_publication_date"] = pd.NaT

    return df


def _find_internal_tag_date(
    internal_tags: list[schemata.ConsultationInternalTag], tag_name: str
) -> pd.Timestamp | pandas.api.typing.NaTType:
    """Find the 'created_at' date of a particular internal tag in the list of internal tags."""
    try:
        tag_object = next(tag for tag in internal_tags if tag["name"] == tag_name)
    except StopIteration:
        return pd.NaT
    return tag_object["created_at"]


@prefect.task
def load_consultation_document_contents() -> pd.Series:
    """Load the content of consultation documents from the Demokratis API.

    Returns a series indexed by document ID.
    Not all documents are available (typically only those from openparldata).
    """
    raw = demokratis_api_request("documents-content", timeout=180.0)
    df = pd.read_json(io.StringIO(raw))
    assert df.columns.tolist() == ["document_id", "document_content"]
    assert df["document_id"].is_unique
    df = df.rename(columns={"document_content": "document_content_plain"})
    series = df.set_index("document_id")["document_content_plain"]
    return series


@prefect.task
def load_consultation_document_stored_files() -> pd.DataFrame:
    """Load references to original stored files from the Demokratis API.

    These files are mirrored from the original sources (federal and cantonal websites) and stored
    in Demokratis object storage.

    Returns a dataframe indexed by stored file ID.
    """
    logger = prefect.logging.get_run_logger()
    raw = demokratis_api_request("stored-files", timeout=180.0)
    df = pd.read_json(io.StringIO(raw))
    df = df.loc[df["type"] == "consultation_document"]
    logger.info("Loaded %d stored files with fields: %r", len(df), df.columns.tolist())
    assert {"id", "path", "size", "mime_type", "file_hash", "file_name"} <= set(df.columns), repr(df.columns)
    df = df.rename(
        columns={
            "id": "stored_file_id",
            "path": "stored_file_path",
            "mime_type": "stored_file_mime_type",
            "file_hash": "stored_file_hash",
        }
    )
    assert df["stored_file_id"].is_unique
    df = df.set_index("stored_file_id")
    # These columns will be joined to the metadata dataframe to create a full dataframe (together with doc contents).
    return df[["stored_file_path", "stored_file_mime_type", "stored_file_hash"]]


@prefect.task
@utils.print_validation_failure_cases()
@pandera.check_input(schemata.ConsultationDocumentMetadataSchemaV1.to_schema())
def detect_document_language(df: pd.DataFrame) -> pd.Series:
    """Detect the language of the document from its content.

    For cantonal documents, the language in the metadata is not reliable.
    We use a language detector instead.
    """
    logger = prefect.logging.get_run_logger()
    language_detector = lingua.LanguageDetectorBuilder.from_languages(
        lingua.Language.FRENCH,
        lingua.Language.GERMAN,
        lingua.Language.ITALIAN,
    ).build()
    # Add the document name to help when the content is too short.
    # Take only the first 1000 characters of the content to speed up the detection.
    samples = df["document_title"] + "\n" + df["document_content_plain"].fillna("").str.slice(0, 1000)
    languages = samples.map(lambda text: language_detector.detect_language_of(text).iso_code_639_1.name.lower())
    languages = languages.astype("category")
    logger.info("Detected languages:\n%r", languages.value_counts())
    return languages


@prefect.task(
    cache_policy=prefect.cache_policies.TASK_SOURCE,
    cache_expiration=datetime.timedelta(hours=1),
)
def find_previously_extracted_content() -> pd.Series:
    """Find the latest output of this flow and return "document_content_plain" for all extracted documents.

    Useful as a warm-up cache for the document content extraction task.
    """
    logger = prefect.logging.get_run_logger()
    fs_dataframe_storage = blocks.ExtendedRemoteFileSystem.load("remote-dataframe-storage")
    latest_path = utils.find_latest_dataframe(OUTPUT_DATAFRAME_PREFIX, fs_dataframe_storage)
    logger.info("Loading latest extracted content from %r", latest_path)
    latest_df = utils.read_dataframe(
        latest_path, columns=["document_id", "document_content_plain"], fs=fs_dataframe_storage
    )
    latest_df = latest_df.dropna()
    return latest_df.set_index("document_id")["document_content_plain"]


@prefect.task
@utils.print_validation_failure_cases()
@pandera.check_types
def extract_document_content(df: schemata.ConsultationDocumentMetadataV1) -> pd.Series:
    """For each document in the dataframe, download the remote PDF file and extract the plain text.

    Local filesystem is used as a cache for the downloaded files and extracted text, so only new
    documents are downloaded and extracted.

    :return: Series with the extracted texts, using the same index as the input dataframe.
    """
    logger = prefect.logging.get_run_logger()

    # Remove missing files
    missing_files = df["stored_file_path"].isna()
    if missing_files_count := int(missing_files.sum()):
        logger.warning(
            "Missing stored files for %d documents (%.1f%%)",
            missing_files_count,
            100 * missing_files_count / len(df),
        )
    df = df.loc[~missing_files]

    # Remove non-PDF files (we can't extract them yet)
    non_pdf_files = df["stored_file_mime_type"] != "application/pdf"
    if non_pdf_files_count := int(non_pdf_files.sum()):
        logger.warning(
            "Ignoring %d non-PDF files (%.1f%%):\n%r",
            non_pdf_files_count,
            100 * non_pdf_files_count / len(df),
            df.loc[non_pdf_files, "stored_file_mime_type"].value_counts(),
        )
    df = df.loc[~non_pdf_files]

    # Extract text from all PDFs that exist
    logger.info("Extracting text from %d PDFs", len(df))
    content = extract_text_from_pdf.map(df["stored_file_path"].tolist()).result()
    return pd.Series(content, index=df.index)


@prefect.task(task_run_name="extract_text_from_pdf({stored_path_pdf})")
def extract_text_from_pdf(stored_path_pdf: pathlib.Path) -> str | None:
    """Extract text from a PDF file in platform file storage and return it."""
    logger = prefect.logging.get_run_logger()
    fs_platform_storage = prefect.filesystems.RemoteFileSystem.load("platform-file-storage")
    try:
        file_data = fs_platform_storage.read_path(stored_path_pdf)
        text = pdf_extraction.extract_text_from_pdf(file_data)
    except (FileNotFoundError, pdf_extraction.PDFExtractionError):
        logger.exception("Error extracting text from PDF %r", stored_path_pdf)
        return None
    if text:
        logger.info("Extracted %.1fkB", len(text) / 1024)
    else:
        logger.warning("Empty text extracted from PDF %r", stored_path_pdf)
    return text


if __name__ == "__main__":
    publish = len(sys.argv) > 1 and sys.argv[1] == "--publish"
    output_path = preprocess_data(publish=publish, store_dataframes_remotely=publish)
    print(output_path)
