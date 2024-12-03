"""Prefect pipeline for preprocessing consultation documents; see the `preprocess_data` flow for details."""

import datetime
import functools
import hashlib
import pathlib
import re
import sys

import httpx
import lingua
import magic
import numpy as np
import pandas as pd
import pandera as pa
import prefect
import prefect.blocks.core
import prefect.cache_policies
import prefect.filesystems
import prefect.futures
import prefect.logging
import prefect.task_runners

from demokratis_ml.data import schemata
from demokratis_ml.pipelines import blocks, simple_pdf_extraction

CONSULTATION_TOPICS_LABEL_SOURCE_MANUAL_REVIEW_SINCE = pd.Timestamp("2024-08-20T00:00:00")
""" For consultations reviewed after this date, the topics are considered to be manually
reviewed and of the highest quality. """

OPENPARLDATA_DOCUMENT_TYPE_MANUAL_REVIEW_SINCE_START_DATE = pd.Timestamp("2024-11-01T00:00:00")
""" For OpenParlData consultations ingested into the platform after this date, we can trust the document type.
Before this date, the document type wasn't consistently reviewed and defaulted to VARIOUS_TEXT."""


@prefect.flow(
    # Max concurrency must be set, otherwise document extraction blows up on too many open files.
    task_runner=prefect.task_runners.ThreadPoolTaskRunner(max_workers=8),
)
@pa.check_output(schemata.FullConsultationDocumentSchemaV1.to_schema())
def preprocess_data(publish: bool) -> pd.DataFrame:
    """Retrieve all available consultation documents from the Demokratis API and preprocess them.

    Main steps:
    - Load metadata and contents of consultation documents.
    - Detect the language of cantonal documents (the Demokratis platform is unreliable at detecting them).
    - Download Fedlex documents and extract their plain text (not provided by the API).
    - Store the resulting dataframe in a Parquet file.

    Only documents with non-empty content are kept in the final dataframe.

    :param publish: If true, upload the resulting dataframe to our remote S3-like storage.
    """
    logger = prefect.logging.get_run_logger()

    # Load raw data
    metadata = load_consultation_document_metadata()
    contents = load_consultation_document_contents()
    df = metadata.join(contents, on="document_id")

    # Language is unreliable for cantonal documents => detect it.
    # This assumes that we do have the content of cantonal documents retrieved from the API.
    openparldata_index = df["document_source"] == "openparldata"
    detected_languages = detect_document_language(df.loc[openparldata_index])
    # TODO: log the % difference between detected_languages and df.loc[openparldata_index, "document_language"]
    df.loc[openparldata_index, "document_language"] = detected_languages
    # Remove document_type labels that are not guaranteed to be correct.
    df.loc[
        openparldata_index
        & (df["consultation_start_date"] < OPENPARLDATA_DOCUMENT_TYPE_MANUAL_REVIEW_SINCE_START_DATE),
        "document_type",
    ] = None

    # Download Fedlex documents and extract text
    fedlex_index = df["document_source"] == "fedlex"
    assert df.loc[fedlex_index, "document_content_plain"].isna().all(), "Fedlex documents should not have content yet"
    extracted_content = download_documents_and_extract_content(df.loc[fedlex_index])
    df.loc[fedlex_index, "document_content_plain"] = extracted_content

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

    # Store the dataframe
    logger.info("Serialising dataframe with %d rows to Parquet", len(df))
    data = df.to_parquet(compression="gzip")
    # TODO: switch between local and S3 storage based on the environment
    fs = blocks.ExtendedLocalFileSystem.load("local-dataframe-storage")
    now = datetime.datetime.now(tz=datetime.UTC)
    path = pathlib.Path(f"consultation-documents-preprocessed-{now:%Y-%m-%d}.parquet")
    if fs.path_exists(path):
        logger.warning("Overwriting existing file %r", path)
    logger.info("Writing %d rows, %.1f MiB to %r", len(df), len(data) / 1024**2, path)
    fs.write_path(path, data)

    if publish:
        remote_fs = prefect.filesystems.RemoteFileSystem.load("remote-dataframe-storage")
        logger.info("Uploading to %s/%s", remote_fs.basepath, path)
        remote_fs.write_path(str(path), data)

    return df


def _get_document_storage() -> blocks.ExtendedLocalFileSystem:
    # TODO: switch between local and S3 storage based on the environment
    return blocks.ExtendedLocalFileSystem.load("local-document-storage")


@prefect.task(
    task_run_name="demokratis_api_request({version}/{endpoint})",
    cache_policy=prefect.cache_policies.TASK_SOURCE + prefect.cache_policies.INPUTS,
    cache_expiration=datetime.timedelta(hours=1),
)
def demokratis_api_request(endpoint: str, version: str = "v0.1", timeout: float = 180.0) -> dict:
    """Make an authenticated request to the Demokratis API and return the JSON response."""
    credentials = blocks.DemokratisAPICredentials.load("demokratis-api-credentials")
    response = httpx.get(
        f"https://www.demokratis.ch/api/{version}/{endpoint}",
        auth=(credentials.username, credentials.password.get_secret_value()),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


@prefect.task
@pa.check_output(schemata.ConsultationDocumentMetadataSchemaV1.to_schema())
def load_consultation_document_metadata() -> pd.DataFrame:
    """Load the metadata of all consultation documents from the Demokratis API.

    Make sure the dataframe types match the schema and drop a few known invalid rows.
    """
    logger = prefect.logging.get_run_logger()
    metadata = demokratis_api_request("documents-metadata")
    df = pd.DataFrame(metadata)

    # Document source is not provided in the API response but we can infer
    # it from the political body, for now.
    index_federal = df["political_body"] == schemata.FEDERAL_CODE
    df.loc[index_federal, "document_source"] = "fedlex"
    df.loc[~index_federal, "document_source"] = "openparldata"

    # Infer consultation_topics_label_source from `document_source` and `consultation_reviewed_at`.
    column = "consultation_topics_label_source"
    manual_index = (
        pd.to_datetime(df["consultation_reviewed_at"]) >= CONSULTATION_TOPICS_LABEL_SOURCE_MANUAL_REVIEW_SINCE
    )
    df.loc[manual_index, column] = "manual"
    df.loc[~manual_index & (df["document_source"] == "openparldata"), column] = "openparldata"
    df.loc[~manual_index & (df["document_source"] == "fedlex"), column] = "organisation_rule"
    df[column] = df[column].astype("category")
    logger.info("Topics label source (documents):\n%r", df[column].value_counts())
    logger.info(
        "Topics label source (consultations):\n%r",
        df.groupby("consultation_id").agg({column: "first"}).value_counts(),
    )

    # Cast to the correct types
    for time_column in ("consultation_start_date", "consultation_end_date", "consultation_reviewed_at"):
        df[time_column] = pd.to_datetime(df[time_column])
    for category_column in ("political_body", "document_source", "document_type", "document_language"):
        df[category_column] = df[category_column].astype("category")
    # Format topics
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

    # Drop known invalid data
    df = _drop_known_documents_with_invalid_urls(df)
    missing_start_date = df["consultation_start_date"].isna()
    if len(missing := df[missing_start_date]) > 13:  # noqa: PLR2004
        logger.warning("Dropping %d consultations with missing start date: %r", len(missing), missing)
    df = df[~missing_start_date]

    return df


def _drop_known_documents_with_invalid_urls(df: pd.DataFrame) -> pd.DataFrame:
    # This function should not be necessary after https://github.com/Demokratis-ch/demokratis/issues/845 is fixed.
    weird_urls = ~(
        df["document_source_url"].str.startswith("https://") | df["document_source_url"].str.startswith("http://")
    )
    weird_document_ids = set(df[weird_urls]["document_id"])
    known_weird_document_ids = {
        40291,  # '#'
        40314,  # '#'
        41369,  # 'resolveuid/...'  (the next 17 are similar)
        41370,
        41371,
        41372,
        41373,
        41374,
        41375,
        41376,
        41377,
        41378,
        41379,
        41380,
        41381,
        41382,
        41383,
        41384,
        41385,
        41386,
        43096,  # 'mailto:...'
        47531,  # missing 'https://'
        52904,  # '#'
    }
    if unexpected := weird_document_ids - known_weird_document_ids:
        raise ValueError(
            "Unexpected document IDs with invalid URLs",
            df.loc[df["document_id"].isin(unexpected), ["document_id", "document_source_url"]],
        )
    return df[~weird_urls]


@prefect.task
def load_consultation_document_contents() -> pd.Series:
    """Load the content of consultation documents from the Demokratis API.

    Returns a series indexed by document ID.
    Not all documents are available (typically only those from openparldata).
    """
    contents = demokratis_api_request("documents-content", timeout=180.0)
    df = pd.DataFrame(contents)
    assert df.columns.tolist() == ["document_id", "document_content"]
    assert df["document_id"].is_unique
    df = df.rename(columns={"document_content": "document_content_plain"})
    series = df.set_index("document_id")["document_content_plain"]
    return series


@prefect.task
@pa.check_input(schemata.ConsultationDocumentMetadataSchemaV1.to_schema())
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


@prefect.task
@pa.check_input(schemata.ConsultationDocumentMetadataSchemaV1.to_schema())
def download_documents_and_extract_content(df: pd.DataFrame) -> pd.Series:
    """For each document in the dataframe, download the remote PDF file and extract the plain text.

    Local filesystem is used as a cache for the downloaded files and extracted text, so only new
    documents are downloaded and extracted.

    :return: Series with the extracted texts, using the same index as the input dataframe.
    """
    logger = prefect.logging.get_run_logger()

    source_urls = df["document_source_url"].tolist()
    # TODO: some of the documents are not actually PDFs. We shouldn't blindly use the .pdf extension.
    local_paths_pdf = df.apply(functools.partial(_generate_local_path, extension="pdf"), axis=1).tolist()
    local_paths_txt = df.apply(functools.partial(_generate_local_path, extension="txt"), axis=1).tolist()

    fs = _get_document_storage()

    # Download all missing PDFs and write them to the filesystem.
    # Extract text from all PDFs that exist on the filesystem but don't have extracted text yet.
    futures = []
    download_count = 0
    extract_count = 0
    for source_url, local_path_pdf, local_path_txt in zip(source_urls, local_paths_pdf, local_paths_txt, strict=True):
        # PDF doesn't exist => download & extract
        if not fs.path_exists(local_path_pdf):
            download_future = download_document.submit(source_url, local_path_pdf)
            futures.append(download_future)
            download_count += 1
            # Since we're downloading the PDF, we'll assume the text isn't extracted yet.
            extract_future = extract_text_from_pdf.submit(
                local_path_pdf,
                local_path_txt,
                wait_for=[download_future],
            )
            futures.append(extract_future)
            extract_count += 1
        # PDF exists but text isn't extracted yet => extract
        elif not fs.path_exists(local_path_txt):
            extract_future = extract_text_from_pdf.submit(local_path_pdf, local_path_txt)
            futures.append(extract_future)
            extract_count += 1

    logger.info("Downloading %d PDFs and extracting text from %d PDFs", download_count, extract_count)
    awaited_futures = prefect.futures.wait(futures)
    assert not awaited_futures.not_done

    # All futures are done, so we can read the extracted content from the filesystem.
    # TODO this is not very efficient as the vast majority of the documents have already been extracted
    # before. We should have a file with the already-extracted texts and use it as cache.
    logger.info("Reading extracted text from %d files", len(local_paths_txt))
    content = [
        fs.read_path(local_path_txt).decode() if fs.path_exists(local_path_txt) else None
        for local_path_txt in local_paths_txt
    ]
    return pd.Series(content, index=df.index)


def _generate_local_path(
    document: pd.Series,
    extension: str,
    suffix: str = "",
) -> pathlib.Path:
    url_hash = hashlib.sha1(document["document_source_url"].encode()).hexdigest()  # noqa: S324
    return pathlib.Path(str(document["consultation_id"])) / (
        f"{document['document_id']}-{document['document_language']}-{document['document_type']}"
        f"-{url_hash}{suffix}.{extension}"
    )


@prefect.task(
    task_run_name="download_document({document_url})",
)
def download_document(document_url: str, local_path: pathlib.Path) -> None:
    """Download a remote document and write it to a local file."""
    assert document_url.startswith(("http:", "https:"))
    response = httpx.get(document_url, timeout=120)
    response.raise_for_status()
    _get_document_storage().write_path(local_path, response.content)
    # The URL is in the task name so we don't need to repeat it in the log message.
    prefect.logging.get_run_logger().info("Downloaded %.1fkB to %r", len(response.content) / 1024.0, local_path)


@prefect.task(
    task_run_name="extract_text_from_pdf({local_path_pdf})",
)
def extract_text_from_pdf(local_path_pdf: pathlib.Path, local_path_txt: pathlib.Path) -> None:
    """Extract text from a PDF file and write it to a text file."""
    logger = prefect.logging.get_run_logger()
    fs = _get_document_storage()
    if not fs.path_exists(local_path_pdf):
        logger.error("PDF file does not exist, cannot extract")
        return

    file_data = fs.read_path(local_path_pdf)

    # TODO: instead of using the magic library here to find out we've downloaded a non-PDF and gave
    # it a .pdf extension, we should use the response MIME headers when downloading the file
    # to apply the correct extension.
    mime_type = magic.from_buffer(file_data[:2048], mime=True)
    if mime_type != "application/pdf":
        logger.error("File is not a PDF, cannot extract; MIME type: %s", mime_type)
        return

    try:
        content = simple_pdf_extraction.extract_text_from_pdf(file_data)
    except simple_pdf_extraction.PDFExtractionError:
        logger.exception("Error extracting text from PDF %r", local_path_pdf)
    else:
        if content:
            fs.write_path(local_path_txt, content.encode())
            logger.info("Extracted %.1fkB to %r", len(content), local_path_txt)
        else:
            logger.warning("Empty content extracted from PDF %r", local_path_pdf)


if __name__ == "__main__":
    publish = len(sys.argv) > 1 and sys.argv[1] == "--publish"
    df = preprocess_data(publish)
    print(df)
