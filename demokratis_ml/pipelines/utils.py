"""Utilities shared by multiple pipelines."""

import hashlib
import pathlib

import pandas as pd

from demokratis_ml.pipelines import blocks


def get_document_storage() -> blocks.ExtendedLocalFileSystem:
    """
    Return an instance of a Prefect block used for storing document files (usually PDFs).

    TODO: switch between local and S3 storage based on the environment
    """
    return blocks.ExtendedLocalFileSystem.load("local-document-storage")


def generate_path_in_document_storage(
    document: pd.Series,
    extension: str,
) -> pathlib.Path:
    """
    For a document processed by the ``preprocess_consultation_documents`` pipeline, generate a path under
    which the file would be stored in the document storage returned by :func:`get_document_storage`.

    :param document: A row from a DataFrame conforming to
    ``demokratis_ml.data.schemata.ConsultationDocumentMetadataSchemaV1``.
    :param extension: The file extension, e.g. "pdf".
    """  # noqa: D205
    url_hash = hashlib.sha1(document["document_source_url"].encode()).hexdigest()  # noqa: S324
    return pathlib.Path(str(document["consultation_id"])) / (
        f"{document['document_id']}-{document['document_language']}-{document['document_type']}"
        f"-{url_hash}.{extension}"
    )
