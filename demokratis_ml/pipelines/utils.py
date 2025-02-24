"""Utilities shared by multiple pipelines."""

import functools
import hashlib
import pathlib
from collections.abc import Callable
from typing import Any, TypeVar

import pandas as pd
import pandera.errors
import prefect.logging

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


F = TypeVar("F", bound=Callable[..., Any])


def print_validation_failure_cases() -> Callable[[F], F]:
    """In case the wrapped function raises a SchemaErrors exception, print the failure cases."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except pandera.errors.SchemaErrors as exc:
                logger = prefect.logging.get_run_logger()
                df_index = exc.failure_cases["index"]
                with pd.option_context(
                    "display.max_rows",
                    None,
                    "display.max_columns",
                    None,
                    "display.max_colwidth",
                    100,
                ):
                    hr = "=" * 80
                    logger.error(  # noqa: TRY400
                        "\n".join(
                            (
                                "",
                                hr,
                                "Schema errors and failure cases:",
                                repr(exc.failure_cases),
                                hr,
                                "DataFrame rows that failed validation:",
                                repr(exc.data.loc[df_index]),
                                hr,
                            )
                        )
                    )
                raise

        return wrapper

    return decorator
