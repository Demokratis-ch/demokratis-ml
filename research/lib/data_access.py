"""Utilities for selecting and loading input data."""

import functools
import logging
import operator
import os
import pathlib
from collections.abc import Iterable
from typing import Any

import boto3
import pandas as pd
import pandera
import pyarrow as pa
import pyarrow.compute

from demokratis_ml.data import schemata


def download_file_from_exoscale(remote_path: pathlib.Path, local_path: pathlib.Path) -> None:
    """Download an arbitrary file from our Exoscale Simple Object Storage bucket."""
    logger = logging.getLogger("download_file_from_exoscale")
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["EXOSCALE_SOS_ACCESS_KEY"],
        aws_secret_access_key=os.environ["EXOSCALE_SOS_SECRET_KEY"],
        endpoint_url=os.environ["EXOSCALE_SOS_ENDPOINT"],
    )
    bucket_name = os.environ["EXOSCALE_SOS_BUCKET_ML"]
    # remote_path = pathlib.Path("dataframes") / local_path.name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s from bucket %s to %s", remote_path, bucket_name, local_path)
    s3.download_file(bucket_name, str(remote_path), local_path)


def ensure_dataframe_is_available(local_path: pathlib.Path) -> None:
    """Download a dataframe from our Exoscale Simple Object Storage bucket if it is not already available locally."""
    logger = logging.getLogger("ensure_dataframe_is_available")
    if local_path.exists():
        logger.info("File %s already exists locally.", local_path)
        return
    download_file_from_exoscale(pathlib.Path("dataframes") / local_path.name, local_path)


@pandera.check_types
def load_consultation_documents(  # noqa: PLR0913
    input_file: pathlib.Path,
    *,
    only_document_sources: Iterable[str] | None = None,
    only_languages: Iterable[str] | None = None,
    only_doc_types: Iterable[str] | None = None,
    starting_year: int | None = None,
    mlflow: Any = None,
) -> schemata.FullConsultationDocumentV1:
    """Load and filter consultation documents from a parquet file.

    ~~If an MLflow client is provided, the loaded dataset is logged as an input artifact.~~ (MLflow logging
    is currently disabled because it's slow.)
    """
    assert input_file.suffix == ".parquet", f"Expected a .parquet file, got {input_file}"
    filters = []
    if only_document_sources is not None:
        filters.append(pyarrow.compute.is_in(pyarrow.compute.field("document_source"), pa.array(only_document_sources)))
    if only_languages is not None:
        filters.append(pyarrow.compute.is_in(pyarrow.compute.field("document_language"), pa.array(only_languages)))
    if only_doc_types is not None:
        assert set(only_doc_types) - {None} <= schemata.DOCUMENT_TYPES, f"Unknown doc types: {only_doc_types}"
        filters.append(pyarrow.compute.is_in(pyarrow.compute.field("document_type"), pa.array(only_doc_types)))
    if starting_year is not None:
        filters.append(pyarrow.compute.year(pyarrow.compute.field("consultation_start_date")) >= starting_year)

    filters_composed = functools.reduce(operator.and_, filters) if filters else None
    df = pd.read_parquet(input_file, filters=filters_composed)
    # Drop legacy columns to ensure we're not relying on them any more.
    df = df.drop(columns=["document_id", "organisation_id", "latest_stored_file_id"])

    # Log to MLflow - this is disabled because it takes 40+ seconds :/
    # if mlflow is not None:
    #     name = input_file.name.replace(".parquet", "")
    #     dataset = mlflow.data.from_pandas(df, source=input_file, name=name)
    #     mlflow.log_input(dataset)
    return df
