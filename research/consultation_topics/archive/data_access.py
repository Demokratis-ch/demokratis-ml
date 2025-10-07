"""Utilities for selecting and loading input data."""

import functools
import operator
import pathlib
from collections.abc import Iterable
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute


def load_consultation_documents(  # noqa: PLR0913
    input_file: pathlib.Path,
    *,
    only_document_sources: Iterable[str] | None = None,
    only_languages: Iterable[str] | None = None,
    only_doc_types: Iterable[str] | None = None,
    starting_year: int | None = None,
    mlflow: Any = None,
) -> pd.DataFrame:
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
        filters.append(pyarrow.compute.is_in(pyarrow.compute.field("document_type"), pa.array(only_doc_types)))
    if starting_year is not None:
        filters.append(pyarrow.compute.year(pyarrow.compute.field("consultation_start_date")) >= starting_year)

    filters_composed = functools.reduce(operator.and_, filters) if filters else None
    df = pd.read_parquet(input_file, filters=filters_composed)

    # Log to MLflow - this is disabled because it takes 40+ seconds :/
    # if mlflow is not None:
    #     name = input_file.name.replace(".parquet", "")
    #     dataset = mlflow.data.from_pandas(df, source=input_file, name=name)
    #     mlflow.log_input(dataset)
    return df
