"""Helper functions for loading data, mainly through DuckDB."""

import datetime
from collections.abc import Iterable

import duckdb
import pandas as pd
import pandera.pandas as pa

from demokratis_ml.data import schemata


def filter_documents(
    rel_documents: duckdb.DuckDBPyRelation,
    only_languages: Iterable[str] | None = None,
    only_consultations_since: datetime.date | None = None,
    only_document_types: Iterable[str] | None = None,
) -> duckdb.DuckDBPyRelation:
    """Filter the documents relation according to the specified criteria."""
    if only_languages is not None:
        # Filter by languages
        rel_documents = rel_documents.filter(isin("document_language", only_languages))
    if only_consultations_since is not None:
        # Filter by consultation date
        rel_documents = rel_documents.filter(
            duckdb.ColumnExpression("consultation_start_date") >= only_consultations_since
        )
    if only_document_types is not None:
        # Filter by document types
        rel_documents = rel_documents.filter(isin("document_type", only_document_types))
    return rel_documents


def isin(column_name: str, values: Iterable[str]) -> duckdb.Expression:
    """Create a duckdb ColumnExpression for an "isin" filter."""
    return duckdb.ColumnExpression(column_name).isin(*map(duckdb.ConstantExpression, values))


def restore_categorical_columns(
    df: pd.DataFrame, schema_cls: type[pa.DataFrameModel] = schemata.FullConsultationDocumentSchemaV1
) -> pd.DataFrame:
    """
    Set column dtypes back to 'category' after a Parquet file is loaded via DuckDB.

    DuckDB cannot preserve Pandas categorical dtypes on load; see https://github.com/duckdb/duckdb/discussions/9617
    Call this function on a freshly loaded dataframe to make it compliant with our schema.
    """
    df = df.copy()
    schema = schema_cls.to_schema()
    for column in schema.columns.values():
        if isinstance(column.dtype.type, pd.CategoricalDtype):
            df[column.name] = pd.Categorical(
                df[column.name],
                categories=schemata.get_allowed_values(schema_cls, column.name),
            )
    return df
