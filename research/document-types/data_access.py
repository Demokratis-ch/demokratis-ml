import pathlib
from collections.abc import Iterable

import pandas as pd
import pandera as pa

from demokratis_ml.data import schemata


@pa.check_output(schemata.FullConsultationDocumentSchemaV1.to_schema())
def load_consultation_documents(
    input_file: pathlib.Path,
    *,
    only_document_sources: Iterable[str] | None = None,
    only_languages: Iterable[str] | None = None,
    only_doc_types: Iterable[str] | None = None,
    starting_year: int | None = None,
    mlflow=None,
) -> pd.DataFrame:
    """Load and filter consultation documents from a parquet file.

    If an MLflow client is provided, the loaded dataset is logged as an input artifact.
    """
    # Read
    assert input_file.suffix == ".parquet", f"Expected a .parquet file, got {input_file}"
    df = pd.read_parquet(input_file)
    # Filter
    if only_document_sources is not None:
        df = df[df["document_source"].isin(only_document_sources)]
    if only_languages is not None:
        df = df[df["document_language"].isin(only_languages)]
    if only_doc_types is not None:
        assert set(only_doc_types) <= schemata.DOCUMENT_TYPES, f"Unknown doc types: {only_doc_types}"
        df = df[df["document_type"].isin(only_doc_types)]
    if starting_year is not None:
        df = df[df["consultation_start_date"].dt.year >= starting_year]
    # Log to MLflow
    if mlflow is not None:
        name = input_file.name.replace(".parquet", "")
        dataset = mlflow.data.from_pandas(df, source=input_file, name=name)
        mlflow.log_input(dataset)
    return df
