import logging
import os
import re
from collections.abc import Iterable

import pandas as pd


def load_single_docs_df(
    input_file: str,
    characters_per_document: int | None,
    only_document_sources: Iterable[str] | None,
    only_languages: Iterable[str] | None,
    only_doc_types: Iterable[str | None] | None,
    starting_year: int | None,
) -> pd.DataFrame:
    assert input_file.endswith(".parquet"), f"Expected a .parquet file, got {input_file}"
    df = pd.read_parquet(input_file)
    if only_document_sources is not None:
        df = df[df["document_source"].isin(only_document_sources)]
    if only_languages is not None:
        df = df[df["document_language"].isin(only_languages)]
    if only_doc_types is not None:
        df = df[df["document_type"].isin(only_doc_types)]
    if starting_year is not None:
        df = df[df["consultation_start_date"].dt.year >= starting_year]
    # if "text_extraction_error" in df.columns:
    #     df = df[df["text_extraction_error"] == ""]
    df["doc_text_repr"] = df["document_content_plain"].str.strip()
    if characters_per_document is not None:
        df["doc_text_repr"] = df["doc_text_repr"].str[:characters_per_document]
    return df[
        [
            "document_id",
            "consultation_id",
            "consultation_title",
            "consultation_description",
            "consultation_url",
            "consultation_topics",
            "document_type",
            "document_language",
            "document_source",
            "consultation_topics_label_source",
            "organisation_id",
            "organisation_name",
            "doc_text_repr",
        ]
    ]


def load_docs(
    input_files: Iterable[str],
    *,
    characters_per_document: int | None = None,
    only_document_sources: Iterable[str] | None = None,
    only_languages: Iterable[str] | None = None,
    only_doc_types: Iterable[str] | None = None,
    starting_year: int | None = None,
    mlflow=None,
) -> pd.DataFrame:
    dfs = []
    for input_file in input_files:
        df = load_single_docs_df(
            input_file,
            characters_per_document,
            only_document_sources,
            only_languages,
            only_doc_types,
            starting_year,
        )
        dfs.append(df)
        if mlflow is not None:
            name = re.sub(r"-preprocessed.*$", "", os.path.basename(input_file))
            name = name.replace("-documents-dump-", ":")
            dataset = mlflow.data.from_pandas(df, source=input_file, name=name)
            mlflow.log_input(dataset)

    df_docs = pd.concat(dfs)
    return df_docs
