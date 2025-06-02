"""Data preparation for document type classification."""

import logging

import pandas as pd

from demokratis_ml.data import schemata
from demokratis_ml.models.document_types import features

MERGE_CLASSES = {
    ("RESPONSE_FORM",): "SURVEY",
    ("DECISION", "PRESS_RELEASE"): "VARIOUS_TEXT",
}


def create_input_dataframe(
    df_documents: schemata.FullConsultationDocumentV1,
    *,
    df_extra_features: pd.DataFrame,
    df_embeddings: pd.DataFrame,
    class_merges: dict[tuple[str, ...], str] = MERGE_CLASSES,
) -> pd.DataFrame:
    """
    Create a model input dataframe (for training or inference) from the documents and their features.

    :param df_documents: The "main" dataframe containing the consultation documents.
    :param df_extra_features: The dataframe containing additional features extracted from PDFs.
    :param df_embeddings: The dataframe containing the embeddings of the documents.
    :param class_merges: :func:`merge_classes` will be applied to both dataframes using this mapping.
    """
    df_documents = _drop_empty_texts(df_documents)
    df = features.add_features(df_documents, df_extra_features)
    df = _add_embeddings(df, df_embeddings)
    df.loc[:, "document_type"] = merge_classes(df["document_type"], class_merges)
    return df


def merge_classes(series: pd.Series, merge_to: dict[tuple[str, ...], str]) -> pd.Series:
    """Merge classes in a series of document type labels.

    :param merge_to: {(classes, to, replace): replacement_class, ...}
    """
    series = series.copy()
    for old_classes, new_class in merge_to.items():
        mask = series.isin(old_classes)
        series.loc[mask] = new_class
    return series


def _add_embeddings(df_documents: pd.DataFrame, df_embeddings: pd.DataFrame) -> pd.DataFrame:
    previous_shape = df_documents.shape
    df = df_documents.join(df_embeddings, on="document_id", how="inner")
    logging.info(
        "%d rows were lost due to missing embeddings. Remaining rows: %d. %d columns were added.",
        previous_shape[0] - df.shape[0],
        df.shape[0],
        df.shape[1] - previous_shape[1],
    )
    return df


def _drop_empty_texts(df: schemata.FullConsultationDocumentV1) -> schemata.FullConsultationDocumentV1:
    empty_index = df["document_content_plain"].str.strip() == ""
    empty_count = empty_index.sum()
    logging.info("Dropping %d documents (%.1f%%) with empty texts", empty_count, 100 * empty_count / len(df))
    return df.loc[~empty_index]
