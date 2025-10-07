"""Data preparation for document type classification."""

import logging

import duckdb
import pandas as pd

from demokratis_ml.data import loading, schemata
from demokratis_ml.models.document_types import features

MERGE_CLASSES = {
    ("RESPONSE_FORM",): "SURVEY",
    ("DECISION", "PRESS_RELEASE"): "VARIOUS_TEXT",
}

logger = logging.getLogger("document_types.preprocessing")


def create_input_dataframe(
    rel_documents: duckdb.DuckDBPyRelation,
    rel_extra_features: duckdb.DuckDBPyRelation,
    rel_embeddings: duckdb.DuckDBPyRelation,
    class_merges: dict[tuple[str, ...], str] = MERGE_CLASSES,
) -> pd.DataFrame:
    """
    Create a model input dataframe (for training or inference) from the documents and their features.

    :param rel_documents: The "main" table containing the consultation documents.
    :param rel_extra_features: The table containing additional features extracted from PDFs.
    :param rel_embeddings: The table containing the embeddings of the documents.
    :param class_merges: :func:`merge_classes` will be applied to the resulting dataframe using this mapping.
    """
    rel_joined = rel_documents.join(
        rel_extra_features, condition=["document_uuid", "stored_file_hash"], how="inner"
    ).join(rel_embeddings, condition="document_uuid", how="inner")
    # This makes no difference: we're not losing documents because of the hashes not matching.
    # We're losing them because the PDF extraction failed for quite a lot of documents.
    # df = df_docs.join(
    #   df_extra_features.reset_index(level="stored_file_hash", drop=True), on="document_uuid", how="inner"
    # )
    df = rel_joined.to_df()
    df = loading.restore_categorical_columns(df)
    df = _drop_empty_texts(df)
    df = features.add_features(df)
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


def _drop_empty_texts(df: schemata.FullConsultationDocumentV1) -> schemata.FullConsultationDocumentV1:
    empty_index = df["document_content_plain"].str.strip() == ""
    empty_count = empty_index.sum()
    logger.info("Dropping %d documents (%.1f%%) with empty texts", empty_count, 100 * empty_count / len(df))
    return df.loc[~empty_index]
