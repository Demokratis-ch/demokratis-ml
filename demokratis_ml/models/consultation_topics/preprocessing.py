"""Data preparation for consultation topics classification."""

import logging
from collections.abc import Iterable

import duckdb
import pandas as pd
import sklearn.preprocessing

from demokratis_ml.data import loading

logger = logging.getLogger("document_types.preprocessing")


def create_input_dataframe(
    rel_documents: duckdb.DuckDBPyRelation,
    rel_document_embeddings: duckdb.DuckDBPyRelation,
    rel_consultation_embeddings: duckdb.DuckDBPyRelation,
    use_document_types: Iterable[str],
    use_attributes: tuple[str, ...] = (
        "consultation_title",
        # "consultation_description",  # Not used by default because many consultations don't have it
        "organisation_name",
    ),
) -> tuple[pd.DataFrame, list[str]]:
    """Create a model input dataframe (for training or inference) from consultation documents and their embeddings.

    Each row corresponds to a consultation, with embeddings of its documents and attributes.
    """
    # Join documents with their embeddings via DuckDB to avoid loading large dataframes into memory
    df_docs_embeddings = (
        rel_documents.filter(loading.isin("document_type", use_document_types))
        .join(rel_document_embeddings, condition="document_uuid", how="inner")
        .df()
        .rename(columns={"embedding": "embedding_documents"})
    )
    df_docs_embeddings = loading.restore_categorical_columns(df_docs_embeddings)
    # Filter consultation embeddings to avoid loading unnecessary attributes
    df_consultation_embeddings = (
        rel_consultation_embeddings.filter(
            duckdb.ColumnExpression("attribute_name").isin(*map(duckdb.ConstantExpression, use_attributes))
        )
        .df()
        .set_index(["consultation_identifier", "attribute_language", "attribute_name"])
    )

    # Grouping by consultation, and joining attribute embeddings happens in Pandas because it's easier to express there
    df = df_docs_embeddings.groupby("consultation_identifier").agg(
        {
            **dict.fromkeys(
                [
                    "consultation_start_date",
                    "consultation_end_date",
                    "consultation_title",
                    "consultation_description",
                    "consultation_url",
                    "consultation_topics",
                    "organisation_uuid",
                    "organisation_name",
                    "political_body",
                ],
                "first",
            ),
            # "embedding_documents": lambda embeddings: np.stack(embeddings).max(axis=0),  # max-pooling
            "embedding_documents": "mean",
            # "document_content_plain": "\n\f".join,  # Not used by the model
            # "document_language": list,  # We're not using this yet
        }
    )
    for attribute in use_attributes:
        len_before = len(df)
        df = df.join(
            _get_embeddings_by_attribute(df_consultation_embeddings, attribute),
            on="consultation_identifier",
            how="inner",
        )
        if lost_rows := len_before - len(df):
            logger.warning("Lost %d rows while joining %s embeddings", lost_rows, attribute)

    non_nullable_columns = list(set(df.columns) - {"consultation_end_date"})
    nulls_per_column = df[non_nullable_columns].isna().any()
    assert not nulls_per_column.any(), repr(nulls_per_column)
    assert df.index.is_unique, (
        "Consultation identifiers must be unique; duplication may have been caused by multiple languages?"
    )
    return encode_topics(df)


def _get_embeddings_by_attribute(df: pd.DataFrame, attribute_name: str) -> pd.Series:
    # TODO: what about languages? Should we average embeddings across languages in this function?
    # This function only works now because the dataframes coming into `create_input_dataframe` are already
    # filtered down to a single language.
    idx = df.index.get_level_values("attribute_name") == attribute_name
    series = df[idx].reset_index(["attribute_language", "attribute_name"], drop=True)["embedding"]
    series.name = f"embedding_{attribute_name}"
    return series


def encode_topics(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Encode the consultation topics as multi-hot columns."""
    topic_binarizer = sklearn.preprocessing.MultiLabelBinarizer()
    one_hot_labels = pd.DataFrame(
        topic_binarizer.fit_transform(df["consultation_topics"]),
        columns=[f"topic_{c}" for c in topic_binarizer.classes_],
        index=df.index,
    )
    topic_columns = one_hot_labels.columns.tolist()
    df_encoded = pd.concat([df, one_hot_labels], axis=1)
    return df_encoded, topic_columns


def drop_underrepresented_topics(
    df: pd.DataFrame,
    topic_columns: Iterable[str],
    min_consultations_in_class: int,
    *,
    always_drop_topics: Iterable[str] = (),
) -> tuple[pd.DataFrame, list[str]]:
    """Drop topics that are not represented in enough consultations.

    Also drops consultations that no longer have any labels after dropping the under-represented topics.
    """
    # Drop columns
    always_drop_topics = {t if t.startswith("topic_") else f"topic_{t}" for t in always_drop_topics}
    if always_drop_topics - set(topic_columns):
        raise ValueError(
            "The following topics are not present in the input data", always_drop_topics - set(topic_columns)
        )
    consultations_per_topic = df[topic_columns].sum(axis=0).sort_values(ascending=False)
    to_drop = consultations_per_topic[
        (consultations_per_topic < min_consultations_in_class)
        | (consultations_per_topic.index.isin(always_drop_topics))
    ]
    print("Dropping these topics:", to_drop, sep="\n")
    df = df.drop(columns=to_drop.index)
    topic_columns = [c for c in topic_columns if c not in to_drop.index]
    # Drop rows that no longer have any labels
    samples_without_label = df[topic_columns].sum(axis=1) == 0
    print(
        "Dropping these samples without any label:",
        len(df[samples_without_label]),
    )
    df = df[~samples_without_label]

    return df, topic_columns
