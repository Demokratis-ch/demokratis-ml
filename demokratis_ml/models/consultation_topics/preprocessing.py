"""Data preparation for consultation topics classification."""

import logging
from collections.abc import Iterable

import pandas as pd
import sklearn.preprocessing

from demokratis_ml.data import schemata

_INPUT_COLUMNS = [
    # metadata
    "consultation_identifier",
    "document_uuid",
    "document_type",
    "document_language",
    "consultation_topics_label_source",
    # X
    "embedding",
    # y
    "consultation_topics",
]

logger = logging.getLogger("document_types.preprocessing")


def create_input_dataframe(
    df_documents: schemata.FullConsultationDocumentV1,
    df_document_embeddings: pd.DataFrame,
    df_attribute_embeddings: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create a model input dataframe (for training or inference) from documents and their features.

    :param df_documents: The "main" dataframe containing the consultation documents.
    :param df_document_embeddings: The dataframe containing the embeddings of the documents.
    :param class_merges: :func:`merge_classes` will be applied to both dataframes using this mapping.
    """
    df_documents = _drop_empty_texts(df_documents)
    df_from_documents = _add_embeddings(df_documents, df_document_embeddings)
    df_from_attributes = _create_input_from_attribute_embeddings(df_documents, df_attribute_embeddings)
    df = pd.concat(
        [df_from_documents[_INPUT_COLUMNS], df_from_attributes],
        axis=0,
    )
    return encode_topics(df)


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


def _create_input_from_attribute_embeddings(
    df_documents: schemata.FullConsultationDocumentV1, df_attribute_embeddings: pd.DataFrame
) -> pd.DataFrame:
    df_consultation_data = df_documents.groupby(["consultation_identifier", "document_language"], observed=True).agg(
        {"consultation_topics": "first", "consultation_topics_label_source": "first"}
    )
    df = df_attribute_embeddings.reset_index(level="attribute_name").join(df_consultation_data, how="left")
    df = df.reset_index().rename(columns={"attribute_name": "document_type"})
    df["document_uuid"] = ""
    df = df[df["consultation_topics"].notna()]
    return df[_INPUT_COLUMNS]


def _add_embeddings(df_documents: pd.DataFrame, df_embeddings: pd.DataFrame) -> pd.DataFrame:
    previous_shape = df_documents.shape
    df = df_documents.join(df_embeddings, on="document_uuid", how="inner")
    logger.info(
        "%d rows were lost due to missing embeddings. Remaining rows: %d. %d columns were added.",
        previous_shape[0] - df.shape[0],
        df.shape[0],
        df.shape[1] - previous_shape[1],
    )
    return df


def _drop_empty_texts(df: schemata.FullConsultationDocumentV1) -> schemata.FullConsultationDocumentV1:
    empty_index = df["document_content_plain"].str.strip() == ""
    empty_count = empty_index.sum()
    logger.info("Dropping %d documents (%.1f%%) with empty texts", empty_count, 100 * empty_count / len(df))
    return df.loc[~empty_index]


def drop_underrepresented_topics(
    df_input: pd.DataFrame,
    topic_columns: Iterable[str],
    min_consultations_in_class: int,
    *,
    always_drop_topics: Iterable[str] = (),
) -> tuple[pd.DataFrame, list[str]]:
    """Drop topics that are not represented in enough samples (documents).

    Also drops documents that no longer have any labels after dropping the under-represented topics.
    """
    # Drop columns
    always_drop_topics = {t if t.startswith("topic_") else f"topic_{t}" for t in always_drop_topics}
    if always_drop_topics - set(topic_columns):
        raise ValueError(
            "The following topics are not present in the input data", always_drop_topics - set(topic_columns)
        )
    consultations_per_topic = (
        df_input.groupby("consultation_identifier").agg(dict.fromkeys(topic_columns, "first")).sum()
    )
    to_drop = consultations_per_topic[
        (consultations_per_topic < min_consultations_in_class)
        | (consultations_per_topic.index.isin(always_drop_topics))
    ]
    print("Dropping these underrepresented classes:", to_drop, sep="\n")
    df_input = df_input.drop(columns=to_drop.index)
    topic_columns = [c for c in topic_columns if c not in to_drop.index]
    # Drop rows that no longer have any labels
    documents_without_label = df_input[topic_columns].sum(axis=1) == 0
    print(
        "Dropping these documents without any label:",
        len(df_input[documents_without_label]),
    )
    df_input = df_input[~documents_without_label]

    return df_input, topic_columns
