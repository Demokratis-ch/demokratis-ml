"""Data preparation for consultation topics classification."""

import logging
from collections.abc import Iterable

import pandas as pd
import sklearn.preprocessing

from demokratis_ml.data import schemata

logger = logging.getLogger("document_types.preprocessing")


def create_input_dataframe(
    df_documents: schemata.FullConsultationDocumentV1,
    df_document_embeddings: pd.DataFrame,
    df_consultation_embeddings: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Create a model input dataframe (for training or inference) from consultation documents and embeddings.

    Each row corresponds to a consultation, with embeddings of its documents and attributes.
    """
    df_docs_embeddings = df_documents.join(df_document_embeddings, on="document_uuid", how="inner").rename(
        columns={"embedding": "embedding_documents"}
    )
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
            "document_content_plain": "\n\f".join,
        }
    )

    for attribute in ("consultation_title", "consultation_description", "organisation_name"):
        len_before = len(df)
        df = df.join(
            _get_embeddings_by_attribute(df_consultation_embeddings, attribute),
            on="consultation_identifier",
            how="inner",
        )
        if lost_rows := len_before - len(df):
            logger.warning("Lost %d rows while joining %s embeddings", lost_rows, attribute)

    nulls_per_column = df.isna().any()
    assert not nulls_per_column.any(), repr(nulls_per_column)
    return encode_topics(df)


def _get_embeddings_by_attribute(df: pd.DataFrame, attribute_name: str) -> pd.Series:
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
