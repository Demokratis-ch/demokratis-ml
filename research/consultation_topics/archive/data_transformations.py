import numpy as np
import pandas as pd


def group_document_labels_by_consultation(
    consultation_identifiers: pd.Series,
    label_names: list[str] | tuple[str, ...] | pd.Index,
    doc_labels: np.ndarray | pd.DataFrame,
    threshold: float = 0.333,
) -> pd.DataFrame:
    """When documents are labelled individually (e.g. when labels are known, or when they're predicted
    by a model), we want to combine these labels to get a single label for each consultation.
    (Recall that each consultation can have multiple documents.)

    This function groups document labels by consultation, and then votes on the labels for each
    consultation. If more than (threshold * 100)% of documents vote for a given label, that label
    is assigned to the consultation.
    """
    if isinstance(doc_labels, pd.DataFrame):
        doc_labels = doc_labels.to_numpy()
    # The code would run but it'd produce nonsense if a Pandas dataframe was passed instead.
    assert isinstance(doc_labels, np.ndarray)
    assert consultation_identifiers.size == doc_labels.shape[0]
    assert len(label_names) == doc_labels.shape[1]
    assert 0 < threshold <= 1

    df_docs = pd.DataFrame(doc_labels, columns=label_names)
    df_docs["consultation_identifier"] = consultation_identifiers.reset_index(drop=True)

    def vote(doc_labels: pd.Series) -> pd.Series:
        return (doc_labels.sum() > doc_labels.size * threshold).astype(int)

    df_consultation_labels = df_docs.groupby("consultation_identifier").agg(vote)
    return df_consultation_labels
