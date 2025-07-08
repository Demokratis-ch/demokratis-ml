import pandas as pd
import pytest

from research.consultation_topics import data_transformations


@pytest.fixture
def documents() -> pd.DataFrame:
    df = pd.DataFrame(
        [
            [100, 1, 0, 0],
            [100, 0, 1, 0],
            [100, 1, 1, 1],
            # ---
            [200, 0, 0, 1],
            # ---
            [300, 0, 1, 0],
            [300, 1, 1, 0],
        ],
        columns=["consultation_identifier", "A", "B", "C"],
    )
    return df.set_index("consultation_identifier")


@pytest.mark.parametrize("threshold", [0.333, 0.5])
def test_group_document_labels_by_consultation(
    documents: pd.DataFrame,
    threshold: float,
) -> None:
    if threshold == 0.333:
        expected_labels = [
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 0],
        ]
    elif threshold == 0.5:
        expected_labels = [
            [1, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
        ]
    else:
        raise ValueError("Unexpected threshold value", threshold)
    expected_df = pd.DataFrame(
        expected_labels,
        index=pd.Index([100, 200, 300], name="consultation_identifier"),
        columns=["A", "B", "C"],
    )

    df_consultation_labels = data_transformations.group_document_labels_by_consultation(
        consultation_identifiers=documents.index.to_series(),
        label_names=documents.columns,
        doc_labels=documents.values,
        threshold=threshold,
    )
    pd.testing.assert_frame_equal(df_consultation_labels, expected_df)
