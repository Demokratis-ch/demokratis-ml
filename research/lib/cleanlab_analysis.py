import pathlib
from collections.abc import Iterable

import cleanlab.internal.multilabel_utils
import numpy as np
import pandas as pd
from cleanlab.datalab.datalab import Datalab as DatalabClass


def cleanlab_issues_to_excel(  # noqa: PLR0913
    lab: DatalabClass,
    pred_probs: np.ndarray,
    dataset: pd.DataFrame,
    output_path: pathlib.Path | str,
    linkify_columns: Iterable[str] = (),
    issue_types: Iterable[str] = (
        "label",
        "near_duplicate",
        "outlier",
        "underperforming_group",
        "non_iid",
    ),
    multilabel_true_labels: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    # Prepare a DataFrame with the given and predicted labels for all samples
    if multilabel_true_labels is None:
        given_labels = dataset[lab.label_name]
        given_labels.name = "given_label"
        predicted_labels = pd.DataFrame(pred_probs, columns=lab.class_names, index=dataset.index).idxmax(axis=1)
        predicted_labels.name = "predicted_label"
        assert len(given_labels) == len(predicted_labels)
        labels = pd.concat([given_labels, predicted_labels], axis=1)
    else:

        def int2labels(ints: list[int]) -> list[str]:
            return [multilabel_true_labels.columns[i] for i in ints]

        given_labels = list(
            map(int2labels, cleanlab.internal.multilabel_utils.onehot2int(multilabel_true_labels.values))
        )
        predicted_labels = (pred_probs > 0.5).astype(int)
        predicted_labels = list(map(int2labels, cleanlab.internal.multilabel_utils.onehot2int(predicted_labels)))
        assert len(given_labels) == len(predicted_labels)
        labels = pd.DataFrame(
            {
                "given_label": given_labels,
                "predicted_label": predicted_labels,
            },
            index=dataset.index,
        )

    returned_dfs = {}

    with pd.ExcelWriter(output_path) as writer:
        # For each issue type, create a DataFrame and save it as a sheet
        for issue_type in issue_types:
            df_examples = lab.get_issues(issue_type).query(f"is_{issue_type}_issue")
            df_examples = df_examples.drop(columns=[f"is_{issue_type}_issue"])
            df = dataset.join(df_examples, how="inner")
            df = df.sort_values(f"{issue_type}_score")
            if issue_type == "label":
                # For the "label" issue type, the columns "given_label" and "predicted_label" are already present.
                # We remove them because we want our own labels (possibly remapped in the multilabel case).
                df = df.drop(columns=["given_label", "predicted_label"])
            # Add labels back
            df = df.join(labels)
            # Don't include this column twice (it's already in "given_label") to avoid confusion
            if lab.label_name in df.columns:
                df = df.drop(columns=[lab.label_name])

            if issue_type == "near_duplicate":
                df = _expand_near_duplicates(df)

            # Save the DataFrame to the returned dictionary before adding Excel formatting/functions
            returned_dfs[issue_type] = df.copy()

            for column in linkify_columns:
                df[column] = df[column].map(lambda x: f'=HYPERLINK("{x}")')
            # Save the DataFrame to a sheet named after the issue type
            df.to_excel(writer, sheet_name=issue_type, index=True)

        return returned_dfs


def _expand_near_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    unique_duplicate_sets = {frozenset(row["near_duplicate_sets"]) | {i} for i, row in df.iterrows()}

    def select_set(s: frozenset[int], set_id: int) -> pd.DataFrame:
        duplicates = df.loc[list(s)]
        return duplicates.assign(
            set_id=f"#{set_id} ({len(s)} docs)",
            total_set_score=duplicates["near_duplicate_score"].sum(),
        )

    df_expanded = pd.concat(
        [select_set(duplicate_set, i) for i, duplicate_set in enumerate(unique_duplicate_sets)],
    )
    df_expanded = df_expanded.sort_values(["total_set_score", "set_id"])
    df_expanded = df_expanded.drop(columns=["total_set_score"])
    return df_expanded
