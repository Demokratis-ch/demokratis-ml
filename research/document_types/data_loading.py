import pathlib
from collections.abc import Iterable

import pandas as pd
import pandera

from demokratis_ml.data import schemata
from research.document_types import document_title_rule_model
from research.lib import data_access


@pandera.check_types
def load_documents(
    document_file: pathlib.Path,
    external_test_labels_file: pathlib.Path,
    *,
    include_rule_labels: set[str],
    only_languages: Iterable[str] | None = None,
    starting_year: int | None = None,
) -> tuple[schemata.FullConsultationDocumentV1, schemata.FullConsultationDocumentV1]:
    """
    Load consultation documents for document type classification.

    Two dataframes are returned:
    - df_input: Documents that can be used for training, CV, and testing.
    - df_external_test_set: Documents that were labelled manually and can be used for more robust evaluation.

    To separate the two sets, a path to an Excel file with the external test labels must be provided.
    The file must contain a column "document_id" with the document IDs and a column "ground_truth" with the
    manually assigned document types.

    :param include_rule_labels: Originally unlabelled documents that receive these labels from the rule-based model
        will be included in df_input.
    :returns: (df_input, df_external_test_set)
    """
    # Load dataframes
    external_test_labels = _load_external_test_labels(external_test_labels_file)
    df_docs = data_access.load_consultation_documents(
        document_file, only_languages=only_languages, starting_year=starting_year
    )
    df_docs["document_type_label_source"] = "explicit"

    # Generate rule-based labels for unlabelled documents
    df_missing_labels = df_docs.loc[df_docs["document_type"].isna()]
    rule_labels = document_title_rule_model.predict(df_missing_labels)
    rule_labels = rule_labels[rule_labels.isin(include_rule_labels)]
    df_docs.loc[rule_labels.index, "document_type"] = rule_labels
    df_docs.loc[rule_labels.index, "document_type_label_source"] = "rule"

    # For df_input, select only documents that either had an explicit label from the start, or documents
    # that were labelled using the rule-based model and their labels are in the set that we want
    # to include (e.g. because we've verified that the rule-based labelling is reliable);
    # HOWEVER, keep all external test documents out of df_input.
    df_input = df_docs.loc[
        # (~df_docs["document_type"].isna() | df_docs.index.isin(rule_labels.index))
        ~df_docs["document_type"].isna() & ~df_docs["document_id"].isin(external_test_labels.index)
    ]

    # For the external test, use documents that we have labelled.
    # Unfortunately, our training data may be missing some document types that we labelled in the external
    # test set, and we must drop such documents from the external test set because the model cannot be trained
    # to recognise them.
    df_external_test = (
        df_docs.join(external_test_labels, on="document_id", how="inner")
        .drop(columns=["document_type"])
        .rename(columns={"ground_truth": "document_type"})
    )
    df_external_test = df_external_test.loc[df_external_test["document_type"].isin(df_input["document_type"].unique())]
    df_external_test.loc[:, "document_type_label_source"] = "external_test"

    # Sanity checks
    assert not df_input["document_type"].isna().any(), "All df_input docs must be labelled"
    assert not df_external_test["document_type"].isna().any(), "All df_external_test docs must be labelled"
    assert not (
        set(df_input["document_id"]) & set(df_external_test["document_id"])
    ), "No overlap may exist between df_input and df_external_test"

    return df_input, df_external_test


def _load_external_test_labels(file: pathlib.Path) -> pd.Series:
    assert file.suffix == ".xlsx"
    df = pd.read_excel(file)[["document_id", "ground_truth"]].set_index("document_id")
    df["ground_truth"] = pd.Categorical(df["ground_truth"], categories=schemata.DOCUMENT_TYPES)
    df = df.dropna()
    return df["ground_truth"]
