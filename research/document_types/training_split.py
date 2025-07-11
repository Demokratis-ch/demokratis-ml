import logging

import pandas as pd
import sklearn.model_selection

from demokratis_ml.models.document_types import preprocessing
from research.document_types import document_title_rule_model

logger = logging.getLogger("training_split")


def train_test_split(
    df: pd.DataFrame,
    random_state: int,
    test_sizes: tuple[float] | tuple[float, float],
    include_rule_labels_in_training: set[str],
    stratify_by_canton: bool,
) -> tuple[pd.DataFrame, tuple[pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Split the input dataframe (documents + extra features + embeddings) into one training and one/two
    test sets.

    All Fedlex documents are used for training. OpenParlData documents are used as follows:
    - Documents that already have labels must have been manually labelled by Demokratis staff; this high-quality
      data is split into training and test sets according to the `test_sizes` parameter.
    - Some documents with labels assigned by the rule-based model are used for training: only classes
      listed in `include_rule_labels_in_training` are used as those are vetted to be reliable.
    - The remaining OpenParlData documents are unlabelled and therefore discarded.

    The resulting dataframes have the same columns as the input dataframe, but the `document_type` column is
    guaranteed to be non-null. In addition, the `document_type_label_source` column is set to "rule"
    for documents with labels assigned by the rule-based model.

    :param stratify_by_canton: If True, the combination of `political_body` and `document_type` is used
        for stratification. If False, only `document_type` is used.
    """
    # All Fedlex documents are for training
    df_fedlex = df.loc[df["document_source"] == "fedlex"]

    df_openparldata = df.loc[df["document_source"] == "openparldata"]
    # From the unlabelled OpenParlData, we use some rule-based labels for training
    df_openparldata_unlabelled = df_openparldata.loc[df_openparldata["document_type"].isna()]
    rule_labels = document_title_rule_model.predict(df_openparldata_unlabelled)
    rule_labels = preprocessing.merge_classes(rule_labels, preprocessing.MERGE_CLASSES)
    rule_labels = rule_labels[rule_labels.isin(include_rule_labels_in_training)]
    df_openparldata_rules = df_openparldata.loc[rule_labels.index]
    df_openparldata_rules.loc[rule_labels.index, "document_type"] = rule_labels
    df_openparldata_rules.loc[rule_labels.index, "document_type_label_source"] = "rule"

    # Manually labelled OpenParlData documents will be split into train and test
    df_openparldata_manual = df_openparldata.loc[df_openparldata["document_type"].notna()]
    len_labelled = len(df_openparldata_manual)
    df_openparldata_manual = df_openparldata_manual.loc[
        # Be strict: accept only documents that are explicitly tagged as reviewed
        df_openparldata_manual["document_type_label_source"] == "manual"
    ]
    logger.info(
        "Number of manually labelled openparldata documents: %d, of which %d are explicitly tagged as reviewed",
        len_labelled,
        len(df_openparldata_manual),
    )
    df_openparldata_manual_train, df_openparldata_manual_test = _split(
        df_openparldata_manual,
        random_state=random_state,
        test_size=sum(test_sizes),
        stratify_by_canton=stratify_by_canton,
    )

    # Concatenate all training data
    df_train = pd.concat([df_fedlex, df_openparldata_rules, df_openparldata_manual_train], ignore_index=True)
    logger.info(
        "Training set: %d fedlex + %d rule labels + %d manual labels = %d",
        len(df_fedlex),
        len(df_openparldata_rules),
        len(df_openparldata_manual_train),
        len(df_train),
    )
    # Only some manually labelled OpenParlData documents will be used for testing
    if len(test_sizes) == 1:
        logger.info("Test set 1/1: %d manual labels", len(df_openparldata_manual_test))
        df_tests = (df_openparldata_manual_test,)
    elif len(test_sizes) == 2:  # noqa: PLR2004
        df_test_1, df_test_2 = _split(
            df_openparldata_manual_test,
            random_state=random_state,
            # Second index: df_test_2
            test_size=test_sizes[1] / sum(test_sizes),
            stratify_by_canton=stratify_by_canton,
        )
        logger.info("Test set 1/2: %d manual labels, Test set 2/2: %d manual labels", len(df_test_1), len(df_test_2))
        df_tests = (df_test_1, df_test_2)
    else:
        raise ValueError("test_sizes must be a tuple of length 1 or 2")

    # Integrity checks
    assert not df_train["document_uuid"].duplicated().any(), "Train set must not contain duplicates"
    assert df_train["document_type"].notna().all(), "Train set must not contain null document types"

    for df_test in df_tests:
        assert not df_test["document_uuid"].duplicated().any(), "Test set must not contain duplicates"
        assert not (
            set(df_train["document_uuid"]) & set(df_test["document_uuid"])
        ), "Train and test sets must not overlap"
        assert df_test["document_type"].notna().all(), "Test set must not contain null document types"
    if len(df_tests) == 2:  # noqa: PLR2004
        assert not (set(df_tests[0]["document_uuid"]) & set(df_tests[1]["document_uuid"])), "Test sets must not overlap"

    return df_train, df_tests


def _split(
    df: pd.DataFrame, random_state: int, test_size: float, stratify_by_canton: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = sklearn.model_selection.StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    if stratify_by_canton:
        stratification_groups = df["political_body"].astype(str) + ":" + df["document_type"].astype(str)
        class_counts = stratification_groups.value_counts()
        minimum_class_size = 2
        rare_classes = class_counts[class_counts < minimum_class_size].index
        rare_index = stratification_groups.isin(rare_classes)
        if not rare_classes.empty:
            logger.warning(
                "Discarding %d documents in the following rare classes (fewer than %d samples): %r",
                len(df[rare_index]),
                minimum_class_size,
                sorted(rare_classes),
            )
        train_index, test_index = next(
            splitter.split(
                X=df[~rare_index],
                y=stratification_groups[~rare_index],
            )
        )
    else:
        train_index, test_index = next(splitter.split(X=df, y=df["document_type"]))

    return df.iloc[train_index], df.iloc[test_index]
