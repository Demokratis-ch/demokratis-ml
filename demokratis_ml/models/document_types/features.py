"""See :func:`add_features`."""

import logging

import pandas as pd

from demokratis_ml.data import schemata


def add_features(df_docs: schemata.FullConsultationDocumentV1, df_extra_features: pd.DataFrame) -> pd.DataFrame:
    """Add additional features to the documents dataframe.

    Some features are calculated on the fly, others are joined from the `df_extra_features` dataframe.
    """
    previous_shape = df_docs.shape
    df = df_docs.join(df_extra_features, on=["document_id", "stored_file_hash"], how="inner")
    df["fraction_pages_containing_tables"] = df["count_pages_containing_tables"] / df["count_pages"]
    df["fraction_pages_containing_images"] = df["count_pages_containing_images"] / df["count_pages"]
    df["contains_synopse_keyword"] = (
        df["document_content_plain"].str.slice(0, 1000).str.contains("synopse", case=False, regex=False)
    )
    df["contains_salutation"] = (
        df["document_content_plain"]
        .str.slice(0, 3000)
        .str.contains(
            r"(?:Sehr\s+geehrte[r]?\s+(?:Frau|Herr|Damen\s+und\s+Herren)|"
            r"Liebe[r]?\s+(?:Frau|Herr|Damen\s+und\s+Herren)|"
            r"Sehr\s+geehrte[r]?\s+(?:"
            r"Bundesr(?:at|ätin)|"
            r"Regierungsr(?:at|ätin)|"
            r"Nationalr(?:at|ätin)|"
            r"Stadtpr[äa]sid(?:ent|entin)|"
            r"Gemeindepr[äa]sid(?:ent|entin)|"
            r"Stadtr(?:at|ätin)|"
            r"Gemeinder(?:at|ätin)|"
            r"Pr[äa]sid(?:ent|entin)))",
            case=False,
            regex=True,
        )
    )
    logging.info(
        "%d rows were lost due to missing features. Remaining rows: %d. %d columns were added.",
        previous_shape[0] - df.shape[0],
        df.shape[0],
        df.shape[1] - previous_shape[1],
    )
    return df
