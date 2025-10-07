"""See :func:`add_features`."""

import logging

import pandas as pd

from demokratis_ml.data import schemata

logger = logging.getLogger("document_types.features")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate additional features and add them to the documents dataframe."""
    # Features derived from PDFs (extra features)
    df["fraction_pages_containing_tables"] = df["count_pages_containing_tables"] / df["count_pages"]
    df["fraction_pages_containing_images"] = df["count_pages_containing_images"] / df["count_pages"]

    # Keyword-like features
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

    # Time features
    df["days_after_consultation_start"] = (df["document_publication_date"] - df["consultation_start_date"]).dt.days
    df["days_after_consultation_end"] = (df["document_publication_date"] - df["consultation_end_date"]).dt.days
    df["consultation_start_timestamp"] = df["consultation_start_date"].astype("int64") // 10**9

    # Categorical features
    df["is_federal_consultation"] = df["political_body"] == schemata.FEDERAL_CODE

    return df
