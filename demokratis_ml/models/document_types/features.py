"""See :func:`add_features`."""

import logging

import pandas as pd

from demokratis_ml.data import schemata

logger = logging.getLogger("document_types.features")


def add_features(df_docs: schemata.FullConsultationDocumentV1, df_extra_features: pd.DataFrame) -> pd.DataFrame:
    """Add additional features to the documents dataframe.

    Some features are calculated on the fly, others are joined from the `df_extra_features` dataframe.
    """
    previous_shape = df_docs.shape
    # This makes no difference: we're not losing documents because of the hashes not matching.
    # We're losing them because the PDF extraction failed for quite a lot of documents.
    # df = df_docs.join(
    #   df_extra_features.reset_index(level="stored_file_hash", drop=True), on="document_uuid", how="inner"
    # )
    df = df_docs.join(df_extra_features, on=["document_uuid", "stored_file_hash"], how="inner")

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
    df["consultation_start_timestamp"] = df["consultation_start_date"].view("int64") // 10**9

    # Categorical features
    df["is_federal_consultation"] = df["political_body"] == schemata.FEDERAL_CODE

    # Reports
    logger.info(
        "%d rows (%.1f%%) were lost due to missing features. Remaining rows: %d. %d columns were added.",
        previous_shape[0] - df.shape[0],
        (previous_shape[0] - df.shape[0]) / previous_shape[0] * 100,
        df.shape[0],
        df.shape[1] - previous_shape[1],
    )
    if previous_shape[0] > df.shape[0]:
        lost_docs = df_docs.loc[~df_docs["document_uuid"].isin(df["document_uuid"])].copy()
        lost_docs["year"] = lost_docs["consultation_start_date"].dt.year
        lost_docs["document_type"] = lost_docs["document_type"].astype(str).fillna("None")
        lost_body_year = lost_docs.pivot_table(
            index="political_body",
            columns="year",
            values="document_uuid",
            aggfunc="count",
            margins=True,
            margins_name="Total",
            observed=False,
        )
        lost_body_type = lost_docs.pivot_table(
            index="political_body",
            columns="document_type",
            values="document_uuid",
            aggfunc="count",
            margins=True,
            margins_name="Total",
            observed=False,
        )
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
            logger.info("Lost documents by political_body/year:\n%r", lost_body_year)
            logger.info("Lost documents by political_body/type:\n%r", lost_body_type)
    return df
