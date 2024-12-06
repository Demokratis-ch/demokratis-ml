"""Pandera schema definitions for the data used in the project."""

from typing import Any, cast

import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import Series

CONSULTATION_TOPICS = {
    "administration",
    "agriculture",
    "communications",
    "culture",
    "defense",
    "economics",
    "education",
    "energy",
    "environment",
    "finance",
    "foreign_policy",
    "health",
    "housing",
    "insurance",
    "it",
    "law",
    "media",
    "migration",
    "political_system",
    "public_finance",
    "science",
    "security",
    "social",
    "spatial_planning",
    "sports",
    "transportation",
}

DOCUMENT_TYPES = {
    # These types also exist in Fedlex data:
    "LETTER",
    "DRAFT",
    "RECIPIENT_LIST",
    "REPORT",
    "FINAL_REPORT",
    "OPINION",
    "VARIOUS_TEXT",
    "SYNOPTIC_TABLE",
    "SURVEY",
    "RESPONSE_FORM",
    # Invented by Demokratis:
    "LINK",
    "DECISION",
}

CANTON_CODES = {
    "ag",
    "ai",
    "ar",
    "be",
    "bl",
    "bs",
    "fr",
    "ge",
    "gl",
    "gr",
    "ju",
    "lu",
    "ne",
    "nw",
    "ow",
    "sg",
    "sh",
    "so",
    "sz",
    "tg",
    "ti",
    "ur",
    "vd",
    "vs",
    "zg",
    "zh",
}

FEDERAL_CODE = "ch"


class ConsultationDocumentMetadataSchemaV1(pa.DataFrameModel):
    """Schema for a denormalized table of document metadata.

    Each row represents a document belonging to a consultation, as well as some attributes
    of the consultation itself.
    """

    # ---------- Consultation attributes ----------
    consultation_id: int
    """ ID of the consultation, assigned by Demokratis """

    consultation_start_date: pd.Timestamp

    consultation_end_date: pd.Timestamp = pa.Field(nullable=True)  # Consultation end may not be set yet

    consultation_title: str = pa.Field(str_length={"min_value": 3})
    """ Title of the consultation """

    consultation_description: str
    """ Short summary of the consultation, sometimes empty """

    consultation_url: str = pa.Field(str_startswith="https://www.demokratis.ch/vernehmlassung/")
    """ URL to the consultation page on Demokratis """

    consultation_topics: np.object_
    """ Zero, one, or more topics that the consultation covers """

    @pa.check("consultation_topics")
    def _check_topics(cls, topics: Series[Any]) -> Series[bool]:  # noqa: N805
        return cast(
            Series[bool],
            topics.map(
                lambda topics: isinstance(topics, np.ndarray)
                # and topics.dtype.type is np.str_
                # The string dtype is not preserved in Parquet but that's okay since we check
                # against the predefined CONSULTATION_TOPICS set anyway.
                and set(topics) <= CONSULTATION_TOPICS
            ),
        )

    consultation_topics_label_source: pd.CategoricalDtype = pa.Field(
        isin={"openparldata", "organisation_rule", "manual"}
    )
    """ How were the topics assigned to the consultation?
    - "openparldata": Topics were provided by the OpenParlData API.
    - "organisation_rule": Topics were assigned by a simple if-then-else rule based on the `organisation_name`.
    - "manual": Topics were assigned by a human reviewer.
    """

    consultation_reviewed_at: pd.Timestamp = pa.Field(nullable=True)
    """ Timestamp when (if) the consultation was reviewed by Demokratis staff.
    A review implies that the consultation attributes (such as topics etc.) were checked by a human
    and are considered correct. """

    organisation_id: int
    """ ID of the organisation that published the consultation; ID is assigned by Demokratis """

    organisation_name: str = pa.Field(str_length={"min_value": 3})
    """ Name of the organisation that published the consultation """

    political_body: pd.CategoricalDtype = pa.Field(isin={FEDERAL_CODE} | CANTON_CODES)
    """ Code of the political body that the consultation belongs to; may be a canton or the federal
    government. Federal consultations have the code "ch". """

    # ---------- Document attributes ----------
    document_id: int
    """ ID of the document, assigned by Demokratis """

    document_source: pd.CategoricalDtype = pa.Field(isin={"fedlex", "openparldata"})
    """ Where did we get the document metadata from? """

    document_source_url: str = pa.Field(str_matches=r"^https?://")
    """ URL of the original document on a cantonal or federal website. Beware that this URL may
    no longer be accessible as websites change over time. """

    document_type: pd.CategoricalDtype = pa.Field(nullable=True, isin=DOCUMENT_TYPES)
    """ The role of this document in the consultation process; may be unknown """

    document_language: pd.CategoricalDtype = pa.Field(isin={"de", "fr", "it", "rm"})
    """ Language of the document """

    document_title: str = pa.Field(
        nullable=True,  # Fedlex documents don't have titles (names)
    )
    """ Name of the document; may be an actual filename with an extension """


class FullConsultationDocumentSchemaV1(ConsultationDocumentMetadataSchemaV1):
    """Schema for a table that includes both the metadata and the text content of the documents."""

    document_content_plain: str
    """ Text content of the document in plain text, typically extracted from a PDF """
