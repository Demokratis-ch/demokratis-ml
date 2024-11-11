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
    "diplomacy",
    "economics",
    "education",
    "energy",
    "environment",
    "finance",
    "health",
    "insurance",
    "it",
    "law",
    "migration",
    "science",
    "security",
    "social",
    "spatial_planning",
    "sports",
    "transportation",
}

DOCUMENT_TYPES = {
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
}


class ConsultationDocumentSchemaV1(pa.DataFrameModel):
    """Schema for a denormalized document table.

    Each row represents a document belonging to a consultation, as well as some attributes
    of the consultation itself.
    """

    # --- Consultation attributes ---
    consultation_id: int
    """ ID of the consultation, assigned by Demokratis """
    consultation_title: str = pa.Field(str_length={"min_value": 3})
    """ Title of the consultation """
    consultation_description: str
    """ Short summary of the consultation, sometimes empty """
    consultation_url: str = pa.Field(str_startswith="https://www.demokratis.ch/")
    """ URL to the consultation page on Demokratis """
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    organisation_id: int
    """ ID of the organisation that published the consultation; ID is assigned by Demokratis """
    organisation_name: str = pa.Field(str_length={"min_value": 3})
    """ Name of the organisation that published the consultation """
    topics: np.object_
    """ Zero, one, or more topics that the consultation covers """

    @pa.check("topics")
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

    # --- Document attributes ---
    document_id: int
    """ ID of the document, assigned by Demokratis """
    doc_language: pd.CategoricalDtype = pa.Field(isin=("de", "fr", "it", "rm"))
    """ Language of the document """
    doc_name: str = pa.Field(
        nullable=True,  # Fedlex documents don't have names
    )
    """ Name of the document; may be an actual filename with an extension """
    document_url: str = pa.Field(str_startswith="https://")
    """ URL of the original document on a cantonal or federal website """
    doc_content: str
    """ Text content of the document, typically extracted from a PDF """
    doc_type: pd.CategoricalDtype = pa.Field(nullable=True, isin=DOCUMENT_TYPES)
    """ The role of this document in the consultation process; may be unknown """
