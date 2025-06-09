"""Pandera schema definitions for the data used in the project."""

from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series

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
    "PRESS_RELEASE",
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


class ConsultationInternalTag(TypedDict):
    """Consultation metadata internal to Demokratis.

    Used to track the manual review process of consultations.
    """

    name: str
    created_at: pd.Timestamp


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

    consultation_internal_tags: object
    """ Consultation metadata internal to Demokratis, mainly used to track the manual review process.
    The object type is actually ``list[ConsultationInternalTag]`` but Pandera doesn't support this yet."""

    # It would be great to have this validation but it's very slow - it takes several minutes on the whole dataset.
    # @pa.check("consultation_internal_tags")
    # def _check_consultation_internal_tags(cls, series: Series[object]) -> Series[bool]:
    #     return cast(
    #         Series[bool],
    #         series.map(
    #             lambda tags: isinstance(tags, list)
    #             and all(
    #                 tag.keys() == {"name", "created_at"} and isinstance(tag["created_at"], pd.Timestamp)
    #                 for tag in tags
    #             )
    #         ),
    #     )

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

    document_publication_date: pd.Timestamp = pa.Field(nullable=True)
    """ Date when the document was published or made available to Demokratis. """

    document_type: pd.CategoricalDtype = pa.Field(nullable=True, isin=DOCUMENT_TYPES)
    """ The role of this document in the consultation process; may be unknown """

    document_type_label_source: pd.CategoricalDtype = pa.Field(nullable=True, isin={"fedlex", "rule", "manual"})
    """ Where did the document type label come from?
    - "fedlex": The document type was provided by Fedlex (it provides it for all documents).
    - "rule": The document type was assigned by a rule-based model using the document title.
    - "manual": The document type was assigned by a human reviewer; this might override the other type sources.
    - None: The document type is unknown, e.g. because it comes from OpenParlData which doesn't provide types.
    """

    document_language: pd.CategoricalDtype = pa.Field(isin={"de", "fr", "it", "rm"})
    """ Language of the document """

    document_title: str = pa.Field(
        nullable=True,  # Fedlex documents don't have titles (names)
    )
    """ Name of the document; may be an actual filename with an extension """

    latest_stored_file_id: pa.Int64 = pa.Field(nullable=True)
    """ ID of the latest file that was stored in the platform file storage. Points to the latest version
    of the document. """


ConsultationDocumentMetadataV1 = DataFrame[ConsultationDocumentMetadataSchemaV1]


class FullConsultationDocumentSchemaV1(ConsultationDocumentMetadataSchemaV1):
    """Schema for a table that includes metadata, stored file attributes, and the text content of documents."""

    stored_file_path: str = pa.Field(
        nullable=True,
        # Expected format: {year_downloaded}/{consultation_id}/{document_id}/{random_uuid}.{ext}
        str_matches=r"^\d{4}/\d+/\d+/[A-Za-z0-9]+(\.[a-z0-9]+)?$",
    )
    """ Path to the stored file in the platform file storage. This path doesn't include the schema and bucket name. """

    stored_file_mime_type: str = pa.Field(nullable=True)
    """ MIME type of the stored file """

    stored_file_hash: str = pa.Field(nullable=True)
    """ SHA1 hash of the stored file's contents """

    document_content_plain: str
    """ Text content of the document in plain text, typically extracted from a PDF """


FullConsultationDocumentV1 = DataFrame[FullConsultationDocumentSchemaV1]


def get_allowed_values(schema_cls: type[pa.DataFrameModel], field_name: str) -> set[str]:
    """Get the allowed values for a field in a Pandera schema."""
    schema = schema_cls.to_schema()
    column = schema.columns[field_name]
    for check in column.checks:
        if check.name == "isin":
            return check.statistics["allowed_values"]
    msg = f"Field '{field_name}' in schema '{schema_cls.__name__}' does not have an 'isin' check."
    raise ValueError(msg)
