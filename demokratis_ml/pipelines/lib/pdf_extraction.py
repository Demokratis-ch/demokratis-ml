"""Utility code for extracting text and other features from PDF files."""

import io
from typing import TypedDict

import pdfplumber
import pymupdf

PAGE_SEPARATOR = "\n\f"


class PDFExtractionError(Exception):
    """Wrapper exception for various errors thrown by the PDF extraction library."""


class BasicPDFFeatures(TypedDict):
    """Extra features we can easily extract from every PDF file, even if it's very large."""

    count_pages: int
    contains_table_on_first_page: bool


class ExtendedPDFFeatures(BasicPDFFeatures):
    """Extra features we can extract from a PDF file that has a reasonable size and can be processed page by page."""

    count_tables: int
    count_images: int
    count_pages_containing_tables: int
    count_pages_containing_images: int
    average_page_aspect_ratio: float


def extract_text_from_pdf(pdf_data: bytes) -> str:
    """Extract plain text from a PDF file.

    May raise a `PDFExtractionError` if the extraction fails.
    May return an empty string without raising an exception.
    """
    # https://github.com/pymupdf/PyMuPDF/issues/209#issuecomment-598274857
    pymupdf.TOOLS.mupdf_display_errors(False)

    try:
        document = pymupdf.Document(stream=pdf_data)
        pages = [page.get_text().strip() for page in document]
    except Exception as e:
        raise PDFExtractionError("Error extracting text from PDF", repr(e)) from e

    pages = [page for page in pages if page]  # Remove empty pages
    raw_text = PAGE_SEPARATOR.join(pages)
    return raw_text


def extract_features_from_pdf(pdf_data: bytes, max_pages_to_process: int) -> BasicPDFFeatures | ExtendedPDFFeatures:
    """Extract features from a PDF file.

    :param max_pages_to_process: If the PDF is longer than this, only :class:`BasicPDFFeatures` will be returned.
        For shorter PDFs we extract the full set of :class:`ExtendedPDFFeatures`.
    """
    try:
        # Open the PDF file
        with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
            basic_features: BasicPDFFeatures = {
                "contains_table_on_first_page": bool(pdf.pages[0].find_tables()),
                "count_pages": len(pdf.pages),
            }
            if basic_features["count_pages"] > max_pages_to_process:
                return basic_features

            tables = [page.find_tables() for page in pdf.pages]

            extended_features: ExtendedPDFFeatures = {
                **basic_features,
                "count_tables": sum(len(ts) for ts in tables),
                "count_pages_containing_tables": sum(bool(ts) for ts in tables),
                "count_images": sum(len(page.images) for page in pdf.pages),
                "count_pages_containing_images": sum(bool(page.images) for page in pdf.pages),
                "average_page_aspect_ratio": (
                    sum(page.width / page.height for page in pdf.pages) / basic_features["count_pages"]
                ),
            }
            return extended_features
    except Exception as e:
        raise PDFExtractionError("Error extracting features from PDF", repr(e)) from e
