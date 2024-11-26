"""Plain text extraction from PDF files using PyMuPDF."""

import pymupdf

PAGE_SEPARATOR = "\n\f"


class PDFExtractionError(Exception):
    """Wrapper exception for various errors thrown by the PDF extraction library."""


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
