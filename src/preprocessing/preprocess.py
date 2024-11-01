from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
import re
from typing import Literal


def format_pdf_text(text: str) -> str:
    """
    Formats raw PDF text extracted with a PDF library.

    This function processes the input text by performing the following steps:
    1. Marks word wraps by replacing instances of '- ' before newlines with a special marker.
    2. Merges text paragraphs by replacing newlines preceded by more than 50 characters and followed by a character with a single space.
    3. Removes the word wrap markers.
    4. Removes extra spaces that are followed by another space or a newline.
    5. Removes multiple consecutive newlines.

    Args:
        text (str): The raw text extracted from a PDF.

    Returns:
        str: The formatted text.
    """

    def mark_word_wrap(text):
        # If there is a '- ' before the newline, replace this string too
        return re.sub(r'(?<=.{65}\w)-\s*\n', '{{word_wrap}}\n', text)

    def merge_text_paragraphs(text):
        # Replace newline preceded by more than 50 characters and followed by a character with a single space
        return re.sub(r'(?<=.{65})\n(\S)', r' \1', text)

    def remove_word_wrap(text):
        return text.replace('{{word_wrap}} ', '')

    def remove_extra_spaces(text):
        # Remove spaces followed by another space or a newline
        return re.sub(r' +(?=[ \n])', '', text)

    def remove_extra_newlines(text):
        # Remove multiple newlines
        return re.sub(r'\n+(?=\n{1})', '', text)

    return remove_extra_newlines(remove_extra_spaces(
        remove_word_wrap(merge_text_paragraphs(mark_word_wrap(text)))
    ))


def extract_text_from_pdf(pdf_path: str, pdf_library: Literal["pdfminer", "PyPDF2"]) -> str:
    """
    Extracts text from a PDF file using the specified PDF library.

    Args:
        pdf_path (str): The path to the PDF file.
        pdf_library (Literal["pdfminer", "PyPDF2"]): The PDF library to use for text extraction.

    Returns:
        str: The extracted and formatted text.
    """
    
    if pdf_library == "pdfminer":
        text_pdf = extract_text(pdf_path)
    elif pdf_library == "PyPDF2":
        reader = PdfReader(pdf_path)
        text_pdf = '\n\n'.join([p.extract_text() for p in reader.pages])
    
    text_pdf = format_pdf_text(text_pdf)

    return text_pdf