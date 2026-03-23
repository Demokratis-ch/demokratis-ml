import pandas as pd
from docling.document_converter import DocumentConverter

df = pd.read_parquet(
    # This is a dataset of all kinds of documents pertaining to Swiss
    # consultation procedures (Vernehmlassungsverfahren).
    "https://huggingface.co/datasets/demokratis/consultation-documents/resolve/main/consultation-documents-preprocessed.parquet",
    # Restrict the columns so that we don't have to download the entire 2 GB dataframe.
    columns=["consultation_start_date", "political_body", "document_type", "document_language", "document_source_url"],
)
df = df.loc[
    (df["document_language"] == "de")  # "fr" and "it" are also available
    & (df["political_body"] == "ch")  # filter for federal documents
    & (df["document_type"] == "DRAFT")  # filter for legal drafts - they have a very regular structure
    & (df["consultation_start_date"].dt.year >= 2010)  # look at recent documents only
]

document_url = df["document_source_url"].sample().iloc[0]
print("-" * 50)
print(document_url)
print("-" * 50)

converter = DocumentConverter()
result = converter.convert(document_url)
print("-" * 50)
print(result.document.export_to_html())
