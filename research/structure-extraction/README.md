# Problem II. Extracting structure from documents

## Using an LLM such as GPT-4o
This seems to be a promising approach but we need to work on the prompts, and we definitely need some checks that the prompts are being followed.

This is an example of a prompt that *partially* works with GPT-4o and federal consultations:

```
This PDF is a Swiss German document proposing an amendment to a federal law, or introducing a new federal law. Please analyse the document structure and return it in JSON format.
- The structure (or outline) should be a hierarchy of titles, sections, articles etc. There may also be one or more appendices.
- Make sure you don't skip any headings and text inside paragraphs.
- Please itemize each paragraph inside the articles as well.
- For each article, put its title separately from the actual paragraphs.
- For each paragraph, include its number or letter (as in the original document) in a separate JSON item.
- When there is a letter-indexed list inside of a paragraph, break out the list items as children of the paragraph. Make sure to place the index (letter or number) separately from the list item text.
- Place footnotes in their own JSON element. Make sure you retain the reference to the footnote in the text, and that there is a machine-readable ID which identifies the footnote element.
```

For the sample document [51276-de-DRAFT-92be4e18116eab4615da2a4279771eb05b4f47e2.pdf](./sample-documents/51276-de-DRAFT-92be4e18116eab4615da2a4279771eb05b4f47e2.pdf) this produces a [partially correct output](./sample-outputs/gpt4o-v1-51276-de-DRAFT-92be4e18116eab4615da2a4279771eb05b4f47e2.json) with several problems. In particular, parts of the document are missing and footnotes are not really referenced and are incomplete.

## Evaluation
For evaluation, different metrics and artifacts are logged to mlflow.
From the parsed structured json a text file is generaated that is compared
to direct text extractions from the PDF (with libraries pdfminer and PyPdf2).
This comparison allows to check if important texts parts are missing,
or hallucinated text has been added by the parsing model. 
For a detailed comparison every PDF extraction needs to be checked manually
using the HTML-Diff file (see below).

Metrics:
* `avg_percnt_added_chars`: Average of characters added by the parsing algorithm
not existing in the extracted PDF texts.
* `avg_percnt_missing_chars`: Average of characters missing in the parsing algorithm
result compared to the extracted PDF texts. Some parts of the PDF texts might be 
irrelevant, so some missing character might be correct.
* `cost`: Cost of external parsing API's (e.g. openai)
* `parsed_files`: Number of successfully parsed files. Sometimes errors occur
in the external API calls.

Artifacts:
* `..._parsed.json`: Parsed output for each input PDF.
* `..._pdfminer.html`: HTML-Diff file for each input PDF comparing the parsed
text with directly extracted text. These files can be used to check for missing
or hallucinated text parts.
* `..._model.py`: Saved model for each run. Saved using mlflow models from code
[https://mlflow.org/docs/latest/models.html#models-from-code]


## Conclusion
So far, the structure extraction with ChatGPT-File upload (see file
`research/structure-extraction/scripts/chatgpt_file_upload.ipynb`) 
doesn't work stable. Every run gives different results, some really good,
others not so good. The PDF from Kanton ZH could never been parsed so far.
Other extraction methods should be investigated, e.g.:
* Nuextract: https://huggingface.co/learn/cookbook/en/information_extraction_haystack_nuextract
* LlamaParse: https://docs.llamaindex.ai/en/stable/llama_cloud/llama_parse/
* PyMuPDF4LLM: https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/index.html
* ...

