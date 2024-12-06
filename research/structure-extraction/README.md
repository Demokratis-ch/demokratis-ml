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

## Evaluation Tracking
We use MLflow to track each test run of a specific model with specific parameters.
1. Run local MLflow tracking server in terminal, this creates and writes to the directories `research/mlruns` and `research/mlartifacts`:
````
cd research && mlflow ui
````
2. Run model evaluations with Jupyter Notebook, the evaluation results are loged to the local MLflow tracking server:
````
research/structure-extraction/scripts/run_model_evaluation.ipynb
````
3. Current and old evaluation results can be inspected and compared by opening the MLflow-GUI in a browser: `http://localhost:5000/` or `http://127.0.0.1:5000/`.

## Evaluation Methods
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
* `..._pdfminer.html` and `...pypdf2.html`: HTML-Diff file for each input PDF comparing the parsed
text with directly extracted text. These files can be used to check for missing
or hallucinated text parts.
* `..._model.py`: Saved model for each run. Saved using mlflow models from code
[https://mlflow.org/docs/latest/models.html#models-from-code]


## Parsing Models
### 1 - ChatGPT-File upload
Pipleline:
1. Uploading PDF files ChatGPT assistant API: https://platform.openai.com/docs/assistants/tools/file-search
2. Asking ChatGPT to generate the desired output structure from the uploaded files.

Notes:
- So far, the structure extraction with ChatGPT-File upload 
doesn't work stable. Every run gives different results, some really good,
others not so good. 
- Structured output with JSON-schema doesn't seem to be supported for assistant API with file-search so far: https://community.openai.com/t/structured-outputs-with-assistants/900658/15
- The PDF from Kanton ZH could never been parsed so far.

### 2 - LlamaParse and manual parsing (BEST pipeline so far)
Pipeline:
1. Using LlamaParse for extracting markdown from PDF's,
2. splitting markdown into nodes using `llama_index.core.node_parser.MarkdownNodeParser`
3. parsing each node into the structured output with a manual python function,
4. merging all parsed nodes to one document.

Notes:
- So far, the most stable approach. 
- Creating the correct output structure manually is quite difficult. There are still some parsing problems, e.g., with nested lists. --> Could be improved with extra brain work...

### 3 - LlamaParse and ChatGPT parsing
Pipline: Same as above, but using ChatGPT in step 3. instead of manual python function for parsing each node.

Notes:
- Time and cost are about twice the direct ChatGPT approach.
- Now, with the completions API we can use strucuted outputs to submit the JSON-schema to ChatGPT: https://platform.openai.com/docs/guides/structured-outputs
- Still, the parsed structure doen't seem very stable to me...

### Further possible investigations
Other extraction methods should be investigated, e.g.:
- LlamaParse JSON response and continous_mode: I have tried the request so far at the bottom of the script `research/structure-extraction/scripts/run_model_evaluation.ipynb`, section "Trying out stuff..."
- Nuextract: https://huggingface.co/learn/cookbook/en/information_extraction_haystack_nuextract
- Unstructured: https://huggingface.co/learn/cookbook/rag_with_unstructured_data
- MinerU: https://github.com/opendatalab/MinerU 
- ...

