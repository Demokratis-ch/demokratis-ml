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
