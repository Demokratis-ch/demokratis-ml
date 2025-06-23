from llama_parse import LlamaParse
from typing import List, Dict, Any, Tuple
from mlflow.pyfunc import PythonModel
from mlflow.models import set_model
import copy
from llama_index.core.node_parser import MarkdownNodeParser
import re
from pathlib import Path
import json
import asyncio
import pickle


class LlamaParseMarkdownParser(PythonModel):
    def __init__(self, results_dir: str = None):
        self.results_dir = results_dir
        self.page_separator = "\n=================\n"

        # Define parsing instructions for a document
        parsing_instruction = """
        This PDF is a German document proposing an amendment to a Swiss federal law, or introducing a new federal law.
        - Format all article titles as headers.
        - Format bold text with "**" at the beginning and end of the text.
        - Format italic text with "*" at the beginning and end of the text.
        - Format underlined text with "_" at the beginning and end of the text.
        - Format strikethrough text with "~~" at the beginning and end of the text.
        - Combine multiple lines for each paragraph and list item into one line, also if words are split at the end of a line with "-".
        - List all paragraphs for each article as unordered lists. Set list indices in square brackets, where existing, e.g. "[1.], [a.], [abis.], ...".
        - Text on top of a new page can still belong to the article/header on the previous page. Don't invent artificial headers at the beginning of a new page.
        - If there is a footnote section at the bottom of a page, start it with an extra header "# [Fussnoten]" and list footnotes section as unordered list. Each footnote should start with the identifier in square brackets with "^", e.g., "[^1], [^2], ...".
        - If there are references to footnotes within the text in the middle of a page, set the references in square brackets with "^", e.g., "[^1], [^2], ...". Sometimes the reference is following a number, e.g., "2006[^3], 1997[^4], ...".
        - Don't insert helper text that doesn't exist in the document, like "Here's the full document content formatted as markdown:"
        - If there are small tables in the document, format them as markdown tables.
        - If there are large tables over whole pages, traverse them them from left to right and extract text into single text column.
        """

        self.params = {
            "result_type": "markdown",  # "markdown" or "json" or "text"
            "verbose": True,
            "language": "de",
            "parsing_instruction": parsing_instruction,
            "page_separator": self.page_separator,
            # "bounding_box": "0.09,0,0.07,0",
            "take_screenshot": False,
            "premium_mode": True
        }

        # Initialize the LlamaParse parser
        self.parser = LlamaParse(**self.params)

        # Initialize the raw text parser
        self.raw_text_parser = LlamaParse(result_type="text")


    async def parse_pdf_to_markdown(self, file_path, cached_result_path: str = None):
        """
        Parse a PDF file to markdown text.

        Args:
            file_path: Path to the PDF file
        
        Returns:
            documents: Array of Document objects with markdown text
        """

        # Check if parsed markdown file already exists in the intermediate results
        if cached_result_path.with_suffix('.pkl').exists() if cached_result_path else False:
            print(f"Loading cached result from {cached_result_path.with_suffix('.pkl')}")
            with open(cached_result_path.with_suffix('.pkl'), 'rb') as pkl_file:
                documents = pickle.load(pkl_file)
                return documents
        
        # Parse the file to markdown pages with LlamaParse
        documents = await self.parser.aload_data(file_path)

        if not documents or documents[0].text == "":
            raise Exception(f"No documents found in the PDF file: {file_path}")

        # Write the parsed markdown to cached result file
        if cached_result_path:
            print(f"Writing parsed markdown to {cached_result_path}")
            cached_result_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cached_result_path, 'w', encoding='utf-8') as f:
                markdown = self.page_separator.join([doc.text for doc in documents])
                f.write(markdown)
            # Save the parsed documents to a pickle file
            print(f"Writing parsed documents to {cached_result_path.with_suffix('.pkl')}")
            with open(cached_result_path.with_suffix('.pkl'), 'wb') as pkl_file:
                pickle.dump(documents, pkl_file)

        return documents


    async def parse_pdf_to_text(self, file_path, cached_result_path: str = None):
        """
        Parse a PDF file to raw text.

        Args:
            file_path: Path to the PDF file
        
        Returns:
            text: Raw text extracted from the PDF file
        """

        # Check if parsed text file already exists in the intermediate results
        if cached_result_path.exists() if cached_result_path else False:
            print(f"Loading cached result from {cached_result_path}")
            with open(cached_result_path, 'rb') as f:
                text = f.read()
                return text
        
        # Parse the file to raw text with LlamaParse
        documents = await self.raw_text_parser.aload_data(file_path)

        text = "\n".join([doc.text for doc in documents if doc.text != ""])

        if not text:
            raise Exception(f"No text found in the PDF file: {file_path}")

        # Write the parsed text to cached result file
        if cached_result_path:
            print(f"Writing parsed text to {cached_result_path}")
            cached_result_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cached_result_path, 'w', encoding='utf-8') as f:
                f.write(text)

        return text


    def markdown_insert_footnotes_into_text(self, documents, cached_result_path: str = None):
        """
        Replace footnote references in the text with the actual footnote text.

        Args:
            documents: List of Document objects with markdown text
        
        Returns:
            documents: List of Document objects with footnotes inserted into the text
        """

        # Insert footnotes into the text
        documents2 = copy.deepcopy(documents)
        for doc in documents2:
            # Split the text and footnotes
            text_footnotes = doc.text.split("# [Fussnoten]")
            text = text_footnotes[0].strip("\n")
            if len(text_footnotes) > 1:
                footnotes = text_footnotes[1]
            else:
                footnotes = ""
            
            # Replace footnote references in the text with the actual footnote text
            for line in footnotes.split("\n"):
                match = re.match(r'.*(\[\^.+\]) (.+)', line)
                if match:
                    fn_index = match.group(1)
                    fn_text = match.group(2)
                    text = text.replace(fn_index, f"[^{fn_text}]")
                    # print(fn_index, fn_text)
            doc.text = text

        # Write the parsed markdown to cached result file
        if cached_result_path:
            print(f"Writing parsed markdown with footnotes to {cached_result_path}")
            cached_result_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cached_result_path, 'w', encoding='utf-8') as f:
                markdown = self.page_separator.join([doc.text for doc in documents2])
                f.write(markdown)

        return documents2


    def parse_markdown_to_json_schema(self, markdown):
        """
        Parses a markdown string into a defined JSON schema structure.
        Args:
            markdown (str): The markdown string to be parsed.
        Returns:
            dict: A JSON schema representation of the markdown content.
                The structure includes:
                - "label": A string label for the node.
                - "type": The type of the node (e.g., "document", "heading", "list", "list_item", "content").
                - "content": A list containing the content of the node.
                - "children": A list of child nodes, each following the same structure.
        """
        lines = markdown.split('\n')
        document = {
            "label": "",
            "type": "document",
            "children": []
        }

        depth = 0
        list_stack = []

        for line in lines:
            # Match table rows
            table_row_match = re.match(r'^\|.*\|$', line)
            if table_row_match:
                if not document["children"] or document["children"][-1]["type"] != "table":
                    node = {
                        "label": "",
                        "type": "table",
                        "content": [],
                        "children": []
                    }
                    document["children"].append(node)
                document["children"][-1]["content"].append(line)
                continue

            # Match list items
            list_item_match = re.match(r'( *)([-*] )?(\[.*?\] *)?(.*)', line)
            if (
                not list_item_match
                or (list_item_match.group(2) is None and list_item_match.group(3) is None)
            ):
                if list_stack and (line.strip() != ""):
                    indent = len(list_item_match.group(1))
                    if depth > 0 and indent >= depth or depth == 0 and indent > 0:
                        # Additional line of the list item
                        list_stack[-1]["children"][-1]["content"].append(list_item_match.group(4))
                        continue
                    elif document["children"]:
                        # Add the list to the document
                        try:
                            document["children"][-1]["children"].append(list_stack[0])
                        except Exception as e:
                            print(e)
                            print(document["children"])
                            raise e
                    else:
                        document["children"].append(list_stack[0])
                    list_stack = []
                    depth = 0
            else:
                indent = len(list_item_match.group(1))
                label = list_item_match.group(3).strip('[] ') if list_item_match.group(3) else ""
                content = list_item_match.group(4)

                node = {
                    "label": label,
                    "type": "list_item",
                    "content": [content],
                    "children": []
                }

                list_node = {
                    "label": "",
                    "type": "list",
                    "children": []
                }

                if not list_stack:
                    list_stack.append(list_node)
                elif indent > depth:
                    list_stack[-1]["children"][-1]["children"].append(list_node)
                    list_stack.append(list_node)
                elif indent < depth:
                    while list_stack and indent < depth:
                        list_stack.pop()
                        depth -= 2

                try:
                    list_stack[-1]["children"].append(node)
                    depth = indent
                except Exception as e:
                    print(f"Error: {e}")
                    print(list_stack)
                    print(node)
                    list_stack.append(list_node)
                    list_stack[-1]["children"].append(node)
                    depth = indent
                continue

            line = line.strip()

            # Match headings
            heading_match = re.match(r'^(#+)\s+(.*)', line)
            if heading_match:
                node = {
                    "label": str(len(heading_match.group(1))),
                    "type": "heading",
                    "content": [heading_match.group(2)],
                    "children": []
                }
                document["children"].append(node)
                continue

            # Match content
            if line:
                node = {
                    "label": "",
                    "type": "content",
                    "content": [line],
                }
                if "children" in document["children"][-1] if document["children"] else False:
                    document["children"][-1]["children"].append(node)
                else:
                    document["children"].append(node)

        # If there are still list items in the stack, add them to the document
        if list_stack:
            if document["children"]:
                try:
                    document["children"][-1]["children"].append(list_stack[0])
                except Exception as e:
                    print(f"Error: {e}")
                    print(document["children"])
                    print(list_stack)
                    print(markdown)
                    raise e
            else:
                document["children"].append(list_stack[0])

        return document


    def parse_documents_to_json(self, documents, cached_result_path: str = None):
        """
        Merge nodes that don't have a header to the previous node.

        Args:
            documents: Array of Document objects with markdown text

        Returns:
            json_schema: JSON schema representation of the nodes
        """

        # Extract nodes from the documents
        parser = MarkdownNodeParser()
        nodes = parser.get_nodes_from_documents(documents)

        # Merge nodes that don't have a header to the previous node
        merged_nodes = []
        for node in nodes:
            if not node.text.startswith("#"):
                if merged_nodes:
                    merged_nodes[-1].text += "\n" + node.text
                else:
                    merged_nodes.append(node)
            else:
                merged_nodes.append(node)
                    
        # Convert nodes to JSON schema
        parsed_dict = {
        "label": "",
        "type": "document",
        "content": [],
        "children": []
        }
        for node in merged_nodes:
            parsed_node = self.parse_markdown_to_json_schema(node.text)
            parsed_dict["children"].extend(parsed_node["children"])

        # Write the parsed json to cached result file
        if cached_result_path:
            print(f"Writing parsed JSON to {cached_result_path}")
            cached_result_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cached_result_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_dict, f, indent=4)

        return parsed_dict


    def predict(self, context, model_input: List[str]) -> Dict[str, Any]:
        """
        Parse the input PDF files to JSON schema.

        Args:
            model_input: List of paths to the input PDF files

        Returns:
            parsed_files: List of dicts with JSON schema representations of the input PDF files
        """
        
        async def parse_file(file_path):
            # Define the path to the intermediate results and output files
            base_path = Path(self.results_dir) if self.results_dir else Path("..")
            input_path = base_path / "sample-documents" / file_path
            text_path = base_path / "intermediate_results" / file_path.replace(".pdf", "_text.txt")
            md_llamaparse_path = base_path / "intermediate_results" / file_path.replace(".pdf", "_llamaparse.md")
            md_footnotes_path = base_path / "intermediate_results" / file_path.replace(".pdf", "_footnotes.md")
            json_schema_path = base_path / "sample-outputs" / file_path.replace(".pdf", "_json_schema.json")

            # Parse the file to markdown pages with LlamaParse
            documents = await self.parse_pdf_to_markdown(input_path, md_llamaparse_path)
            
            # Parse the file to raw text with LlamaParse
            # raw_text = await self.parse_pdf_to_text(input_path, text_path)

            # Insert footnotes into the text
            documents = self.markdown_insert_footnotes_into_text(documents, md_footnotes_path)

            # Convert documents to JSON schema
            parsed_dict = self.parse_documents_to_json(documents, json_schema_path)

            return parsed_dict
        
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent tasks
        async def parse_file_with_semaphore(file_path):
            async with semaphore:
                return await parse_file(file_path)
        
        tasks = [parse_file_with_semaphore(file_path) for file_path in model_input]
        
        loop = asyncio.get_event_loop()
        parsed_files = loop.run_until_complete(asyncio.gather(*tasks))

        # Load intermediate results
        # raw_texts = []
        parsed_markdowns_llamaparse = []
        parsed_markdowns = []
        for file_path in model_input:
            base_path = Path(self.results_dir) if self.results_dir else Path("..")
            text_path = base_path / "intermediate_results" / file_path.replace(".pdf", "_text.txt")
            md_llamaparse_path = base_path / "intermediate_results" / file_path.replace(".pdf", "_llamaparse.md")
            md_footnotes_path = base_path / "intermediate_results" / file_path.replace(".pdf", "_footnotes.md")
            # if text_path.exists():
            #     with open(text_path, 'r', encoding='utf-8') as f:
            #         raw_texts.append(f.read())
            if md_llamaparse_path.exists():
                with open(md_llamaparse_path, 'r', encoding='utf-8') as f:
                    parsed_markdowns_llamaparse.append(f.read())
            if md_footnotes_path.exists():
                with open(md_footnotes_path, 'r', encoding='utf-8') as f:
                    parsed_markdowns.append(f.read())

        return {
            "parsed_files": parsed_files,
            # "raw_texts": raw_texts,
            "parsed_markdowns": parsed_markdowns,
            "parsed_markdowns_llamaparse": parsed_markdowns_llamaparse
        }


# Specify which definition in this script represents the model instance
set_model(LlamaParseMarkdownParser())