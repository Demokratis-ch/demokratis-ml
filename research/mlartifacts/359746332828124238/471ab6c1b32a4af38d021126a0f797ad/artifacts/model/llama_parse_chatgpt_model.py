from llama_parse import LlamaParse
from typing import List, Dict, Any, Tuple
from mlflow.pyfunc import PythonModel
from mlflow.models import set_model
import copy
from llama_index.core.node_parser import MarkdownNodeParser
import re
from openai import OpenAI, AsyncOpenAI
import asyncio
import os
import json



class LlamaParseMarkdownParser(PythonModel):
    def __init__(self):

        # Define parsing instructions for a document
        parsing_instruction = """
        This PDF is a German document proposing an amendment to a Swiss federal law, or introducing a new federal law.
        - Format all article titles as headers.
        - Combine multiple lines for each paragraph and list item into one line, also if words are split at the end of a line with "-".
        - List all paragraphs for each article as unordered lists. Set list indices in square brackets, where existing, e.g. "[1.], [a.], [abis.], ...".
        - Text on top of a new page might still belong to the article on the previous page. Merge this text with the previous page and don't invent artificial headers.
        - If there is a footnote section at the bottom of a page, start it with an extra header "# [Fussnoten]" and list footnotes section as unordered list. Each footnote should start with the identifier in square brackets with "^", e.g., "[^1], [^2], ...".
        - If there are references to footnotes within the text in the middle of a page, set the references in square brackets with "^", e.g., "[^1], [^2], ...". Sometimes the reference is following a number, e.g., "2006[^3], 1997[^4], ...".
        - Do not insert a note or any other text that doesn't exist in the document.
        """

        self.params = {
            "result_type": "markdown",  # "markdown" or "json" or "text"
            "verbose": True,
            "language": "de",
            "parsing_instruction": parsing_instruction,
            "page_separator": "\n=================\n",
            "bounding_box": "0.09,0,0.07,0",
            "take_screenshot": False,
            "premium_mode": True
        }

        # Initialize the LlamaParse parser
        self.pdf_parser = LlamaParse(**self.params)

        # Initialize the OpenAI client
        self.openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model_str = "gpt-4o-mini"
        self.md_parsing_prompt = """
        Parse the markdown text and return the structure in JSON format according to the submitted JSON schema file.
        - output json file only
        - put all content in the markdown text into the content field of the document node
        - don't add any extra content, such as comments or summaries
        - the document node has no label
        - the label of a heading node is the header level
        - the content of a heading node is the header title
        - replace footnotes in text with identifier, e.g., [^1], and also use this identifier as the label of a footnote
        - the label of a footnote node is the footnote identifier
        """
        self.json_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "DocNode",
            "type": "object",
            "properties": {
            "label": {
                "type": "string",
                "description": "Label of the node, e.g. '1.2' or '1bis' or 'a)' etc. May be empty."
            },
            "type": {
                "type": "string",
                "enum": ["document", "heading", "content", "list", "list_item", "footnote", "image"],
                "description": "Type of the node."
            },
            "content": {
                "type": "array",
                "items": {
                "type": "string"
                },
                "description": "Array of content strings. Usually only one string, but may be more in case of multiple paragraphs."
            },
            "children": {
                "type": "array",
                "items": {
                "$ref": "#"
                },
                "description": "Array of nested nodes."
            }
            },
            "required": ["label", "type", "content", "children"],
            "additionalProperties": False
        }


    def markdown_insert_footnotes_into_text(self, documents):
        """
        Replace footnote references in the text with the actual footnote text.

        Args:
            documents: List of Document objects with markdown text
        
        Returns:
            documents: List of Document objects with footnotes inserted into the text
        """

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
        
        return documents2


    def extract_nodes(self, documents):
        """
        Merge nodes that don't have a header to the previous node.

        Args:
            nodes: List of Node objects

        Returns:
            merged_nodes: List of Node objects with merged text
        """
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
        
        return merged_nodes


    async def parse_markdown_to_json_schema(self, markdown):
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

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model_str,
                messages=[
                    {"role": "system", "content": self.md_parsing_prompt},
                    {"role": "user", "content": markdown}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "pdf_structure",
                        "strict": True,
                        "schema": self.json_schema
                    }
                },
                # strict=True
            )
        except Exception as e:
            # handle errors like finish_reason, refusal, content_filter, etc.
            print(f"Failed to create completion: {e}")

        return json.loads(response.choices[0].message.content), response.usage.prompt_tokens, response.usage.completion_tokens


    def nodes_to_json(self, nodes):
        """
        Convert a list of Node objects to a JSON schema.

        Args:
            nodes: List of Node objects

        Returns:
            json_schema: JSON schema representation of the nodes
        """
        parsed_doc = {
        "label": "",
        "type": "document",
        "content": [],
        "children": []
        }
        
        async def parse_node(node):
            return await self.parse_markdown_to_json_schema(node.text)
        
        semaphore = asyncio.Semaphore(5)  # Limit to 10 concurrent tasks
        async def parse_node_with_semaphore(node):
            async with semaphore:
                return await parse_node(node)
        
        tasks = [parse_node_with_semaphore(node) for node in nodes]
        
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(asyncio.gather(*tasks))
        parsed_nodes, num_input_tokens, num_output_tokens = zip(*results)            

        # print(parsed_nodes)

        for parsed_node in parsed_nodes:
            parsed_doc["children"].extend(parsed_node["children"])
        
        # for node in nodes:
        #     parsed_node = self.parse_markdown_to_json_schema(node.text)
        #     parsed_doc["children"].extend(parsed_node["children"])
        
        return parsed_doc, sum(num_input_tokens), sum(num_output_tokens)


    def predict(self, context, model_input: List[str]) -> Dict[str, Any]:
        # Model input is a list of file paths
        
        parsed_files = []
        markdown_texts = []
        num_input_tokens_tot = []
        num_output_tokens_tot = []
        for file_path in model_input:
            # Parse the documents (pages) for file
            documents = self.pdf_parser.load_data(file_path)

            # Insert footnotes into the text
            documents = self.markdown_insert_footnotes_into_text(documents)
            
            markdown_texts.append("\n".join([doc.text for doc in documents]))

            # Extract nodes from the documents
            nodes = self.extract_nodes(documents)

            # Convert nodes to JSON schema
            json_schema, num_input_tokens, num_output_tokens = self.nodes_to_json(nodes)
            parsed_files.append(json_schema)
            num_input_tokens_tot.append(num_input_tokens)
            num_output_tokens_tot.append(num_output_tokens)

        return {
            "parsed_dicts": parsed_files, 
            "markdown_texts": markdown_texts,
            "num_input_tokens": num_input_tokens_tot,
            "num_output_tokens": num_output_tokens_tot,
            "model_str": self.model_str
        }


# Specify which definition in this script represents the model instance
set_model(LlamaParseMarkdownParser())