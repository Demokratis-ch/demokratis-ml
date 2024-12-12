from llama_parse import LlamaParse
from typing import List, Dict, Any, Tuple
from mlflow.pyfunc import PythonModel
from mlflow.models import set_model
import copy
from llama_index.core.node_parser import MarkdownNodeParser
import re
from pathlib import Path


class LlamaParseMarkdownParser(PythonModel):
    def __init__(self, results_dir: str = None):
        self.results_dir = results_dir

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
        self.parser = LlamaParse(**self.params)


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
            # Match list items
            list_item_match = re.match(r'( *)[-*] (\[.*?\] *)?(.*)', line)
            if not list_item_match:
                if list_stack:
                    if document["children"]:
                        document["children"][-1]["children"].append(list_stack[0])
                    else:
                        document["children"].append(list_stack[0])
                    list_stack = []
                    depth = 0
            else:
                indent = len(list_item_match.group(1))
                label = list_item_match.group(2).strip('[] ') if list_item_match.group(2) else ""
                content = list_item_match.group(3)

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
                    list_stack.pop()

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

        return document
    

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
        for node in nodes:
            parsed_node = self.parse_markdown_to_json_schema(node.text)
            parsed_doc["children"].extend(parsed_node["children"])
        
        return parsed_doc


    def predict(self, context, model_input: List[str]) -> Dict[str, Any]:
        # Model input is a list of file paths
        
        parsed_files = []
        markdown_texts = []

        # TODO: This loop can be parallelized with asyncio

        async def parse_file(file_path):
            # Define the path to the intermediate results and output files
            base_path = Path(self.results_dir) if self.results_dir else Path("..")
            filename = Path(file_path).stem
            md_llamaparse_path = base_path / "intermediate_results" / f"{filename}_llamaparse.md"
            md_footnotes_path = base_path / "intermediate_results" / f"{filename}_footnotes.md"
            json_schema_path = base_path / "sample-outputs" / f"{filename}_json_schema.json"

            # Parse the file to markdown pages with LlamaParse
            if md_llamaparse_path.exists():
                with open(md_llamaparse_path, 'r', encoding='utf-8') as f:
                    markdown = f.read()
            else:
                documents = await self.parser.aload_data(file_path)
                markdown = "\n".join([doc.text for doc in documents])
                with open(md_llamaparse_path, 'w', encoding='utf-8') as f:
                    f.write(markdown)
            documents = await self.parser.aload_data(file_path)

            return await self.parse_markdown_to_json_schema(node.text)
        
        semaphore = asyncio.Semaphore(5)  # Limit to 10 concurrent tasks
        async def parse_node_with_semaphore(node):
            async with semaphore:
                return await parse_node(node)
        
        tasks = [parse_node_with_semaphore(node) for node in nodes]
        
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(asyncio.gather(*tasks))
        parsed_nodes, num_input_tokens, num_output_tokens = zip(*results)            


        for file_path in model_input:
            # Parse the file to markdown pages with LlamaParse
            documents = self.parser.load_data(file_path)

            # Insert footnotes into the text
            documents = self.markdown_insert_footnotes_into_text(documents)
            
            markdown_texts.append("\n".join([doc.text for doc in documents]))

            # Extract nodes from the documents
            nodes = self.extract_nodes(documents)

            # Convert nodes to JSON schema
            json_schema = self.nodes_to_json(nodes)
            parsed_files.append(json_schema)

        return {
            "parsed_dicts": parsed_files, 
            "markdown_texts": markdown_texts
        }


# Specify which definition in this script represents the model instance
set_model(LlamaParseMarkdownParser())