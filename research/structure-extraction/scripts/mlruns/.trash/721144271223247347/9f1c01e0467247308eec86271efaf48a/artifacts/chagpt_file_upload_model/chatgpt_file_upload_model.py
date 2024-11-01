
import os
from openai import OpenAI, AsyncOpenAI
from openai import files
from typing import List, Dict, Any
from mlflow.pyfunc import PythonModel
from mlflow.models import set_model
import asyncio
import json
import time




class ChatgptFileParsing(PythonModel):
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model_str = "gpt-4o"

        # Define the prompt
        self.prompt = """Answer from file context:
        This PDF is a German document proposing an amendment to a Swiss federal law, or introducing a new federal law. 
        Parse the document structure and return it in JSON format.
        - Respond with only JSON without using markdown code blocks.
        - The structure (or outline) should be a hierarchy of titles, sections, articles etc. There may also be one or more appendices.
        - A section title consists of a roman numeral.
        - Make sure you don't skip any headings and text inside paragraphs.
        - Itemize each paragraph inside the articles as well.
        - For each article, put its title separately from the actual paragraphs.
        - For each paragraph, include its number or letter (as in the original document) in a separate JSON item.
        - When there is a letter-indexed list inside of a paragraph, break out the list items as children of the paragraph. Make sure to place the index (letter or number) separately from the list item text.
        - Place footnotes in their own JSON element. Replace the references to footnotes in the text with '{{footnote_id}}'.
        - List all sections in the document.
        - List all articles for each section.
        - List all paragraphs for each article.
        - Return a valid json. Don't fill in placeholders like "// List articles under this section".

        JSON structure:
        {
            "document_title": "",
            "amendment": "",
            "sections": [
                {
                "section": "",
                "articles": [
                    "article": "",
                    "title": "",
                    "minorities": "",
                    "text": "",
                    "paragraphs": [
                    {
                        "number": "",
                        "text": "",
                        "list": [
                        {
                            "index": "",
                            "text": "",
                        },
                        ]
                    },
                    ],
                ],
                },
            ],
            "footnotes": [
                {
                "footnote_id": "",
                "text": "",
                },
            ],
        }
        """

    async def upload_file(self, filename: str) -> Dict[str, Any]:
        # Upload the file to OpenAI
        file = await self.client.files.create(
            file=open(filename, "rb"), 
            purpose="assistants"
        )
        return file

    async def delete_file(self, file_id):
        # Delete file from OpenAI
        try:
            response = await self.client.files.delete(file_id)
            print(f"File {file_id} deleted successfully.")
        except Exception as e:
            print(f"Failed to delete file {file_id}: {e}")

    async def parse_file(self, file_path: str, semaphore: asyncio.Semaphore) -> Dict[str, str]:
        async with semaphore:
            # Upload the file to OpenAI
            print(f"Uploading file {file_path}...")
            file = await self.upload_file(file_path)

            # Create assistant
            pdf_assistant = await self.client.beta.assistants.create(
                name="PDF assistant",
                model=self.model_str,
                description="An assistant to extract the contents of PDF files.",
                tools=[{"type": "file_search"}]
            )

            # Create thread
            thread = await self.client.beta.threads.create(
                messages=[
                    {
                    "role": "user",
                    "content": self.prompt,
                    # Attach the new file to the message.
                    "attachments": [
                        { "file_id": file.id, "tools": [{"type": "file_search"}] }
                    ],
                    }
                ]
            )

            # Run thread
            print("Running thread...")
            run = await self.client.beta.threads.runs.create_and_poll(
                thread_id=thread.id, 
                assistant_id=pdf_assistant.id, 
                timeout=60
            )
            print("Thread completed.")

            run_status = await self.client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            print(run_status.status)
            if run_status.status != 'completed':
                return {'status': run_status.status}, run.usage.prompt_tokens, run.usage.completion_tokens

            # Get messages
            # messages = await self.client.beta.threads.messages.list(thread_id=thread.id)

            messages_cursor = await self.client.beta.threads.messages.list(thread_id=thread.id)
            messages = [message for message in messages_cursor]
            # print(messages)

            # Output text
            text_res = messages[0][1][0].content[0].text.value

            # Convert text to JSON
            dict_res = json.loads(text_res)
                                
            # Delete file from OpenAI
            await self.delete_file(file.id)

        return dict_res, run.usage.prompt_tokens, run.usage.completion_tokens


    async def predict(self, context, model_input: List[str]) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        tasks = [self.parse_file(filename, semaphore) for filename in model_input]

        results = await asyncio.gather(*tasks)

        parsed_dicts, num_input_tokens, num_output_tokens = zip(*results)            
        
        # # Use multiprocessing to parse multiple files concurrently
        # with multiprocessing.Pool() as pool:
        #     results = pool.map(parse_file, model_input)

        return parsed_dicts, num_input_tokens, num_output_tokens
        
        # results = []
        # num_input_tokens = []
        # num_output_tokens = []
        # for file in model_input:
        #     parsed_dict, num_input_token, num_output_token = parse_file(file)
        #     results.append(parsed_dict)
        #     num_input_tokens.append(num_input_token)
        #     num_output_tokens.append(num_output_token)
        
        # return results, num_input_tokens, num_output_tokens


# Specify which definition in this script represents the model instance
set_model(ChatgptFileParsing())
