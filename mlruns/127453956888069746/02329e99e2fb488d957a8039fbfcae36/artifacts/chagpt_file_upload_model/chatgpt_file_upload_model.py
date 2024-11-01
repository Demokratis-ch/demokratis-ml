
import os
from openai import OpenAI
from openai import files
from typing import List, Dict, Any
from mlflow.pyfunc import PythonModel
from mlflow.models import set_model
import multiprocessing
import json
import time


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Create assistant
pdf_assistant = client.beta.assistants.create(
    name="PDF assistant",
    model="gpt-4o",
    description="An assistant to extract the contents of PDF files.",
    tools=[{"type": "file_search"}]
)

# Define the prompt
prompt = """Answer from file context:
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

def upload_file(filename: str) -> Dict[str, Any]:
    # Upload the file to OpenAI
    file = client.files.create(
        file=open(filename, "rb"), 
        purpose="assistants"
    )
    return file

def delete_file(file_id):
    # Delete file from OpenAI
    try:
        response = files.delete(file_id)
        print(f"File {file_id} deleted successfully.")
    except Exception as e:
        print(f"Failed to delete file {file_id}: {e}")

def parse_file(file_path: str) -> Dict[str, str]:
    # Upload the file to OpenAI
    file = upload_file(file_path)

    # Create thread
    thread = client.beta.threads.create(
        messages=[
            {
            "role": "user",
            "content": prompt,
            # Attach the new file to the message.
            "attachments": [
                { "file_id": file.id, "tools": [{"type": "file_search"}] }
            ],
            }
        ]
    )

    # Run thread
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id, 
        assistant_id=pdf_assistant.id, 
        timeout=1000
    )

    while True:
    # Retrieve the run status
        run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        time.sleep(10)
        if run_status.status == 'completed':
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            break
        else:
            ### sleep again
            print("Waiting for the run to complete...")
            time.sleep(2)

    messages_cursor = client.beta.threads.messages.list(thread_id=thread.id)
    messages = [message for message in messages_cursor]

    # Output text
    text_res = messages[0].content[0].text.value

    # Convert text to JSON
    dict_res = json.loads(text_res)

    # Delete file from OpenAI
    delete_file(file.id)

    return dict_res


class ChatgptFileParsingManager(PythonModel):
    def predict(self, context, model_input: List[str]) -> List[Dict]:
        # Use multiprocessing to parse multiple files concurrently
        # with multiprocessing.Pool() as pool:
        #     results = pool.map(parse_file, model_input)
        
        results = []
        for file in model_input:
            results.append(parse_file(file))
        
        return results


# Specify which definition in this script represents the model instance
set_model(ChatgptFileParsingManager())
