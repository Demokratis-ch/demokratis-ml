import sys
import os

# Get the current working directory
current_dir = os.getcwd()
print(current_dir)

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from dotenv import load_dotenv
import os
from pathlib import Path
# import pandas as pd
import mlflow
from src.processing.process import extract_text_from_pdf
from src.evaluation import evaluate
import json
from typing import List



# Load environment variables from .env file
load_dotenv()
assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY environment variable must be set"
assert "LLAMA_CLOUD_API_KEY" in os.environ, "LLAMA_CLOUD_API_KEY environment variable must be set"


def load_file_paths(dir_path: str) -> List[str]:
    """
    Create a DataFrame with the file paths and the extracted text
    """
    dir_path = Path(dir_path)
    file_paths = list(dir_path.glob('**/*'))
    print("input files:")
    print([str(s) for s in file_paths])
    # df = pd.DataFrame([str(s) for s in file_paths], columns=["file_path"])

    return [str(s) for s in file_paths]


def evaluate_model(file_paths: List[str], experiment_name, model_path: str):
    """
    Evaluate the model on the extracted text
    """
    # Define the relative path for MLflow tracking
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    exp_info = mlflow.get_experiment_by_name(experiment_name)
    exp_id = exp_info.experiment_id if exp_info else mlflow.create_experiment(experiment_name)

    with mlflow.start_run(experiment_id=exp_id):
        # Log the model
        model_info = mlflow.pyfunc.log_model(
            python_model=model_path,
            artifact_path="model",
            #input_example=["../sample-documents/51276-de-DRAFT-92be4e18116eab4615da2a4279771eb05b4f47e2.pdf"],
        )

        # Load the model
        model = mlflow.pyfunc.load_model(model_info.model_uri)

        # Predict
        result_dict = model.predict(file_paths)

        parsed_dicts = result_dict["parsed_files"]
        markdown_texts = result_dict.get("markdown_texts", None)
        num_input_tokens = result_dict.get("num_input_tokens", None)
        num_output_tokens = result_dict.get("num_output_tokens", None)
        model_str = result_dict.get("model_str", None)

        # General
        mlflow.set_tag("description", model_path.split("/")[-1])
        mlflow.log_metric("num_files", len(file_paths))
        mlflow.log_metric("parsed_files", sum([p.get("status") != "failed" for p in parsed_dicts]))
        mlflow.log_metric("valid_schema", sum(evaluate.validate_json_schema(parsed_dicts)))
        
        # Costs
        if model_str and num_input_tokens and num_output_tokens:
            mlflow.log_metric("num_input_tokens", num_input_tokens)
            mlflow.log_metric("num_output_tokens", num_output_tokens)
            costs = evaluate.get_costs(model_str, num_input_tokens, num_output_tokens)
            mlflow.log_metric("costs", sum(costs))

        # Extract text from PDFs using PyPDF2 and pdfminer
        pypdf2_texts = [extract_text_from_pdf(p, "PyPDF2") for p in file_paths]
        pdfminer_texts = [extract_text_from_pdf(p, "pdfminer") for p in file_paths]
        
        # Compare parsed dictionaries with the extracted text 
        percnt_missing_characters, percnt_added_characters = evaluate.percnt_missing_and_added_characters(parsed_dicts, pypdf2_texts)
        mlflow.log_metric("avg_percnt_missing_chars_pypdf2", sum(percnt_missing_characters) / len(file_paths))
        mlflow.log_metric("avg_percnt_added_chars_pypdf2", sum(percnt_added_characters) / len(file_paths))
        percnt_missing_characters, percnt_added_characters = evaluate.percnt_missing_and_added_characters(parsed_dicts, pdfminer_texts)
        mlflow.log_metric("avg_percnt_missing_chars_pdfminer", sum(percnt_missing_characters) / len(file_paths))
        mlflow.log_metric("avg_percnt_added_chars_pdfminer", sum(percnt_added_characters) / len(file_paths))


        mlflow.log_artifact(model_path)
        
        html_diff_pypdf2 = evaluate.compare_texts_html(parsed_dicts, pypdf2_texts)
        html_diff_pdfminer = evaluate.compare_texts_html(parsed_dicts, pdfminer_texts)
        
        # Log the parsed dictionaries and HTML diffs
        for i in range(len(file_paths)):
            filename_parsed = file_paths[i].split('/')[-1].replace(".pdf", "_parsed.json")
            print(filename_parsed)
            mlflow.log_text(json.dumps(parsed_dicts[i], indent=4), filename_parsed)
            
            filename_md = file_paths[i].split('/')[-1].replace(".pdf", "_md.md")
            print(filename_md)
            mlflow.log_text(markdown_texts[i], filename_md)

            filename_pypdf2 = file_paths[i].split('/')[-1].replace(".pdf", "_pypdf2.html")
            print(filename_pypdf2)
            mlflow.log_text(html_diff_pypdf2[i], filename_pypdf2)

            filename_pdfminer = file_paths[i].split('/')[-1].replace(".pdf", "_pdfminer.html")
            print(filename_pdfminer)
            mlflow.log_text(html_diff_pdfminer[i], filename_pdfminer)
            


if __name__ == "__main__":
    # Load the file paths
    file_paths = load_file_paths("../sample-documents")

    # Evaluate the model
    evaluate_model(
        file_paths, 
        experiment_name="structure-extraction2",
        model_path="../models/llama_parse_markdown_model.py", 
    )