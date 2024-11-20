from typing import Dict, Any, Tuple, List
import difflib
import re
import json
import jsonschema
import os


def flatten_dict_and_extract_text(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Recursively flattens a nested dictionary and extracts text from nested lists and dictionaries.
    Args:
        d (dict): The dictionary to flatten.
        parent_key (str, optional): The base key to use for the flattened keys. Defaults to ''.
        sep (str, optional): The separator to use between keys. Defaults to '_'.
    Returns:
        dict: A flattened dictionary with keys representing the nested structure and values being the extracted text or values.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict_and_extract_text(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(flatten_dict_and_extract_text(item, f"{new_key}{sep}{i}", sep=sep).items())
                else:
                    items.append((f"{new_key}{sep}{i}", item))
        else:
            items.append((new_key, v))
    return dict(items)


def remove_extra_newlines(text):
    # Remove multiple newlines
    return re.sub(r'\n+(?=\n{1})', '', text)


def extract_text_from_structured_dict_old(d: Dict[str, Any]) -> str:
    """
    Extracts and merges text values from a structured dictionary.
    This function flattens a nested dictionary and extracts text values. It then merges
    specific pairs of text values based on predefined key suffix combinations.
    Args:
        d (dict): The structured dictionary to extract text from.
    Returns:
        str: A single string containing the merged text values, separated by newlines.
    Note:
        The function merges text values based on the following key suffix combinations:
        - ("article", "title")
        - ("number", "text")
        - ("index", "text")
        - ("footnote_id", "title")
    """

    flattened_dict = flatten_dict_and_extract_text(d)
    text_values = [str(v) for v in flattened_dict.values()]

    merge_combinations = [
        ("article", "title"),
        ("number", "text"),
        ("index", "text"),
        ("footnote_id", "title"),
    ]
    text_values = []
    for i, (k, v) in enumerate(flattened_dict.items()):
        key_suffix = k.split("_")[-1]
        prev_key_suffix = None
        if i > 1:
            prev_key = list(flattened_dict.keys())[i - 1]
            prev_key_suffix = prev_key.split("_")[-1]
        if (prev_key_suffix, key_suffix) in merge_combinations:
            text_values[-1] = f"{text_values[-1]} {v}".strip()
        else:
            text_values.append(v)

    return "\n".join(text_values)


def extract_text_from_structured_dict(d: Dict[str, Any]) -> str:
    """
    Extracts and merges text values from a structured dictionary.
    This function flattens a nested dictionary and extracts text values. It then merges
    specific pairs of text values based on predefined key suffix combinations.
    Args:
        d (dict): The structured dictionary to extract text from.
    Returns:
        str: A single string containing the merged text values, separated by newlines.
    Note:
        The function merges text values based on the following key suffix combinations:
        - ("article", "title")
        - ("number", "text")
        - ("index", "text")
        - ("footnote_id", "title")
    """

    flattened_dict = flatten_dict_and_extract_text(d)
    flattened_dict = {
        k: v for k, v in flattened_dict.items() 
        if "content" in k or "label" in k or "type" in k
        }

    text_values = []
    for i, (k, v) in enumerate(flattened_dict.items()):
        if i > 2:
            prev2_key = list(flattened_dict.keys())[i - 2]
            prev_key = list(flattened_dict.keys())[i - 1]
            if "label" in prev2_key and flattened_dict[prev_key]!='heading' and "content" in k:
                text_values.append(f"{flattened_dict[prev2_key]} {v}".strip())
            elif "label" in prev2_key and flattened_dict[prev_key]=='heading' and "content" in k:
                text_values.append(str(v))
        else:
            text_values.append(str(v))
    return "\n".join(text_values)


def percnt_missing_and_added_characters(parsed_dicts: List, extracted_texts: List[str]) -> Tuple[List, List]:
    """
    Evaluates the text extraction from a structured dictionary by comparing it with the expected text.
    Args:
        parsed_dict (Dict[str, Any]): The structured dictionary containing the sections to extract text from.
        extracted_text (str): The expected text to compare against the extracted text.
    Returns:
        Tuple[int, int]: A tuple containing the number of missing characters and the number of added characters.
    """

    percnt_missing_characters, percnt_added_characters = [], []

    for parsed_dict, extracted_text in zip(parsed_dicts, extracted_texts):
        text_res_flat = extract_text_from_structured_dict(parsed_dict)

        d = difflib.Differ()
        diff = list(d.compare(
            remove_extra_newlines(extracted_text).splitlines(), 
            remove_extra_newlines(text_res_flat).splitlines()
        ))

        text_len = len(extracted_texts)
        num_missing_characters = sum(
                [len(line)-2 for i, line in enumerate(diff) 
                if line.startswith('- ') and (not diff[i+1].startswith('? ') if i<len(diff)-1 else True)]
                ) + sum(
                    [sum(c == '-' for c in line) for line in diff if line.startswith('? ')]
                    )

        num_added_characters = sum(
                [len(line)-2 for i, line in enumerate(diff) 
                if line.startswith('+ ') and (not diff[i-1].startswith('? ') if i>0 else True)]
                ) + sum(
                    [sum(c == '^' for c in line) for line in diff if line.startswith('? ')]
                    )
        
        percnt_missing_characters.append(num_missing_characters / text_len)
        percnt_added_characters.append(num_added_characters / text_len)

    return percnt_missing_characters, percnt_added_characters


def calculate_similarity_score(parsed_dicts: List, extracted_texts: List[str]) -> List[float]:
    """
    Compares the text extracted from a structured dictionary with the expected text using difflib's SequenceMatcher.
    Args:
        parsed_dict (Dict[str, Any]): The structured dictionary containing the sections to extract text from.
        extracted_text (str): The expected text to compare against the extracted text.
    Returns:
        float: A similarity score between 0 and 1, where 1 indicates a perfect match.
    """
    similarities = []
    for parsed_dict, extracted_text in zip(parsed_dicts, extracted_texts):
        text_res_flat = extract_text_from_structured_dict(parsed_dict)
        matcher = difflib.SequenceMatcher(
            None, 
            remove_extra_newlines(extracted_text).splitlines(), 
            remove_extra_newlines(text_res_flat).splitlines()
        )
        similarities.append(matcher.ratio())

    return similarities


def compare_texts_html(
        parsed_dicts: List,
        extracted_texts: List[str], 
        # html_paths: List[str]
        ) -> List[str]:
    """
    Compares the extracted text with the text parsed from a structured dictionary and generates an HTML file highlighting the differences.
    Args:
        extracted_text (str): The text extracted from the source to be compared.
        parsed_dict (Dict[str, Any]): The structured dictionary containing parsed sections.
        html_path (str): The file path where the HTML diff output will be saved.
    Returns:
        None
    """

    html_diffs = []
    for parsed_dict, extracted_text in zip(parsed_dicts, extracted_texts):
        text_res_flat = extract_text_from_structured_dict(parsed_dict)

        d = difflib.HtmlDiff()
        html_diff = d.make_file(
            remove_extra_newlines(extracted_text).splitlines(), 
            remove_extra_newlines(text_res_flat).splitlines()
        )

        # Modify the HTML table style max column width of 500px
        html_diff = html_diff.replace(
            """.diff_chg {background-color:#ffff77}
        .diff_sub {background-color:#ffaaaa}""", 
        """.diff_chg {background-color:#ffff77}
        .diff_sub {background-color:#ffaaaa}
        .max-width-column {
            max-width: 500px;
            word-wrap: break-word;
        }""")
        html_diff = html_diff.replace('nowrap="nowrap"', 'class="max-width-column"')
        html_diffs.append(html_diff)

        # with open(html_path, "w", encoding="utf-8") as f:
        #     f.write(html_diff)

    return html_diffs


def get_costs(model: str, input_tokens: List[int], output_tokens: List[int]) -> float:
    # Define the pricing per 1,000 tokens (example pricing, check OpenAI pricing page for actual values)
    pricing_per_1M_input_tokens = {
        "gpt-4o": 2.50,
        "gpt-4o-2024-08-06": 2.50,
        "gpt-4o-mini": 0.150,
        "gpt-4o-mini-2024-07-18": 0.150
    }
    pricing_per_1M_output_tokens = {
        "gpt-4o": 10.00,
        "gpt-4o-2024-08-06": 10.00,
        "gpt-4o-mini": 0.600,
        "gpt-4o-mini-2024-07-18": 0.600
    }

    costs = []
    for input_token, output_token in zip(input_tokens, output_tokens):
        cost_per_input_token = pricing_per_1M_input_tokens.get(model, 0) / 1000000
        cost_per_output_token = pricing_per_1M_output_tokens.get(model, 0) / 1000000
        total_cost = input_token * cost_per_input_token + output_token * cost_per_output_token
        costs.append(total_cost)

    return costs


def validate_json_schema(parsed_dicts: List[Dict[str, Any]]) -> List[bool]:
    """
    Validates the JSON schema of the parsed dictionaries.
    Args:
        parsed_dicts (List[Dict[str, Any]]): A list of parsed dictionaries to validate.
    Returns:
        List[bool]: A list of boolean values indicating whether the JSON schema is valid for each parsed dictionary.
    """
    
    # Open and load the schema JSON file
    path = os.path.join(os.path.dirname(__file__), 'json_schema.json')
    with open(path, 'r') as file:
        schema = json.load(file)

    result = []
    for parsed_dict in parsed_dicts:
        try:
            jsonschema.validate(parsed_dict, schema)
            result.append(True)
        except jsonschema.exceptions.ValidationError:
            result.append(False)

    return result


