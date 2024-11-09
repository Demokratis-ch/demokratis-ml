import sys
import os

# Add the src directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)

import pytest
import json
from src.preprocessing.preprocess import parse_markdown_to_json_schema

def test_parse_markdown_to_json_schema():
    markdown = """
# Art. 1 Sachüberschrift, Abs. 1 und 1bis
Gegenstand und Geltungsbereich

- [1.] Dieses Gesetz regelt:
  - [a.] die Veranstaltung, die Aufbereitung, die Übertragung und den Empfang von Radio- und Fernsehprogrammen;
  - [b.] die Fördermassnahmen zugunsten der elektronischen Medien.
  - [abis.] blabla.
    """
    expected_output = {
        "label": "",
        "type": "document",
        "content": [],
        "children": [
            {
            "label": "1",
            "type": "heading",
            "content": [
                "Art. 1 Sach\u00fcberschrift, Abs. 1 und 1bis"
            ],
            "children": [
                {
                "label": "",
                "type": "content",
                "content": [
                    "Gegenstand und Geltungsbereich"
                ],
                "children": []
                },
                {
                "label": "",
                "type": "list",
                "content": [],
                "children": [
                    {
                    "label": "1.",
                    "type": "list_item",
                    "indent_level": 0,
                    "content": [
                        "Dieses Gesetz regelt:"
                    ],
                    "children": [
                        {
                        "label": "",
                        "type": "list",
                        "content": [],
                        "children": [
                            {
                            "label": "a.",
                            "type": "list_item",
                            "indent_level": 2,
                            "content": [
                                "die Veranstaltung, die Aufbereitung, die \u00dcbertragung und den Empfang von Radio- und Fernsehprogrammen;"
                            ],
                            "children": []
                            },
                            {
                            "label": "b.",
                            "type": "list_item",
                            "indent_level": 2,
                            "content": [
                                "die F\u00f6rdermassnahmen zugunsten der elektronischen Medien."
                            ],
                            "children": []
                            },
                            {
                            "label": "abis.",
                            "type": "list_item",
                            "indent_level": 2,
                            "content": [
                                "blabla."
                            ],
                            "children": []
                            }
                        ]
                        }
                    ]
                    }
                ]
                }
            ]
            }
        ]
    }

    # print(json.dumps(parse_markdown_to_json_schema(markdown), indent=4))

    result = parse_markdown_to_json_schema(markdown)
    assert result == expected_output

if __name__ == '__main__':
    pytest.main()
