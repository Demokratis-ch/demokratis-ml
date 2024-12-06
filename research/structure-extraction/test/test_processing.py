import sys
import os

# Add the src directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)

import pytest
import json
from models.llama_parse_markdown_model import LlamaParseMarkdownParser



def test_parse_markdown_to_json_schema():
    model = LlamaParseMarkdownParser()

    test_cases = [
        {
            "markdown": """
# Art. 1 Sachüberschrift, Abs. 1 und 1bis
Gegenstand und Geltungsbereich

- [1.] Dieses Gesetz regelt:
  - [a.] die Veranstaltung, die Aufbereitung, die Übertragung und den Empfang von Radio- und Fernsehprogrammen;
  - [b.] die Fördermassnahmen zugunsten der elektronischen Medien.
  - [abis.] blabla.
- [2.] Es gilt für:
  - [a.] die Veranstalter von Radio- und Fernsehprogrammen;

Einfügen der Art. 76a-76c vor dem Gliederungstitel des 4. Kapitels
            """,
            "expected_output": {
                "label": "",
                "type": "document",
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
                        ]
                        },
                        {
                        "label": "",
                        "type": "list",
                        "children": [
                            {
                            "label": "1.",
                            "type": "list_item",
                            "content": [
                                "Dieses Gesetz regelt:"
                            ],
                            "children": [
                                {
                                "label": "",
                                "type": "list",
                                "children": [
                                    {
                                    "label": "a.",
                                    "type": "list_item",
                                    "content": [
                                        "die Veranstaltung, die Aufbereitung, die \u00dcbertragung und den Empfang von Radio- und Fernsehprogrammen;"
                                    ],
                                    "children": []
                                    },
                                    {
                                    "label": "b.",
                                    "type": "list_item",
                                    "content": [
                                        "die F\u00f6rdermassnahmen zugunsten der elektronischen Medien."
                                    ],
                                    "children": []
                                    },
                                    {
                                    "label": "abis.",
                                    "type": "list_item",
                                    "content": [
                                        "blabla."
                                    ],
                                    "children": []
                                    }
                                ]
                                }
                            ]
                            },
                            {
                            "label": "2.",
                            "type": "list_item",
                            "content": [
                                "Es gilt f\u00fcr:"
                            ],
                            "children": [
                                {
                                "label": "",
                                "type": "list",
                                "children": [
                                    {
                                    "label": "a.",
                                    "type": "list_item",
                                    "content": [
                                        "die Veranstalter von Radio- und Fernsehprogrammen;"
                                    ],
                                    "children": []
                                    }
                                ]
                                }
                            ]
                            }
                        ]
                        },
                        {
                        "label": "",
                        "type": "content",
                        "content": [
                            "Einf\u00fcgen der Art. 76a-76c vor dem Gliederungstitel des 4. Kapitels"
                        ]
                        }
                    ]
                    }
                ]
                }
        },
        {
            "markdown": """
# Art. 76a Selbstregulierung der Branche
* Das BAKOM kann in der Branche anerkannte Organisationen, die Regeln für die journalistische Praxis entwickeln und deren Einhaltung beaufsichtigen, auf ihr Gesuch hin finanziell unterstützen.
            """,
            "expected_output": {
                "label": "",
                "type": "document",
                "children": [
                    {
                    "label": "1",
                    "type": "heading",
                    "content": [
                        "Art. 76a Selbstregulierung der Branche"
                    ],
                    "children": [
                        {
                        "label": "",
                        "type": "list",
                        "children": [
                            {
                            "label": "",
                            "type": "list_item",
                            "content": [
                                "Das BAKOM kann in der Branche anerkannte Organisationen, die Regeln f\u00fcr die journalistische Praxis entwickeln und deren Einhaltung beaufsichtigen, auf ihr Gesuch hin finanziell unterst\u00fctzen."
                            ],
                            "children": []
                            }
                        ]
                        }
                    ]
                    }
                ]
            }
        },
        {
            "markdown": """
# § 2 Beitragsarten
- [1.] Beiträge werden als Stipendien bzw. Darlehen oder als Arbeitsmarktstipendien gewährt.
- [4.] Arbeitsmarktstipendien werden gewährt für Weiterbildungen, die dem Erwerb, dem Erhalt und der Stärkung der Arbeitsmarktfähigkeit dienen. Sie werden ausgerichtet als:
  - [a.] Beitrag an die anerkannten Kosten der Weiterbildung;
  - [b.] Erwerbsersatz an den weiterbildungsbedingten Erwerbsausfall.
            """,
            "expected_output": {
                "label": "",
                "type": "document",
                "children": [
                    {
                    "label": "1",
                    "type": "heading",
                    "content": [
                        "\u00a7 2 Beitragsarten"
                    ],
                    "children": [
                        {
                        "label": "",
                        "type": "list",
                        "children": [
                            {
                            "label": "1.",
                            "type": "list_item",
                            "content": [
                                "Beitr\u00e4ge werden als Stipendien bzw. Darlehen oder als Arbeitsmarktstipendien gew\u00e4hrt."
                            ],
                            "children": []
                            },
                            {
                            "label": "4.",
                            "type": "list_item",
                            "content": [
                                "Arbeitsmarktstipendien werden gew\u00e4hrt f\u00fcr Weiterbildungen, die dem Erwerb, dem Erhalt und der St\u00e4rkung der Arbeitsmarktf\u00e4higkeit dienen. Sie werden ausgerichtet als:"
                            ],
                            "children": [
                                {
                                "label": "",
                                "type": "list",
                                "children": [
                                    {
                                    "label": "a.",
                                    "type": "list_item",
                                    "content": [
                                        "Beitrag an die anerkannten Kosten der Weiterbildung;"
                                    ],
                                    "children": []
                                    },
                                    {
                                    "label": "b.",
                                    "type": "list_item",
                                    "content": [
                                        "Erwerbsersatz an den weiterbildungsbedingten Erwerbsausfall."
                                    ],
                                    "children": []
                                    }
                                ]
                                }
                            ]
                            }
                        ]
                        },
                    ]
                    }
                ]
            }
        }
    ]

    for i, case in enumerate(test_cases):
        print(f"Running test case {i}")
        result = model.parse_markdown_to_json_schema(case["markdown"])
        assert json.dumps(result, indent=2) == json.dumps(case["expected_output"], indent=2)
        #assert result == case["expected_output"]

if __name__ == '__main__':
    pytest.main()
