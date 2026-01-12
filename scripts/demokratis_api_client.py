"""CLI tool to interact with the Demokratis API and get sample responses."""

import json
import os
import sys
from typing import Any

import httpx


def demokratis_api_request(endpoint: str, version: str = "v0.1", timeout: float = 180.0) -> list[dict[str, Any]]:
    """Make an authenticated request to the Demokratis API and return the JSON response."""
    username = os.environ["DEMOKRATIS_API_USERNAME"]
    password = os.environ["DEMOKRATIS_API_PASSWORD"]

    url = f"https://www.demokratis.ch/api/{version}/{endpoint}"
    print(url, file=sys.stderr)
    response = httpx.get(url, auth=(username, password), timeout=timeout)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    if len(sys.argv) < 2:  # noqa: PLR2004
        print(
            "Usage: python demokratis_api_client.py <documents-metadata|documents-content|stored-files>",
            file=sys.stderr,
        )
        sys.exit(1)

    endpoint = sys.argv[1]

    response_data = demokratis_api_request(endpoint)
    if response_data:
        print(f"{len(response_data)} records returned from the API.", file=sys.stderr)
        print(json.dumps(response_data, indent=2))
    else:
        print("No data returned from the API.", file=sys.stderr)
