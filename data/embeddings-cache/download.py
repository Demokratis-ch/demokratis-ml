"""Run this script with `uv run data/embeddings-cache/download.py` to download embeddings caches from Exoscale.

This overwrites your local cache!
"""

import logging
import pathlib
import sys

import dotenv

CACHE_FILES = ("openai--text-embedding-3-large.parquet",)

CACHE_DIRECTORY = pathlib.Path(__file__).parent
REPOSITORY_ROOT = (CACHE_DIRECTORY / ".." / "..").resolve()

sys.path.append(str(REPOSITORY_ROOT))

import research.lib.data_access  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dotenv.load_dotenv()
    for cache_file in CACHE_FILES:
        research.lib.data_access.download_file_from_exoscale(
            remote_path=pathlib.Path("tmp") / "embeddings-cache" / cache_file,
            local_path=CACHE_DIRECTORY / cache_file,
        )
