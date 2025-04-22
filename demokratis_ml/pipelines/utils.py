"""Utilities shared by multiple pipelines."""

import datetime
import functools
import pathlib
import time
from collections.abc import Callable
from typing import Any, TypeVar

import pandas as pd
import pandera.errors
import prefect.logging
import pyarrow.parquet

from demokratis_ml.pipelines import blocks

F = TypeVar("F", bound=Callable[..., Any])


def print_validation_failure_cases() -> Callable[[F], F]:
    """In case the wrapped function raises a SchemaErrors exception, print the failure cases."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except pandera.errors.SchemaErrors as exc:
                logger = prefect.logging.get_run_logger()
                df_index = exc.failure_cases["index"]
                with pd.option_context(
                    "display.max_rows",
                    None,
                    "display.max_columns",
                    None,
                    "display.max_colwidth",
                    100,
                ):
                    hr = "=" * 80
                    logger.error(  # noqa: TRY400
                        "\n".join(
                            (
                                "",
                                hr,
                                "Schema errors and failure cases:",
                                repr(exc.failure_cases),
                                hr,
                                "DataFrame rows that failed validation:",
                                repr(exc.data.loc[df_index]),
                                hr,
                            )
                        )
                    )
                raise

        return wrapper

    return decorator


def get_dataframe_storage(store_dataframes_remotely: bool) -> blocks.ExtendedFileSystemType:
    """Return the file system to use for storing DataFrames."""
    logger = prefect.logging.get_run_logger()
    if store_dataframes_remotely:
        fs = blocks.ExtendedRemoteFileSystem.load(storage_block_name := "remote-dataframe-storage")
    else:
        fs = blocks.ExtendedLocalFileSystem.load(storage_block_name := "local-dataframe-storage")
    logger.info(
        "store_dataframes_remotely=%s => using storage=%r with basepath=%s",
        store_dataframes_remotely,
        storage_block_name,
        fs.basepath,
    )
    return fs


def store_dataframe(df: pd.DataFrame, name_prefix: str, fs: blocks.ExtendedFileSystemType) -> bytes:
    """Serialise a DataFrame to Parquet and store it in the given file system.

    :returns: The serialised data.
    """
    logger = prefect.logging.get_run_logger()
    logger.info("Serialising dataframe with %d rows to Parquet", len(df))
    t0 = time.monotonic()
    data = df.to_parquet(compression="snappy")
    logger.info("Serialised dataframe in %.1f seconds", time.monotonic() - t0)
    now = datetime.datetime.now(tz=datetime.UTC)
    path = pathlib.Path(f"{name_prefix}-{now:%Y-%m-%d}.parquet")
    if fs.path_exists(path):
        logger.warning("Overwriting existing file %r", path)
    logger.info("Writing %d rows, %.1f MiB to %s/%s", len(df), len(data) / 1024**2, fs.basepath, path)
    t0 = time.monotonic()
    fs.write_path(str(path), data)
    logger.info("Wrote file in %.1f seconds", time.monotonic() - t0)
    return data


def find_latest_dataframe(name_prefix: str, fs: blocks.ExtendedFileSystemType) -> pathlib.Path:
    """Find the most recent DataFrame stored in the given file system."""
    logger = prefect.logging.get_run_logger()
    paths = [p for p in fs.iterdir() if p.name.startswith(name_prefix) and p.name.endswith(".parquet")]
    logger.debug("Found existing dataframes: %r", paths)
    if not paths:
        msg = f"No existing dataframe prefixed '{name_prefix}' found"
        raise FileNotFoundError(msg)
    return max(paths)


def read_dataframe(path: pathlib.Path, columns: list[str] | None, fs: blocks.ExtendedFileSystemType) -> pd.DataFrame:
    """Read a Parquet file from the file system, ideally transferring just the columns we need."""
    with fs.open(path, "rb") as f:
        parquet_file = pyarrow.parquet.ParquetFile(f)
        table = parquet_file.read(columns=columns)
        return table.to_pandas()
