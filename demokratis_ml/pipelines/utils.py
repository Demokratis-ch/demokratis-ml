"""Utilities shared by multiple pipelines."""

import contextlib
import datetime
import functools
import pathlib
import socket
import time
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

import pandas as pd
import pandera.errors
import prefect.context
import prefect.logging
import prefect.settings
import prefect_slack
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


def store_dataframe(
    df: pd.DataFrame, name_prefix: str, fs: blocks.ExtendedFileSystemType
) -> tuple[pathlib.Path, bytes]:
    """Serialise a DataFrame to Parquet and store it in the given file system.

    :returns: (path to the written file, the serialised data).
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
    return path, data


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


def slack_status_report() -> Callable[[F], F]:
    """Context manager decorator that reports flow execution time to Slack.

    This decorator wraps the function in a context manager that:
    1. Records the start time
    2. Executes the function
    3. Records the end time
    4. Sends a Slack message with the function name and execution time

    Requires the "slack-status-webhook" block to be configured in Prefect.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            @contextlib.contextmanager
            def execution_timer() -> Iterator[None]:
                logger = prefect.logging.get_run_logger()
                webhook_client = prefect_slack.SlackWebhook.load("slack-status-webhook").get_client(sync_client=True)
                context = prefect.context.FlowRunContext.get()
                run_url = f"{prefect.settings.PREFECT_UI_URL}/runs/flow-run/{context.flow_run.id}"
                version = context.flow_run.deployment_version or ""
                start_time = time.monotonic()
                exception = None
                try:
                    yield
                except Exception as exc:
                    exception = repr(exc)
                    raise
                finally:
                    end_time = time.monotonic()
                    execution_time = end_time - start_time
                    execution_time_repr = f"{execution_time // 60:.0f}m {execution_time % 60:02.1f}s"

                    icon = ":large_green_circle:" if exception is None else ":red_circle:"
                    hostname = socket.gethostname()
                    message = (
                        f"{icon} `{func.__module__}.{func.__name__}` {version} executed in {execution_time_repr}"
                        f" on {hostname}"
                        f"\n{run_url}"
                    )
                    if exception is not None:
                        message += f"\n*Exception:* {exception}"

                    response = webhook_client.send(text=message)
                    if response.status_code != 200:  # noqa: PLR2004
                        logger.error(
                            "Failed to send Slack notification. Status code: %d, Response: %s",
                            response.status_code,
                            response.text,
                        )

            with execution_timer():
                return func(*args, **kwargs)

        return wrapper

    return decorator
