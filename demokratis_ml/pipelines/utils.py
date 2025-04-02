"""Utilities shared by multiple pipelines."""

import functools
from collections.abc import Callable
from typing import Any, TypeVar

import pandas as pd
import pandera.errors
import prefect.logging

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
