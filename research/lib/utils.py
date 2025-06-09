import pprint

import mlflow


def log_metrics(**metrics: float) -> None:
    """
    Round metrics to 4 decimal places, print them, and log them to MLflow.
    """
    rounded_metrics = {k: round(v, 4) for k, v in sorted(metrics.items())}
    pprint.pprint(rounded_metrics)  # noqa: T203
    mlflow.log_metrics(rounded_metrics)
