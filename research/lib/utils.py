import logging
import os
import pprint

import mlflow


def set_up_logging_and_mlflow(experiment_name: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger(__name__)

    if os.environ.get("MLFLOW_TRACKING_USERNAME") and os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        mlflow.set_tracking_uri(uri := "https://mlflow.ml1.demokratis.ch/")
        logger.info("MLflow tracking to %s", uri)
    else:
        logger.warning("MLflow credentials not found, will track locally.")
        mlflow.set_tracking_uri("sqlite:///mlruns.db")

    mlflow.set_experiment(experiment_name)
    if run := mlflow.active_run():
        logger.warning("Run = %s is already active, closing it.", run.info.run_name)
        mlflow.end_run()
    run = mlflow.start_run()
    logger.info("Starting run = %s", run.info.run_name)


def log_metrics(**metrics: float) -> None:
    """
    Round metrics to 4 decimal places, print them, and log them to MLflow.
    """
    rounded_metrics = {k: round(v, 4) for k, v in sorted(metrics.items())}
    pprint.pprint(rounded_metrics)  # noqa: T203
    mlflow.log_metrics(rounded_metrics)
