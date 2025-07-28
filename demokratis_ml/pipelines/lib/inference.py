"""Helpers for flows that perform inference using trained models."""

import json
import os
import pathlib
from typing import Any

import mlflow
import mlflow.sklearn
import prefect.logging
import sklearn.pipeline

from demokratis_ml.pipelines.lib import blocks


def load_model(model_name: str, model_version: int | str) -> tuple[sklearn.pipeline.Pipeline, str]:
    """Load a model from MLflow and return it along with its MLflow URI."""
    logger = prefect.logging.get_run_logger()

    # Connect to MLflow
    mlflow_credentials = blocks.MLflowCredentials.load("mlflow-credentials")
    mlflow.set_tracking_uri(mlflow_credentials.tracking_uri)
    # Horrible, but seems to be the only way apart from creating a config file:
    # https://github.com/mlflow/mlflow/discussions/12881
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_credentials.username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_credentials.password.get_secret_value()
    logger.info("Using MLflow tracking URI %s and username %s", mlflow.get_tracking_uri(), mlflow_credentials.username)

    # Load the model
    model_uri = (
        f"models:/{model_name}/{model_version}"
        if isinstance(model_version, int)
        else f"models:/{model_name}{model_version}"  # model alias, e.g. "document_type_classifier@production"
    )
    logger.info("Loading model from %s", model_uri)
    model = mlflow.sklearn.load_model(model_uri=model_uri)
    logger.info("Loaded model: %s", model)
    assert model is not None, "Model must be defined"
    return model, model_uri


def write_outputs(data: dict[str, Any]) -> pathlib.Path:
    """Write the output data (predictions) to a JSON file in the remote storage.

    The path and the file name are inferred from the metadata contained in the output.
    """
    logger = prefect.logging.get_run_logger()
    fs_model_output_storage = blocks.ExtendedRemoteFileSystem.load("remote-model-output-storage")
    output_path = pathlib.Path(data["model"]["name"]) / f"{data['generated_at']}_{data['output_format_version']}.json"
    output_bytes = json.dumps(data, indent=2).encode("utf-8")
    logger.info("Storing output to %s/%s (%d bytes)", fs_model_output_storage.basepath, output_path, len(output_bytes))
    fs_model_output_storage.write_path(str(output_path), output_bytes)
    logger.info("Writing the same data to latest.json")
    fs_model_output_storage.write_path(str(output_path.with_name("latest.json")), output_bytes)
    # pathlib.Path("test.json").write_bytes(output_bytes)  # Debugging output only
    return output_path
