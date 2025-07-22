"""Serve all our flows - this script is typically run inside a Docker container that has access to our Prefect server.

- The value of the ``STORE_DATAFRAMES_REMOTELY`` environment variable is used to set the ``store_dataframes_remotely``
  parameter of the flows.
- The ``CRON_MAIN_INGESTION`` environment variable, if set, is used to schedule the ``main-ingestion`` flow.
  The variable must be a valid cron string (e.g. "0 9 * * *").
"""

import os

import prefect
import prefect.client.schemas.schedules

from demokratis_ml.pipelines import (
    embed_documents,
    extract_document_features,
    main_ingestion,
    predict_document_types,
    preprocess_consultation_documents,
)


def _schedule_from_env(key: str) -> list[prefect.client.schemas.schedules.CronSchedule] | None:
    """Return a cron schedule from the environment variable with the given key, or None if not set."""
    if key in os.environ:
        return [
            prefect.client.schemas.schedules.CronSchedule(
                cron=os.environ[key],
                timezone="Europe/Zurich",
            )
        ]
    return None


if __name__ == "__main__":
    store_dataframes_remotely = os.environ.get("STORE_DATAFRAMES_REMOTELY", "0").lower() in {"1", "true", "yes"}
    deployment_version = os.environ.get("DOCKER_IMAGE_TAG")

    main_ingestion_deployment = main_ingestion.main_ingestion.to_deployment(
        name="main-ingestion",
        parameters={
            "publish": False,
            "store_dataframes_remotely": store_dataframes_remotely,
            "bootstrap_from_previous_output": True,
        },
        schedules=_schedule_from_env("CRON_MAIN_INGESTION"),
        version=deployment_version,
    )

    preprocess_consultation_documents_deployment = preprocess_consultation_documents.preprocess_data.to_deployment(
        name="preprocess-consultation-documents",
        parameters={
            "publish": False,
            "store_dataframes_remotely": store_dataframes_remotely,
        },
        version=deployment_version,
    )

    embed_documents_deployment = embed_documents.embed_documents.to_deployment(
        name="embed-documents",
        parameters={"store_dataframes_remotely": store_dataframes_remotely},
        version=deployment_version,
    )

    extract_document_features_deployment = extract_document_features.extract_document_features.to_deployment(
        name="extract-document-features",
        parameters={"store_dataframes_remotely": store_dataframes_remotely},
        version=deployment_version,
    )

    predict_document_types_deployment = predict_document_types.predict_document_types.to_deployment(
        name="predict-document-types",
        parameters={"store_dataframes_remotely": store_dataframes_remotely},
        version=deployment_version,
    )

    prefect.serve(
        main_ingestion_deployment,
        preprocess_consultation_documents_deployment,
        embed_documents_deployment,
        extract_document_features_deployment,
        predict_document_types_deployment,
    )
