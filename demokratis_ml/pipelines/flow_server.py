"""Serve all our flows - this script is typically run inside a Docker container that has access to our Prefect server.

- The value of the ``STORE_DATAFRAMES_REMOTELY`` environment variable is used to set the ``store_dataframes_remotely``
  parameter of the flows.
- The ``CRON_PREPROCESS_CONSULTATION_DOCUMENTS`` environment variable, if set, is used to schedule the
  ``preprocess-consultation-documents`` flow. The variable must be a valid cron string (e.g. "0 9 * * *").
"""

import os

import prefect
import prefect.client.schemas.schedules

from demokratis_ml.pipelines import extract_document_features, preprocess_consultation_documents


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

    preprocess_consultation_documents_deployment = preprocess_consultation_documents.preprocess_data.to_deployment(
        name="preprocess-consultation-documents",
        parameters={
            "publish": False,
            "store_dataframes_remotely": store_dataframes_remotely,
        },
        schedules=_schedule_from_env("CRON_PREPROCESS_CONSULTATION_DOCUMENTS"),
    )

    extract_document_features_deployment = extract_document_features.extract_document_features.to_deployment(
        name="extract-document-features",
        parameters={
            # "consultation_documents_file": "",
            "bootstrap_from_previous_output": False,
            "store_dataframes_remotely": store_dataframes_remotely,
        },
        # Can't schedule this on its own because it needs the `consultation_documents_file` parameter to be set.
        # schedules=_schedule_from_env("CRON_EXTRACT_DOCUMENT_FEATURES"),
    )

    prefect.serve(
        preprocess_consultation_documents_deployment,
        extract_document_features_deployment,
    )
