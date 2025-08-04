"""Serve all our flows - this script is typically run inside a Docker container that has access to our Prefect server.

- The value of the ``STORE_DATAFRAMES_REMOTELY`` environment variable is used to set the ``store_dataframes_remotely``
  parameter of the flows.
- The ``CRON_MAIN_INGESTION_STANDARD`` environment variable, if set, is used to schedule the ``main-ingestion`` flow.
  The variable must be a valid cron string (e.g. "0 9 * * *").
- If the ``CRON_MAIN_INGESTION_PUBLISH`` environment variable is also set, it is used to schedule the
  ``main-ingestion`` flow with the `publish` parameter set to True. This tells the flow to upload the data
  to Hugging Face.
- The ``CRON_EXPIRE_EXOSCALE_SOS_OBJECTS`` environment variable, if set, is used to schedule the
  ``expire_exoscale_sos_objects`` flow.
- Any cron schedules are in the timezone specified by the ``TZ`` environment variable, defaulting to UTC.
"""

import contextlib
import dataclasses
import os
from collections.abc import Iterator

import prefect
import prefect.schedules

from demokratis_ml.pipelines import (
    embed_consultations,
    embed_documents,
    expire_exoscale_sos_objects,
    extract_document_features,
    main_ingestion,
    predict_consultation_topics,
    predict_document_types,
    preprocess_consultation_documents,
)


def _get_schedule_from_env_var(env_var_name: str) -> prefect.schedules.Schedule:
    return prefect.schedules.Schedule(
        cron=os.environ[env_var_name],
        timezone=os.environ.get("TZ", "UTC"),
    )


def _get_main_ingestion_schedules() -> Iterator[prefect.schedules.Schedule]:
    with contextlib.suppress(KeyError):
        yield _get_schedule_from_env_var("CRON_MAIN_INGESTION_STANDARD")
    with contextlib.suppress(KeyError):
        schedule = _get_schedule_from_env_var("CRON_MAIN_INGESTION_PUBLISH")
        yield dataclasses.replace(schedule, parameters={"publish": True})


def _get_expire_exoscale_sos_objects_schedule() -> Iterator[prefect.schedules.Schedule]:
    with contextlib.suppress(KeyError):
        yield _get_schedule_from_env_var("CRON_EXPIRE_EXOSCALE_SOS_OBJECTS")


if __name__ == "__main__":
    store_dataframes_remotely = os.environ.get("STORE_DATAFRAMES_REMOTELY", "0").lower() in {"1", "true", "yes"}
    deployment_version = os.environ.get("DOCKER_IMAGE_TAG")
    main_ingestion_schedules = list(_get_main_ingestion_schedules())
    expire_exoscale_sos_objects_schedule = list(_get_expire_exoscale_sos_objects_schedule())
    print("main-ingestion schedules:", main_ingestion_schedules)
    print("expire-exoscale-sos-objects schedule:", expire_exoscale_sos_objects_schedule)

    main_ingestion_deployment = main_ingestion.main_ingestion.to_deployment(
        name="main-ingestion",
        parameters={
            "publish": False,
            "store_dataframes_remotely": store_dataframes_remotely,
            "bootstrap_from_previous_output": True,
        },
        schedules=main_ingestion_schedules,
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

    embed_consultations_deployment = embed_consultations.embed_consultations.to_deployment(
        name="embed-consultations",
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

    predict_consultation_topics_deployment = predict_consultation_topics.predict_consultation_topics.to_deployment(
        name="predict-consultation-topics",
        parameters={"store_dataframes_remotely": store_dataframes_remotely},
        version=deployment_version,
    )

    expire_exoscale_sos_objects_deployment = expire_exoscale_sos_objects.expire_exoscale_sos_objects.to_deployment(
        name="expire-exoscale-sos-objects",
        parameters={
            "storage_block_name": "remote-dataframe-storage",
            "path_glob": "*.parquet",
            "max_age_days": 60,
            "dry_run": False,
        },
        schedules=expire_exoscale_sos_objects_schedule,
        version=deployment_version,
    )

    prefect.serve(
        main_ingestion_deployment,
        preprocess_consultation_documents_deployment,
        embed_consultations_deployment,
        embed_documents_deployment,
        extract_document_features_deployment,
        predict_document_types_deployment,
        predict_consultation_topics_deployment,
        expire_exoscale_sos_objects_deployment,
    )
