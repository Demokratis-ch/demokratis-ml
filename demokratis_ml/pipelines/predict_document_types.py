"""Prefect pipeline for inference via the document_type_classifier model; see the `predict_document_types` flow."""

import datetime
import pathlib
from collections.abc import Iterable

import pandas as pd
import prefect
import prefect.logging

import demokratis_ml.models.document_types.model
import demokratis_ml.models.document_types.preprocessing
from demokratis_ml.pipelines.lib import inference, utils

OUTPUT_FORMAT_VERSION = "v0.1"


@prefect.flow()
@utils.slack_status_report(":page_with_curl:")
def predict_document_types(  # noqa: PLR0913
    data_files_version: datetime.date,
    store_dataframes_remotely: bool,
    model_name: str = "document_type_classifier",
    model_version: int | str = 5,
    embedding_model_name: str = "openai/text-embedding-3-large",
    only_consultations_since: datetime.date = datetime.date(2019, 1, 1),
    only_languages: Iterable[str] | None = ("de",),
) -> pathlib.Path:
    """
    Load consultation documents with missing document types and predict the types using a trained model.

    Dataframes with extra features and embeddings are also loaded.

    Model name and version are passed as parameters and the model is loaded from MLflow. Note however that
    we quietly assume that the trained model is compatible with the input data format which is implemented
    in `demokratis_ml.models.document_types.*` modules. This code is not stored in MLflow!

    The output is encoded as JSON and stored in the "remote-model-output-storage" file system.

    :param data_files_version: Version (date) of the data files to use. The date is a part of the file names.
    :param store_dataframes_remotely: If true, read inputs from Exoscale object storage.
    :param embedding_model_name: Used to determine which embedding dataframe to load.
    :param only_consultations_since: Only process documents from consultations that started on or after this date.
        This is to avoid processing old and likely irrelevant documents.
    :param only_languages: If set, only documents in the specified languages will be processed. This is to
        save time and resources at a stage where we're only developing the models and don't cover all languages yet.
    """
    logger = prefect.logging.get_run_logger()
    if only_languages is not None:
        only_languages = set(only_languages)

    classifier, model_uri, model_metadata = inference.load_model(model_name, model_version)

    # Choose where to load source dataframes from and where to store the resulting dataframe
    fs_dataframe_storage = utils.get_dataframe_storage(store_dataframes_remotely)

    # TODO: these file name templates should be shared between flows
    documents_dataframe_name = f"consultation-documents-preprocessed-{data_files_version}.parquet"
    embeddings_dataframe_name = f"consultation-documents-embeddings-beginnings-{embedding_model_name.replace('/', '-')}-{data_files_version}.parquet"  # noqa: E501
    features_dataframe_name = f"consultation-documents-features-{data_files_version}.parquet"
    logger.info(
        "Loading documents from %r, embeddings from %r, features from %r",
        documents_dataframe_name,
        embeddings_dataframe_name,
        features_dataframe_name,
    )
    # Load preprocessed documents
    df_documents = utils.read_dataframe(
        pathlib.Path(documents_dataframe_name),
        columns=None,
        fs=fs_dataframe_storage,
    )
    # Only generate predictions for documents which don't have document_type yet
    # TODO: move this filter to the loading stage
    df_documents = df_documents[df_documents["document_type"].isna()]
    # Filter by languages
    # TODO: move this filter to the loading stage
    if only_languages is not None:
        df_documents = df_documents[df_documents["document_language"].isin(only_languages)]
    # Filter by age
    # TODO: move this filter to the loading stage
    df_documents = df_documents[df_documents["consultation_start_date"] >= pd.Timestamp(only_consultations_since)]

    logger.info(
        "Loaded %d documents to predict document types for. Filters: only_languages=%r, only_consultations_since=%s",
        len(df_documents),
        only_languages,
        only_consultations_since,
    )

    df_embeddings = utils.read_dataframe(pathlib.Path(embeddings_dataframe_name), columns=None, fs=fs_dataframe_storage)
    df_features = utils.read_dataframe(pathlib.Path(features_dataframe_name), columns=None, fs=fs_dataframe_storage)

    df_input = demokratis_ml.models.document_types.preprocessing.create_input_dataframe(
        df_documents=df_documents,
        df_extra_features=df_features,
        df_embeddings=df_embeddings,
    )
    logger.info("Input dataframe for the model has %d rows and %d columns", df_input.shape[0], df_input.shape[1])

    # Generate predictions
    x, _ = demokratis_ml.models.document_types.model.create_matrices(df_input)
    logger.info("Input feature matrix has shape %s", x.shape)
    y_proba = classifier.predict_proba(x)
    df_predictions = pd.DataFrame(y_proba, columns=classifier.classes_, index=df_input["document_uuid"])

    # Format the output
    generated_at = datetime.datetime.now(tz=datetime.UTC)
    assert OUTPUT_FORMAT_VERSION == "v0.1", "The code below produces this version"
    output = {
        "generated_at": generated_at.isoformat(),
        "output_format_version": OUTPUT_FORMAT_VERSION,
        "model": {
            "name": model_name,
            "version": model_version,
            "uri": model_uri,
            "metadata": model_metadata,
        },
        "features": {
            "embedding_model": embedding_model_name,
        },
        "input_files": {
            "version": data_files_version.isoformat(),
            "documents_dataframe_name": documents_dataframe_name,
            "embeddings_dataframe_name": embeddings_dataframe_name,
            "features_dataframe_name": features_dataframe_name,
        },
        "input_filters": {
            "only_consultations_since": only_consultations_since.isoformat(),
            "only_languages": list(only_languages) if only_languages is not None else None,
        },
        "outputs": serialize_predictions(df_predictions),
    }

    output_path = inference.write_outputs(output)
    return output_path


def serialize_predictions(df_predictions: pd.DataFrame) -> list[dict]:
    """For each document, return a list of labels sorted by scores (highest first)."""
    return [
        {
            "document_uuid": idx,
            "output": [
                {"label": label, "score": round(proba, 4)} for label, proba in row.sort_values(ascending=False).items()
            ],
        }
        for idx, row in df_predictions.iterrows()
    ]


if __name__ == "__main__":
    import sys

    data_files_version = datetime.date.fromisoformat(sys.argv[1])
    output_path = predict_document_types(
        data_files_version=data_files_version,
        store_dataframes_remotely=False,
    )
    print(output_path)
