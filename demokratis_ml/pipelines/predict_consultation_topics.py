"""Pipeline for inference via the consultation_topic_classifier model; see the `predict_consultation_topics` flow."""

import datetime
import pathlib
from collections.abc import Iterable, Iterator

import pandas as pd
import prefect
import prefect.logging

import demokratis_ml.models.consultation_topics.model
import demokratis_ml.models.consultation_topics.preprocessing
from demokratis_ml.pipelines.lib import inference, utils

OUTPUT_FORMAT_VERSION = "v0.1"


@prefect.flow()
@utils.slack_status_report(":speech_balloon:")
def predict_consultation_topics(  # noqa: PLR0913
    data_files_version: datetime.date,
    store_dataframes_remotely: bool,
    model_name: str = "consultation_topic_classifier",
    model_version: int | str = 4,
    embedding_model_name: str = "openai/text-embedding-3-large",
    only_consultations_since: datetime.date = datetime.date(2019, 1, 1),
    only_languages: Iterable[str] | None = ("de",),
) -> pathlib.Path:
    """
    Load consultations which don't have manually reviewed topics yet, and predict their topics using a trained model.

    Model name and version are passed as parameters and the model is loaded from MLflow. Note however that
    we quietly assume that the trained model is compatible with the input data format which is implemented
    in `demokratis_ml.models.consultation_topics.*` modules. This code is not stored in MLflow!

    The output is encoded as JSON and stored in the "remote-model-output-storage" file system.

    :param data_files_version: Version (date) of the data files to use. The date is a part of the file names.
    :param store_dataframes_remotely: If true, read inputs from Exoscale object storage.
    :param embedding_model_name: Used to determine which embedding dataframe to load.
    :param only_consultations_since: Only process consultations that started on or after this date.
        This is to avoid processing old and likely irrelevant consultations.
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
    document_embeddings_dataframe_name = f"consultation-documents-embeddings-beginnings-{embedding_model_name.replace('/', '-')}-{data_files_version}.parquet"  # noqa: E501
    consultation_embeddings_dataframe_name = f"consultation-attributes-embeddings-beginnings-{embedding_model_name.replace('/', '-')}-{data_files_version}.parquet"  # noqa: E501
    logger.info(
        "Loading documents from %r, document embeddings from %r, consultation embeddings from %r",
        documents_dataframe_name,
        document_embeddings_dataframe_name,
        consultation_embeddings_dataframe_name,
    )
    # Load preprocessed documents
    df_documents = utils.read_dataframe(
        pathlib.Path(documents_dataframe_name),
        columns=None,
        fs=fs_dataframe_storage,
    )
    # Only generate predictions for consultations which don't have manual review tag yet.
    # TODO: move this filter to the loading stage
    df_documents = df_documents[df_documents["consultation_topics_label_source"] != "manual"]
    # Filter by languages
    # TODO: move this filter to the loading stage
    if only_languages is not None:
        df_documents = df_documents[df_documents["document_language"].isin(only_languages)]
    # Filter by age
    # TODO: move this filter to the loading stage
    df_documents = df_documents[df_documents["consultation_start_date"] >= pd.Timestamp(only_consultations_since)]

    logger.info(
        "Loaded %d documents for %d consultations. Filters: only_languages=%r, only_consultations_since=%s",
        len(df_documents),
        df_documents["consultation_identifier"].nunique(),
        only_languages,
        only_consultations_since,
    )

    df_document_embeddings = utils.read_dataframe(
        pathlib.Path(document_embeddings_dataframe_name), columns=None, fs=fs_dataframe_storage
    )
    df_consultation_embeddings = utils.read_dataframe(
        pathlib.Path(consultation_embeddings_dataframe_name), columns=None, fs=fs_dataframe_storage
    )
    if only_languages is not None:
        df_consultation_embeddings = df_consultation_embeddings[
            df_consultation_embeddings.index.get_level_values("attribute_language").isin(only_languages)
        ]

    # Can't trust unfiltered_topic_columns because the topic dropping doesn't happen here
    df_input, unfiltered_topic_columns = demokratis_ml.models.consultation_topics.preprocessing.create_input_dataframe(
        df_documents=df_documents,
        df_document_embeddings=df_document_embeddings,
        df_consultation_embeddings=df_consultation_embeddings,
    )
    logger.info("Input dataframe shape: %s", df_input.shape)
    # Generate predictions
    x, _ = demokratis_ml.models.consultation_topics.model.create_matrices(df_input, unfiltered_topic_columns)
    logger.info("Input feature matrix shape: %s", x.shape)
    pred_probs = demokratis_ml.models.consultation_topics.model.get_predicted_label_probabilities(
        classifier.predict_proba(x)
    )
    df_predictions = pd.DataFrame(pred_probs, columns=model_metadata["supported_topics"], index=df_input.index)
    logger.info("Outuput dataframe shape: %s", df_predictions.shape)

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
            "document_embeddings_dataframe_name": document_embeddings_dataframe_name,
            "consultation_embeddings_dataframe_name": consultation_embeddings_dataframe_name,
        },
        "input_filters": {
            "only_consultations_since": only_consultations_since.isoformat(),
            "only_languages": list(only_languages) if only_languages is not None else None,
        },
        "outputs": list(serialize_predictions(df_predictions)),
    }

    output_path = inference.write_outputs(output)
    return output_path


def serialize_predictions(df_predictions: pd.DataFrame, output_proba_threshold: float = 0.5) -> Iterator[dict]:
    """For each consultation, return a list of labels with probabilities > threshold.

    Since this is a multi-label classification task, we don't want to return all labels, but only those
    that the model is confident about.
    """
    for idx, row in df_predictions.iterrows():
        outputs = [
            {"label": label, "score": round(proba, 4)} for label, proba in row.items() if proba > output_proba_threshold
        ]
        # Do not return empty outputs (where the model isn't confident enough)
        if outputs:
            yield {
                "consultation_identifier": idx,
                "output": outputs,
            }


if __name__ == "__main__":
    import sys

    data_files_version = datetime.date.fromisoformat(sys.argv[1])
    output_path = predict_consultation_topics(
        data_files_version=data_files_version,
        store_dataframes_remotely=False,
    )
    print(output_path)
