"""Create block instances and save them to the Prefect server."""

import os

import prefect.filesystems
import prefect_slack

from demokratis_ml.pipelines.lib import blocks

demokratis_api_credentials = blocks.DemokratisAPICredentials(
    username=os.environ["DEMOKRATIS_API_USERNAME"],
    password=os.environ["DEMOKRATIS_API_PASSWORD"],
)
demokratis_api_credentials.save("demokratis-api-credentials", overwrite=True)

openai_credentials = blocks.OpenAICredentials(
    api_key=os.environ["OPENAI_API_KEY"],
    organization=None,
)
openai_credentials.save("openai-credentials", overwrite=True)


hf_credentials = blocks.HuggingFaceDatasetUploadCredentials(
    token=os.environ["HF_TOKEN"],
)
hf_credentials.save("huggingface-dataset-upload-credentials", overwrite=True)


mlflow_credentials = blocks.MLflowCredentials(
    tracking_uri="https://mlflow.ml1.demokratis.ch/",
    username=os.environ["MLFLOW_TRACKING_USERNAME"],
    password=os.environ["MLFLOW_TRACKING_PASSWORD"],
)
mlflow_credentials.save("mlflow-credentials", overwrite=True)


local_dataframe_storage = blocks.ExtendedLocalFileSystem(basepath="data/dataframes")
local_dataframe_storage.save("local-dataframe-storage", overwrite=True)


remote_dataframe_storage = blocks.ExtendedRemoteFileSystem(
    basepath=f"s3://{os.environ['EXOSCALE_SOS_BUCKET_ML']}/dataframes",
    settings={
        "key": os.environ["EXOSCALE_SOS_ACCESS_KEY"],
        "secret": os.environ["EXOSCALE_SOS_SECRET_KEY"],
        "client_kwargs": {
            "endpoint_url": os.environ["EXOSCALE_SOS_ENDPOINT"],
        },
    },
)
remote_dataframe_storage.save("remote-dataframe-storage", overwrite=True)


# The web platform stores mirrored documents in here:
platform_file_storage = prefect.filesystems.RemoteFileSystem(
    basepath=f"s3://{os.environ['EXOSCALE_SOS_BUCKET_PLATFORM_FILE_STORAGE']}/",
    settings={
        "key": os.environ["EXOSCALE_SOS_ACCESS_KEY"],
        "secret": os.environ["EXOSCALE_SOS_SECRET_KEY"],
        "client_kwargs": {
            "endpoint_url": os.environ["EXOSCALE_SOS_ENDPOINT"],
        },
    },
)
platform_file_storage.save("platform-file-storage", overwrite=True)


remote_model_output_storage = blocks.ExtendedRemoteFileSystem(
    basepath=f"s3://{os.environ['EXOSCALE_SOS_BUCKET_ML']}/model_outputs",
    settings={
        "key": os.environ["EXOSCALE_SOS_ACCESS_KEY"],
        "secret": os.environ["EXOSCALE_SOS_SECRET_KEY"],
        "client_kwargs": {
            "endpoint_url": os.environ["EXOSCALE_SOS_ENDPOINT"],
        },
    },
)
remote_model_output_storage.save("remote-model-output-storage", overwrite=True)


slack_status = prefect_slack.SlackWebhook(url=os.environ["SLACK_STATUS_WEBHOOK_URL"])
slack_status.save("slack-status-webhook", overwrite=True)
