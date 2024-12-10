"""Create block instances and save them to the Prefect server."""

import os

import prefect.filesystems

from demokratis_ml.pipelines.blocks import (
    DemokratisAPICredentials,
    ExtendedLocalFileSystem,
    HuggingFaceDatasetUploadCredentials,
)

demokratis_api_credentials = DemokratisAPICredentials(
    username=os.environ["DEMOKRATIS_API_USERNAME"],
    password=os.environ["DEMOKRATIS_API_PASSWORD"],
)
demokratis_api_credentials.save("demokratis-api-credentials", overwrite=True)


hf_credentials = HuggingFaceDatasetUploadCredentials(
    token=os.environ["HF_TOKEN"],
)
hf_credentials.save("huggingface-dataset-upload-credentials", overwrite=True)


local_document_storage = ExtendedLocalFileSystem(basepath="data/consultation-documents")
local_document_storage.save("local-document-storage", overwrite=True)


local_dataframe_storage = ExtendedLocalFileSystem(basepath="data/dataframes")
local_dataframe_storage.save("local-dataframe-storage", overwrite=True)


# TODO: define document storage on Exoscale SOS.

remote_dataframe_storage = prefect.filesystems.RemoteFileSystem(
    basepath=f"s3://{os.environ['EXOSCALE_SOS_BUCKET']}/dataframes",
    settings={
        "key": os.environ["EXOSCALE_SOS_ACCESS_KEY"],
        "secret": os.environ["EXOSCALE_SOS_SECRET_KEY"],
        "client_kwargs": {
            "endpoint_url": os.environ["EXOSCALE_SOS_ENDPOINT"],
        },
    },
)
remote_dataframe_storage.save("remote-dataframe-storage", overwrite=True)
