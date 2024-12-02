"""Create block instances and save them to the Prefect server."""

import os

from demokratis_ml.pipelines.blocks import DemokratisAPICredentials, ExtendedLocalFileSystem

demokratis_api_credentials = DemokratisAPICredentials(
    username=os.environ["DEMOKRATIS_API_USERNAME"],
    password=os.environ["DEMOKRATIS_API_PASSWORD"],
)
demokratis_api_credentials.save("demokratis-api-credentials", overwrite=True)


local_document_storage = ExtendedLocalFileSystem(basepath="data/consultation-documents")
local_document_storage.save("local-document-storage", overwrite=True)


local_dataframe_storage = ExtendedLocalFileSystem(basepath="data/dataframes")
local_dataframe_storage.save("local-dataframe-storage", overwrite=True)


# TODO: define equivalent storages on Exoscale SOS.
