"""Serve all our flows - this script is typically run inside a Docker container that has access to our Prefect server.

The value of the ``STORE_DATAFRAMES_REMOTELY`` environment variable is used to set the ``store_dataframes_remotely``
parameter of the flow(s).
"""

import os

import prefect

from demokratis_ml.pipelines import preprocess_consultation_documents

if __name__ == "__main__":
    store_dataframes_remotely = os.environ.get("STORE_DATAFRAMES_REMOTELY", "0").lower() in {"1", "true", "yes"}

    preprocess_consultation_documents_deployment = preprocess_consultation_documents.preprocess_data.to_deployment(
        name="preprocess-consultation-documents",
        parameters={
            "publish": False,
            "store_dataframes_remotely": store_dataframes_remotely,
        },
    )
    prefect.serve(preprocess_consultation_documents_deployment)
