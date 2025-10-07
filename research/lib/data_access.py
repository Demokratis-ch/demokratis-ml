"""Utilities for selecting and loading input data."""

import logging
import os
import pathlib

import boto3


def download_file_from_exoscale(remote_path: pathlib.Path, local_path: pathlib.Path) -> None:
    """Download an arbitrary file from our Exoscale Simple Object Storage bucket."""
    logger = logging.getLogger("download_file_from_exoscale")
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["EXOSCALE_SOS_ACCESS_KEY"],
        aws_secret_access_key=os.environ["EXOSCALE_SOS_SECRET_KEY"],
        endpoint_url=os.environ["EXOSCALE_SOS_ENDPOINT"],
    )
    bucket_name = os.environ["EXOSCALE_SOS_BUCKET_ML"]
    # remote_path = pathlib.Path("dataframes") / local_path.name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s from bucket %s to %s", remote_path, bucket_name, local_path)
    s3.download_file(bucket_name, str(remote_path), local_path)


def ensure_dataframe_is_available(local_path: pathlib.Path) -> None:
    """Download a dataframe from our Exoscale Simple Object Storage bucket if it is not already available locally."""
    logger = logging.getLogger("ensure_dataframe_is_available")
    if local_path.exists():
        logger.info("File %s already exists locally.", local_path)
        return
    download_file_from_exoscale(pathlib.Path("dataframes") / local_path.name, local_path)
