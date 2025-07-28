"""Prefect pipeline for uploading our datasets to Hugging Face; see the `publish_data` flow for details."""

import pathlib

import huggingface_hub
import prefect
import prefect.logging

from demokratis_ml.pipelines.lib import blocks, utils


@prefect.flow(
    task_runner=prefect.task_runners.ThreadPoolTaskRunner(max_workers=4),
)
@utils.slack_status_report()
def publish_data(use_remote_storage: bool, files: dict[pathlib.Path, str]) -> None:
    """
    Publish our datasets to Hugging Face.

    :param files: A dictionary mapping source file paths (within the appropriate file system) to destination file names
        in the Hugging Face dataset repository.
        The source files are read from the file system specified by `use_remote_storage`.
    """
    hf_token = blocks.HuggingFaceDatasetUploadCredentials.load("huggingface-dataset-upload-credentials").token
    hf_api = huggingface_hub.HfApi(token=hf_token.get_secret_value())
    fs = utils.get_dataframe_storage(use_remote_storage)
    futures = upload_to_huggingface.map(
        source_file=list(files.keys()),
        destination_file_name=list(files.values()),
        fs=prefect.unmapped(fs),
        hf_api=prefect.unmapped(hf_api),
        repository_id=prefect.unmapped("demokratis/consultation-documents"),
    )
    futures.wait()


@prefect.task(
    task_run_name="upload_to_huggingface({destination_file_name})",
)
def upload_to_huggingface(
    fs: blocks.ExtendedFileSystemType,
    source_file: pathlib.Path,
    destination_file_name: str,
    hf_api: huggingface_hub.HfApi,
    repository_id: str,
) -> None:
    """Upload a file to our Hugging Face dataset repository."""
    logger = prefect.logging.get_run_logger()
    logger.info("Reading file %s from %s", source_file, fs.__class__.__name__)
    data = fs.read_path(str(source_file))
    assert isinstance(data, bytes)
    logger.info(
        "Uploading %.1fMB to Hugging Face repository %s, file %s",
        len(data) / 1024**2,
        repository_id,
        destination_file_name,
    )
    info = hf_api.upload_file(
        repo_id=repository_id,
        path_in_repo=destination_file_name,
        path_or_fileobj=data,
        repo_type="dataset",
        commit_message=f"Upload {destination_file_name} from {source_file.name}",
    )
    logger.info("Commit info: %r", info)
