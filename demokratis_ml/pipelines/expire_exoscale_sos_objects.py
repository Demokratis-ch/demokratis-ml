"""Implement expiration of objects in Exoscale Object Storage."""

import datetime

import prefect

from demokratis_ml.pipelines.lib import blocks


@prefect.flow
def expire_exoscale_sos_objects(
    storage_block_name: str,
    path_glob: str,
    max_age_days: int,
    dry_run: bool = False,
) -> None:
    """Expire objects in Exoscale Object Storage that are older than max_age_days.

    We have to implement this ourselves because the Exoscale Object Storage API does not provide a built-in
    expiration mechanism yet. See
    https://community.exoscale.com/product/storage/object-storage/how-to/bucketlifecycle/#lifecycle-on-object-storage
    """
    logger = prefect.get_run_logger()
    fs = blocks.ExtendedRemoteFileSystem.load(storage_block_name)
    now = datetime.datetime.now(tz=datetime.UTC)
    paths = fs.glob(path_glob)
    mb_deleted = 0
    mb_kept = 0
    files_deleted = 0
    for path in sorted(paths):
        info = fs.info(path)
        size_mb = info["size"] / 1024**2
        age_days = (now - info["LastModified"]).days
        if age_days > max_age_days:
            logger.info("DELETING %s (%d days old, %.0f MB)", path, age_days, size_mb)
            if dry_run:
                logger.warning("Dry run, not deleting %s", path)
            else:
                fs.rm(path)
            mb_deleted += size_mb
            files_deleted += 1
        else:
            logger.info("Keeping %s (%d days old, %.0f MB)", path, age_days, size_mb)
            mb_kept += size_mb

    logger.info("Deleted %.0f MB (%d files), kept %.0f MB", mb_deleted, files_deleted, mb_kept)


if __name__ == "__main__":
    expire_exoscale_sos_objects(
        storage_block_name="remote-dataframe-storage",
        path_glob="*.parquet",
        max_age_days=100,
        dry_run=True,
    )
