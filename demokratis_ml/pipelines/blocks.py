"""Define custom block *types* for Prefect.

Configured block *instances* are created in the file `create_blocks.py`.
"""

import pathlib
from typing import IO, cast

import openai
import prefect.blocks.abstract
import prefect.blocks.core
import prefect.filesystems
import pydantic


class DemokratisAPICredentials(prefect.blocks.core.Block):
    """Credentials used for HTTP Basic Auth to the Demokratis API."""

    username: str
    password: pydantic.SecretStr


DemokratisAPICredentials.register_type_and_schema()


class HuggingFaceDatasetUploadCredentials(prefect.blocks.core.Block):
    """Authentication token for uploading datasets to HuggingFace."""

    token: pydantic.SecretStr


HuggingFaceDatasetUploadCredentials.register_type_and_schema()


class OpenAICredentials(prefect.blocks.abstract.CredentialsBlock):
    """Credentials used to authenticate with OpenAI.

    Lifted from https://github.com/PrefectHQ/prefect-openai/blob/main/prefect_openai/credentials.py
    which is now unmaintained.
    """

    api_key: pydantic.SecretStr
    organization: str | None = None

    def get_client(self) -> openai.OpenAI:
        """Return an OpenAI client using the credentials in this block."""
        return openai.OpenAI(api_key=self.api_key.get_secret_value(), organization=self.organization)


OpenAICredentials.register_type_and_schema()


class MLflowCredentials(prefect.blocks.core.Block):
    """Credentials used for HTTP Basic Auth to MLflow."""

    tracking_uri: str
    username: str
    password: pydantic.SecretStr


MLflowCredentials.register_type_and_schema()


class ExtendedLocalFileSystem(prefect.filesystems.LocalFileSystem):
    """Local filesystem with extra methods."""

    _block_type_name = "Extended Local File System"

    def iterdir(self, path: str | pathlib.Path = "") -> list[pathlib.Path]:
        """Iterate over the contents of a directory.

        Paths are relative to ``self.basepath``.
        """
        assert self.basepath is not None
        base = self._resolve_path("")
        path = self._resolve_path(str(path))
        return [pathlib.Path(p).relative_to(base) for p in path.iterdir()]

    def open(self, path: str | pathlib.Path, mode: str) -> IO:
        """Open a file on the local file system and return a file-like object.

        The resulting object should be used as a context manager so it is properly closed afterwards.
        """
        return self._resolve_path(str(path)).open(mode)

    def path_exists(self, path: str | pathlib.Path) -> bool:
        """Check if a path exists."""
        return self._resolve_path(str(path)).exists()


class ExtendedRemoteFileSystem(prefect.filesystems.RemoteFileSystem):
    """Remote filesystem with extra methods."""

    _block_type_name = "Extended Remote File System"

    def iterdir(self, path: str | pathlib.Path = "") -> list[pathlib.Path]:
        """Iterate over the contents of a directory.

        Paths are relative to ``self.basepath``.
        """
        assert self.basepath.startswith("s3://")
        base = pathlib.Path(self.basepath[len("s3://") :])
        path = self._resolve_path(str(path))
        return [pathlib.Path(p).relative_to(base) for p in self.filesystem.ls(path, detail=False)]

    def open(self, path: str | pathlib.Path, mode: str) -> IO:
        """Open a file on the remote file system and return a file-like object.

        The resulting object should be used as a context manager so it is properly closed afterwards.
        """
        return cast(IO, self.filesystem.open(self._resolve_path(str(path)), mode))

    def path_exists(self, path: str | pathlib.Path) -> bool:
        """Check if a path exists."""
        return self.filesystem.exists(self._resolve_path(str(path)))


ExtendedFileSystemType = ExtendedRemoteFileSystem | ExtendedLocalFileSystem
