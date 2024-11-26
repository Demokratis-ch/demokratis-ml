"""Define custom block *types* for Prefect.

Configured block *instances* are created in the file `create_blocks.py`.
"""

import pathlib

import prefect.blocks.core
import prefect.filesystems
import pydantic


class DemokratisAPICredentials(prefect.blocks.core.Block):
    """Credentials used for HTTP Basic Auth to the Demokratis API."""

    username: str
    password: pydantic.SecretStr


DemokratisAPICredentials.register_type_and_schema()


class ExtendedLocalFileSystem(prefect.filesystems.LocalFileSystem):
    """Local filesystem with extra methods."""

    _block_type_name = "Extended Local File System"

    def path_exists(self, path: str | pathlib.Path) -> bool:
        """Check if a path exists."""
        return self._resolve_path(str(path)).exists()
