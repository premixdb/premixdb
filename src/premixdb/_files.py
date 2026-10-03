"""Publish complete files once, with data and directory synchronization."""

import os
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile


def publish(path: Path, data: bytes | Iterable[bytes]) -> bool:
    """Return whether this writer published; leave an existing file untouched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".tmp-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.writelines((data,) if isinstance(data, bytes) else data)
            stream.flush()
            os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
                created = True
            except FileExistsError:
                created = False
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return created
        finally:
            temporary.unlink(missing_ok=True)
