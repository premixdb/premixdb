"""Capture pinned object manifests, local files and Hub dataset rows."""

from __future__ import annotations

from pathlib import Path
from typing import Generator

from premixdb.runtime.assets import read
from premixdb.v1.storage_pb2 import Source


def capture(
    source: Source, local_root: str | Path | None = None
) -> Generator[tuple[str, str], None, None]:
    if source.HasField("manifest"):
        for key, reference in sorted(source.manifest.objects.items()):
            yield key, read(reference, local_root=local_root).decode("utf-8")
    elif source.HasField("hugging_face"):
        from premixdb.runtime.hub_capture import capture as hub_capture

        yield from hub_capture(source)
    else:
        raise ValueError("unsupported source")
