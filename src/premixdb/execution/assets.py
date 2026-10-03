"""Resolve immutable, bounded local assets and captured source objects."""

from pathlib import Path
from urllib.parse import unquote, urlsplit

from blake3 import blake3

from ..v1.storage_pb2 import ObjectRef


def read(
    ref: ObjectRef,
    *,
    local_root: str | Path | None = None,
    inline: bytes | None = None,
    limit: int = 64 * 1024 * 1024,
) -> bytes:
    if len(ref.blake3_digest) != 32 or ref.size_bytes > limit:
        raise ValueError("asset requires a digest and bounded size")
    if inline is not None:
        data = inline
    else:
        uri = urlsplit(ref.uri)
        if uri.scheme != "file" or local_root is None:
            raise ValueError("assets require a local file URI and authorized root")
        path = Path(unquote(uri.path)).resolve()
        if uri.netloc not in ("", "localhost") or not path.is_relative_to(
            Path(local_root).resolve()
        ):
            raise ValueError("asset escapes the authorized root")
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
    if (
        len(data) > limit
        or (ref.size_bytes and len(data) != ref.size_bytes)
        or blake3(data).digest() != ref.blake3_digest
    ):
        raise ValueError("asset integrity check failed")
    return data
