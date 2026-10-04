"""Shared storage root with corpus/snapshot/query/dataset/mixture object namespaces."""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import Self
from urllib.parse import urlsplit

from blake3 import blake3
from google.protobuf.message import Message

from premixdb.storage.metadata import MetadataStore
from premixdb.storage.publication import publish
from premixdb.v1.status_pb2 import STATUS_ERROR
from premixdb.v1.storage_pb2 import ObjectProfile, ObjectRef, SpanRef

PREFIXES = frozenset(
    (
        "corpus",
        "snapshot",
        "query",
        "dataset",
        "mixture",
        "derivation",
        "field",
        "index",
        "execution",
        "submission",
        "tokenizer",
    )
)


class ObjectStore:
    def __init__(
        self,
        location: str | Path,
        *,
        metadata_path: str | Path | None = None,
        read_only: bool = False,
    ) -> None:
        if urlsplit(str(location)).scheme:
            raise ValueError("storage must be a local filesystem path")
        self.root = Path(location).resolve()
        self.read_only = read_only
        self._closed = False
        self._lock = RLock()
        path = Path(metadata_path) if metadata_path is not None else self.root / "metadata.sqlite3"
        self.metadata = MetadataStore(path, read_only=read_only)
        try:
            if not read_only:
                self._migrate_metadata()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.metadata.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _legacy_names(self, prefix: str) -> builtins.list[str]:
        return [path.name for path in (self.root / prefix).glob("*.ref")]

    def _migrate_metadata(self) -> None:
        """Resumable import; keep legacy files intact and verify every resource."""
        if self.metadata.migrated("legacy-refs-v1"):
            return
        for prefix in sorted(PREFIXES):
            for name in sorted(self._legacy_names(prefix)):
                stem = name.removesuffix(".ref")
                identity, dot, tail = stem.partition(".")
                if (
                    not name.endswith(".ref")
                    or not identity
                    or any(c not in "0123456789abcdef" for c in identity)
                    or len(identity) % 2
                ):
                    continue
                id, suffix = bytes.fromhex(identity), dot + tail
                if self.metadata.contains(prefix, id, suffix=suffix):
                    continue
                ref = SpanRef.FromString(self._get(f"{prefix}/{name}"))
                data = self._get(f"{prefix}/objects/{ref.object.blake3_digest.hex()}")
                if (
                    ref.start != 0
                    or ref.end != len(data)
                    or (blake3(data).digest() != ref.blake3_digest)
                ):
                    raise ValueError("stored resource integrity check failed")
                self.metadata.save_bytes(prefix, id, data, suffix=suffix)
        self.metadata.mark_migrated("legacy-refs-v1")

    def object_uri(self, relative: str | Path) -> str:
        return (self.root / relative).as_uri()

    def _put(self, relative: str | Path, data: bytes) -> None:
        """Write once; a concurrent publisher must have identical bytes."""
        path = self.root / relative
        if not publish(path, data) and path.read_bytes() != data:
            raise ValueError("conflicting immutable storage object")

    def _get(self, relative: str | Path, limit: int = 64 * 1024 * 1024) -> bytes:
        try:
            with (self.root / relative).open("rb") as stream:
                data = stream.read(max(0, limit + 1))
        except FileNotFoundError:
            raise KeyError(relative) from None
        if len(data) > limit:
            raise ValueError("storage object exceeds size limit")
        return data

    def put(self, prefix: str, data: bytes, *, profile: ObjectProfile | None = None) -> ObjectRef:
        if self.read_only:
            raise PermissionError("catalog is read-only")
        if prefix not in PREFIXES:
            raise ValueError("invalid storage namespace")
        digest = blake3(data).digest()
        relative = f"{prefix}/objects/{digest.hex()}"
        with self._lock:
            self._put(relative, data)
        return ObjectRef(
            blake3_digest=digest,
            profile=profile,
            size_bytes=len(data),
            uri=self.object_uri(relative),
        )

    def save(
        self, prefix: str, id: bytes, message: Message, *, suffix: str = "", failure: bool = False
    ) -> None:
        if prefix not in PREFIXES:
            raise ValueError("invalid storage namespace")
        from premixdb.v1 import corpus_pb2 as c
        from premixdb.v1 import data_mixture_pb2 as d
        from premixdb.v1 import query_pb2 as q

        if failure and (
            not isinstance(message, (q.Query, d.Dataset))
            or suffix != ".failed"
            or message.id != id
            or message.status != STATUS_ERROR
        ):
            raise ValueError("only failed execution status pointers are mutable")
        head = prefix == "corpus" and suffix == ".latest"
        if head and (
            not isinstance(message, c.Corpus)
            or message.id != id
            or len(message.latest_snapshot_id) != 32
        ):
            raise ValueError("corpus heads require a corpus and a complete snapshot ID")
        mutable = failure or head
        if self.read_only:
            raise PermissionError("catalog is read-only")
        self.metadata.save(prefix, id, message, suffix=suffix, mutable=mutable)

    def read_object(self, prefix: str, ref: ObjectRef) -> bytes:
        if prefix not in PREFIXES or len(ref.blake3_digest) != 32:
            raise ValueError("invalid stored object reference")
        data = self._get(f"{prefix}/objects/{ref.blake3_digest.hex()}")
        if len(data) != ref.size_bytes or blake3(data).digest() != ref.blake3_digest:
            raise ValueError("enrichment object integrity check failed")
        return data

    def load[T: Message](
        self, prefix: str, id: bytes, message_type: type[T], *, suffix: str = ""
    ) -> T:
        return self.metadata.load(prefix, id, message_type, suffix=suffix)

    def list[T: Message](
        self, prefix: str, message_type: type[T], *, suffix: str = ""
    ) -> builtins.list[T]:
        return self.metadata.list(prefix, message_type, suffix=suffix)

    def verify_snapshot_metadata(self, id: bytes) -> None:
        """Verify committed manifest and inventory pages without reading text frames."""
        from premixdb.engine.snapshots import COMMIT_HEADER

        name = id.hex()
        commit = self._get(f"snapshot/snapshots/{name}", len(COMMIT_HEADER) + 32)
        if len(commit) != len(COMMIT_HEADER) + 32 or not commit.startswith(COMMIT_HEADER):
            raise ValueError("invalid snapshot commit")

        def object(digest: bytes | list[int]) -> bytes:
            digest = bytes(digest)
            if len(digest) != 32:
                raise ValueError("invalid snapshot object digest")
            relative = f"snapshot/objects/{digest.hex()}"
            data = self._get(relative)
            if blake3(data).digest() != digest:
                raise ValueError("snapshot object integrity check failed")
            return data

        manifest = json.loads(object(commit[-32:]))
        for page in manifest.get("pages") or []:
            json.loads(object(page["digest"]))
