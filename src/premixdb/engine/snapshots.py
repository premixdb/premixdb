"""Immutable plain-text snapshots and verified, backward-compatible storage."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import Iterable, Iterator, Protocol

from blake3 import blake3

from .._files import publish as _publish
from .._inputs import Source
from .._typing import (
    JSON,
    json_integer,
    json_integers,
    json_list,
    json_object,
    json_string,
    load_json,
)
from .contracts import (
    Changes,
    Counts,
    DocumentRecord,
    Frame,
    Manifest,
    ObjectRecord,
    ObjectSpan,
    SnapshotDescription,
    TextProfile,
)
from .identity import Canonical, CodeVersion, digest, identity_domain
from .records import FRAME_BYTES as FRAME_BYTES
from .records import (
    decode_document,
    decode_frame,
    schema,
    text_profile,
    totals,
    validate_document,
)

PAGE_BYTES = 1024 * 1024
MANIFEST_BYTES = 64 * 1024 * 1024
COMMIT_HEADER = identity_domain("snapshot-commit")


def encode(value: object) -> bytes:
    import json

    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


class CountedDocument(Protocol):
    @property
    def size(self) -> int: ...
    @property
    def characters(self) -> int: ...


def counts(documents: Iterable[CountedDocument]) -> Counts:
    result: Counts = dict(documents=0, bytes=0, characters=0)
    for document in documents:
        result["documents"] += 1
        result["bytes"] += document.size
        result["characters"] += document.characters
    return result


@dataclass(frozen=True)
class Document:
    corpus_id: str
    source_key: str
    text: str

    def __post_init__(self) -> None:
        if not self.source_key.strip():
            raise ValueError("source key must not be empty")
        self.source_key.encode()
        self.text.encode()

    @cached_property
    def id(self) -> str:
        return (
            Canonical("document")
            .fixed(digest(self.corpus_id, 16))
            .string(self.source_key)
            .string("0")
            .string(self.text)
            .u64(0)
            .finish()
            .hex()
        )

    @cached_property
    def content(self) -> bytes:
        return blake3(self.text.encode()).digest()

    @cached_property
    def size(self) -> int:
        return len(self.text.encode())

    @property
    def characters(self) -> int:
        return len(self.text)


class FrameReader(Protocol):
    def _frame(self, frame: Frame) -> bytes: ...


class StoredDocument(Document):
    """A captured document whose text is read only for the current operation."""

    record: DocumentRecord
    store: "FrameReader"

    def __init__(
        self, corpus_id: str, key: str, record: DocumentRecord, store: FrameReader, identity: str
    ) -> None:
        object.__setattr__(self, "corpus_id", corpus_id)
        object.__setattr__(self, "source_key", key)
        object.__setattr__(self, "record", record)
        object.__setattr__(self, "store", store)
        object.__setattr__(self, "id", identity)

    @property
    def content(self) -> bytes:
        return bytes(self.record["content"])

    @property
    def size(self) -> int:
        return self.record["bytes"]

    @property
    def characters(self) -> int:
        return sum(frame["profile"]["characters"] for frame in self.record["frames"])

    @property
    def text(self) -> str:
        data = b"".join(self.store._frame(frame) for frame in self.record["frames"])
        if len(data) != self.size or blake3(data).digest() != self.content:
            raise RuntimeError("stored document integrity mismatch")
        return data.decode()


class Snapshot:
    def __init__(
        self,
        corpus: str,
        sources: Iterable[tuple[str, str | Document]],
        code: CodeVersion,
        base: Snapshot | None = None,
    ) -> None:
        self.corpus_id = digest(corpus, 16).hex()
        self.code = code
        self.base = base.id if base else None
        if base and base.corpus_id != self.corpus_id:
            raise ValueError("base belongs to another corpus")
        documents = {}
        changes: Changes = Changes(added=0, changed=0, removed=0, unchanged=0, reused=0)
        for key, text in sources:
            if key in documents:
                raise ValueError("duplicate source key")
            doc = text if isinstance(text, Document) else Document(self.corpus_id, key, text)
            if doc.corpus_id != self.corpus_id or doc.source_key != key:
                raise ValueError("captured document belongs to different inputs")
            old = base.documents.get(key) if base else None
            same = old is not None and old.content == doc.content
            changes["unchanged" if same else "changed" if old else "added"] += 1
            reuse = same and base is not None and base.code == code
            changes["reused"] += int(reuse)
            documents[key] = old if reuse and old is not None else doc
        self.documents = dict(sorted(documents.items()))
        changes["removed"] = len(set(base.documents) - documents.keys()) if base else 0
        self._changes = changes
        self._summary = counts(self.documents.values())
        inventory = Canonical("inventory").u64(len(documents))
        for key, doc in self.documents.items():
            inventory.string(key).fixed(doc.content).u64(doc.size)
        self.inventory = inventory.finish()
        members = sorted({d.id for d in documents.values()})
        membership = Canonical("membership").u64(len(members))
        for member in members:
            membership.fixed(digest(member))
        self.membership = membership.finish()
        self._id = self.identity()

    @property
    def id(self) -> str:
        return self._id

    def identity(self) -> str:
        identity = Canonical("snapshot").fixed(digest(self.corpus_id, 16))
        identity.fixed(bytes([self.base is not None]))
        if self.base:
            identity.fixed(digest(self.base))
        return (
            identity.fixed(self.inventory)
            .string("plain-text/v1")
            .fixed(self.code.canonical_digest())
            .finish()
            .hex()
        )

    def summary(self) -> Counts:
        return self._summary.copy()

    def changes(self) -> Changes:
        return self._changes.copy()


def _decode_manifest(raw: dict[str, JSON]) -> Manifest:
    schema(
        raw,
        (
            "version",
            "id",
            "corpus",
            "code",
            "base",
            "inventory",
            "membership",
            "summary",
            "changes",
        ),
        ("documents", "pages"),
    )
    summary = json_object(raw["summary"])
    changes = json_object(raw["changes"])
    code = json_object(raw["code"])
    totals(summary, ("documents", "bytes", "characters"))
    totals(changes, ("added", "changed", "removed", "unchanged", "reused"))
    schema(code, ("repository", "commit", "environment"))
    result = Manifest(
        version=json_integer(raw["version"]),
        id=json_integers(raw["id"]),
        corpus=json_integers(raw["corpus"]),
        inventory=json_integers(raw["inventory"]),
        membership=json_integers(raw["membership"]),
        base=json_integers(raw["base"]) if raw["base"] is not None else None,
        code=dict(
            repository=json_string(code["repository"]),
            commit=json_string(code["commit"]),
            environment=json_integers(code["environment"]),
        ),
        summary=Counts(
            documents=json_integer(summary["documents"]),
            bytes=json_integer(summary["bytes"]),
            characters=json_integer(summary["characters"]),
        ),
        changes=Changes(
            added=json_integer(changes["added"]),
            changed=json_integer(changes["changed"]),
            removed=json_integer(changes["removed"]),
            unchanged=json_integer(changes["unchanged"]),
            reused=json_integer(changes["reused"]),
        ),
    )
    if "pages" in raw:
        result["pages"] = [decode_frame(p) for p in json_list(raw["pages"])]
    if "documents" in raw:
        result["documents"] = [decode_document(json_object(d)) for d in json_list(raw["documents"])]
    return result


class Store:
    def __init__(self, path: str | PathLike[str]) -> None:
        self.root = Path(path).resolve()

    def _object(self, value: bytes | list[int], limit: int = MANIFEST_BYTES) -> bytes:
        try:
            with (self.root / "objects" / bytes(value).hex()).open("rb") as stream:
                data = stream.read(limit + 1)
        except OSError as exc:
            raise RuntimeError("cannot read committed object") from exc
        if len(data) > limit or blake3(data).digest() != bytes(value):
            raise RuntimeError("object checksum or size mismatch")
        return data

    def _put(self, data: bytes) -> list[int]:
        value = blake3(data).digest()
        _publish(self.root / "objects" / value.hex(), data)
        if self._object(value, len(data)) != data:
            raise RuntimeError("conflicting content-addressed object")
        return list(value)

    def _manifest(self, id: str) -> Manifest:
        id = digest(id).hex()
        try:
            with (self.root / "snapshots" / id).open("rb") as stream:
                commit = stream.read(len(COMMIT_HEADER) + 33)
        except FileNotFoundError as exc:
            raise ValueError("snapshot is not committed") from exc
        if len(commit) != len(COMMIT_HEADER) + 32 or not commit.startswith(COMMIT_HEADER):
            raise RuntimeError("invalid snapshot completion record")
        raw = json_object(load_json(self._object(commit[-32:])))
        manifest = _decode_manifest(raw)
        schema(
            manifest,
            (
                "version",
                "id",
                "corpus",
                "code",
                "base",
                "inventory",
                "membership",
                "summary",
                "changes",
            ),
            ("documents", "pages"),
        )
        totals(manifest["summary"], ("documents", "bytes", "characters"))
        totals(manifest["changes"], ("added", "changed", "removed", "unchanged", "reused"))
        schema(manifest["code"], ("repository", "commit", "environment"))
        if manifest["id"] != list(bytes.fromhex(id)):
            raise RuntimeError("snapshot manifest ID mismatch")
        return manifest

    def _records(self, manifest: Manifest) -> Iterator[DocumentRecord]:
        """Validate inventory records one page at a time, including global ordering."""

        def records() -> Iterator[DocumentRecord]:
            if manifest["version"] == 1 and "documents" in manifest and "pages" not in manifest:
                yield from manifest["documents"]
            elif manifest["version"] == 2 and "pages" in manifest and "documents" not in manifest:
                for page in manifest["pages"]:
                    rows = json_list(load_json(self._frame(page, PAGE_BYTES)))
                    if not rows:
                        raise RuntimeError("empty inventory page")
                    yield from (decode_document(json_object(row)) for row in rows)
            else:
                raise RuntimeError("invalid snapshot manifest layout or version")

        previous, count = None, 0
        for record in records():
            validate_document(record)
            key = record["key"]
            if previous is not None and key <= previous:
                raise RuntimeError("invalid snapshot inventory")
            previous, count = key, count + 1
            yield record
        if count != manifest["summary"]["documents"]:
            raise RuntimeError("invalid snapshot inventory")

    def _frame(self, frame: Frame, limit: int = 8 * FRAME_BYTES) -> bytes:
        schema(frame, ("digest", "bytes"), ("profile",))
        size = frame["bytes"]
        if len(bytes(frame["digest"])) != 32 or type(size) is not int or not 0 < size <= limit:
            raise RuntimeError("invalid frame size")
        data = self._object(frame["digest"], size)
        if len(data) != size:
            raise RuntimeError("frame size mismatch")
        return data

    def capture(
        self,
        corpus: str,
        sources: Iterable[Source],
        code: CodeVersion,
        base: Snapshot | None = None,
    ) -> Snapshot:
        return self.capture_inputs(corpus, ((s.key, s.text) for s in sources), (), code, base)

    def capture_inputs(
        self,
        corpus: str,
        texts: Iterable[tuple[str, str]],
        files: Iterable[tuple[str, str | PathLike[str]]],
        version: CodeVersion | tuple[str, str, str],
        base: Snapshot | None = None,
        *,
        stream: bool = False,
    ) -> Snapshot:
        corpus_id = digest(corpus, 16).hex()

        def sources() -> Iterator[tuple[str, str | Document]]:
            try:
                for key, text in chain(
                    texts, ((key, Path(path).read_bytes().decode()) for key, path in files)
                ):
                    if stream:
                        doc = Document(corpus_id, key, text)
                        record = self._document_record(doc, FRAME_BYTES)
                        yield key, StoredDocument(doc.corpus_id, key, record, self, doc.id)
                    else:
                        yield key, text
            except (OSError, UnicodeError) as exc:
                raise ValueError(str(exc)) from exc

        snapshot = Snapshot(
            corpus,
            sources(),
            version if isinstance(version, CodeVersion) else CodeVersion(*version),
            base,
        )
        if base and snapshot.inventory == base.inventory:
            self.save(base)
            return base
        self.save(snapshot)
        return snapshot

    def _document_record(self, doc: Document, frame_bytes: int) -> DocumentRecord:
        if (
            isinstance(doc, StoredDocument)
            and isinstance(doc.store, Store)
            and doc.store.root == self.root
        ):
            return doc.record
        data = doc.text.encode()
        frames: list[Frame] = []
        start = 0
        while start < len(data):
            end = min(start + frame_bytes, len(data))
            while end < len(data) and data[end] & 0xC0 == 0x80:
                end -= 1
            part = data[start:end]
            frames.append(dict(digest=self._put(part), bytes=len(part), profile=text_profile(part)))
            start = end
        return DocumentRecord(
            key=doc.source_key, content=list(doc.content), bytes=len(data), frames=frames
        )

    def save(self, snapshot: Snapshot, frame_bytes: int = FRAME_BYTES) -> None:
        if not 4 <= frame_bytes <= 8 * FRAME_BYTES:
            raise ValueError("invalid frame size")
        if (self.root / "snapshots" / snapshot.id).exists():
            if self.load(snapshot.id, lazy=True)._changes != snapshot._changes:
                raise RuntimeError("conflicting snapshot change counts")
            return
        pages: list[Frame] = []
        page = bytearray(b"[")
        for key, doc in snapshot.documents.items():
            record = self._document_record(doc, frame_bytes)
            raw = encode(record)
            if len(raw) + 2 > PAGE_BYTES:
                raise ValueError("document metadata exceeds manifest page size limit")
            if len(page) > 1 and len(page) + len(raw) + 2 > PAGE_BYTES:
                data = bytes(page) + b"]"
                pages.append(dict(digest=self._put(data), bytes=len(data)))
                page = bytearray(b"[")
            if len(page) > 1:
                page.extend(b",")
            page.extend(raw)
        if len(page) > 1:
            data = bytes(page) + b"]"
            pages.append(dict(digest=self._put(data), bytes=len(data)))
        manifest: Manifest = Manifest(
            version=2,
            id=list(digest(snapshot.id)),
            corpus=list(digest(snapshot.corpus_id, 16)),
            code=dict(
                repository=snapshot.code.repository,
                commit=snapshot.code.commit,
                environment=list(digest(snapshot.code.environment)),
            ),
            base=list(digest(snapshot.base)) if snapshot.base else None,
            inventory=list(snapshot.inventory),
            membership=list(snapshot.membership),
            summary=snapshot._summary,
            changes=snapshot._changes,
            pages=pages,
        )
        raw = encode(manifest)
        if len(raw) > MANIFEST_BYTES:
            raise ValueError("snapshot manifest exceeds size limit")
        value = self._put(raw)
        self._restore(manifest, lazy=True)
        _publish(self.root / "snapshots" / snapshot.id, COMMIT_HEADER + bytes(value))
        self.load(snapshot.id, lazy=True)

    def load(self, id: str, *, lazy: bool = False) -> Snapshot:
        return self._restore(self._manifest(id), lazy=lazy)

    def _restore(self, manifest: Manifest, *, lazy: bool = False) -> Snapshot:
        sources = []
        corpus_id = bytes(manifest["corpus"]).hex()
        for record in self._records(manifest):
            parts = []
            size = 0
            content_hash = blake3()
            identity = (
                Canonical("document")
                .fixed(digest(corpus_id, 16))
                .string(record["key"])
                .string("0")
                .u64(record["bytes"])
            )
            for frame in record["frames"]:
                part = self._frame(frame)
                text = part.decode()
                if "profile" in frame and frame["profile"] != text_profile(part):
                    raise RuntimeError("text profile mismatch")
                size += len(part)
                if size > record["bytes"]:
                    raise RuntimeError("document size exceeds manifest")
                content_hash.update(part)
                identity.fixed(part)
                if not lazy:
                    parts.append(text)
                elif "profile" not in frame:
                    frame["profile"] = text_profile(part)
            if size != record["bytes"] or content_hash.digest() != bytes(record["content"]):
                raise RuntimeError("document size or content mismatch")
            sources.append(
                (
                    record["key"],
                    StoredDocument(
                        corpus_id, record["key"], record, self, identity.u64(0).finish().hex()
                    )
                    if lazy
                    else "".join(parts),
                )
            )
        code = manifest["code"]
        result = Snapshot(
            bytes(manifest["corpus"]).hex(),
            sources,
            CodeVersion(code["repository"], code["commit"], bytes(code["environment"]).hex()),
        )
        changes = manifest["changes"]
        if (
            changes["added"] + changes["changed"] + changes["unchanged"] != len(sources)
            or changes["reused"] > changes["unchanged"]
            or any(type(v) is not int or v < 0 for v in changes.values())
            or (manifest["base"] is None and changes != result._changes)
        ):
            raise RuntimeError("invalid snapshot change counts")
        result.base = bytes(manifest["base"]).hex() if manifest["base"] else None
        result._changes = changes
        result._id = result.identity()
        if (
            list(digest(result.id)) != manifest["id"]
            or list(result.inventory) != manifest["inventory"]
            or list(result.membership) != manifest["membership"]
            or result._summary != manifest["summary"]
        ):
            raise RuntimeError("snapshot identity or summary mismatch")
        return result

    def describe(self, id: str) -> SnapshotDescription:
        m = self._manifest(id)
        return SnapshotDescription(
            corpus_id=bytes(m["corpus"]).hex(),
            parent_snapshot_id=bytes(m["base"]).hex() if m["base"] else "",
            changes=deepcopy(m["changes"]),
        )

    def objects(self, id: str) -> dict[str, ObjectRecord]:
        rows: dict[str, ObjectRecord] = {}
        for record in self._records(self._manifest(id)):
            spans: list[ObjectSpan] = []
            start = 0
            profile: TextProfile = TextProfile(content_bytes=0, characters=0, newlines=0)
            for frame in record["frames"]:
                p = frame.get("profile")
                if p is None or p["content_bytes"] != frame["bytes"]:
                    raise RuntimeError("missing or invalid text profile")
                end = start + frame["bytes"]
                spans.append(
                    dict(start=start, end=end, content_digest=bytes(frame["digest"]).hex(), **p)
                )
                start = end
                profile["content_bytes"] += p["content_bytes"]
                profile["characters"] += p["characters"]
                profile["newlines"] += p["newlines"]
            if start != record["bytes"]:
                raise RuntimeError("document size mismatch")
            rows[record["key"]] = dict(
                spans=len(record["frames"]),
                content_digest=bytes(record["content"]).hex(),
                **profile,
                span_refs=spans,
            )
        return rows
