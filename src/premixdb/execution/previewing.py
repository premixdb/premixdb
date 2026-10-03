"""Published document descriptors let readers fetch only requested preview frames."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Iterable

from .._protobuf import parse
from .._typing import json_object, load_json
from ..engine.curation import RetainedDocument
from ..engine.queries import Row
from ..engine.snapshots import StoredDocument, _decode_document
from .storage import ObjectStore

if TYPE_CHECKING:
    from .catalog_reader import Catalog

from blake3 import blake3

from .. import _requests
from ..internal import derivation_pb2 as d
from ..v1 import query_pb2 as q
from ..v1 import snapshot_pb2 as s
from ..v1 import status_pb2 as status

SHARD_BYTES = 1024 * 1024


def publish(store: ObjectStore, kind: str, identity: bytes, rows: Iterable[Row]) -> None:
    index, shard = d.DocumentIndex(resource_id=identity), d.SelectionShard()
    size = 0

    def flush() -> None:
        count = len(shard.rows)
        index.shards.add(
            object=store.put(kind, shard.SerializeToString(deterministic=True)),
            first=index.documents,
            count=count,
        )
        index.documents += count
        shard.Clear()

    for row in rows:
        original = (
            row.document.original if isinstance(row.document, RetainedDocument) else row.document
        )
        if not isinstance(original, StoredDocument):
            raise ValueError("previews require stored document frames")
        ranges = getattr(row.document, "source_ranges", None)
        record = d.SelectedDocument(
            document_id=bytes.fromhex(row.id),
            corpus_id=bytes.fromhex(row.corpus_id),
            source_record=json.dumps(
                original.record, sort_keys=True, separators=(",", ":")
            ).encode(),
            transformed=ranges is not None,
        )
        if ranges is not None:
            record.ranges.extend(d.ByteRange(start=a, end=b) for a, b in ranges)
        width = record.ByteSize() + 10
        if shard.rows and size + width > SHARD_BYTES:
            flush()
            size = 0
        shard.rows.append(record)
        size += width
    if shard.rows:
        flush()
    store.save(kind, identity, index, suffix=".preview-index")


def _text(store: ObjectStore, record: d.SelectedDocument, width: int) -> tuple[str, bool]:
    source = _decode_document(json_object(load_json(record.source_record)))
    intervals = (
        [(r.start, r.end) for r in record.ranges] if record.transformed else [(0, source["bytes"])]
    )
    if any(
        not 0 <= a < b <= source["bytes"] or i and intervals[i - 1][1] > a
        for i, (a, b) in enumerate(intervals)
    ):
        if intervals != [(0, 0)]:
            raise ValueError("invalid preview source ranges")
    if width == 0:
        return "", any(a < b for a, b in intervals)
    chunks: list[str] = []
    characters, cursor = 0, 0
    for frame in source["frames"]:
        first, end = cursor, cursor + frame["bytes"]
        cursor = end
        slices = [
            (max(first, a) - first, min(end, b) - first)
            for a, b in intervals
            if a < end and b > first
        ]
        if not slices:
            continue
        digest = bytes(frame["digest"])
        data = store._get("snapshot/objects/" + digest.hex(), frame["bytes"])
        if len(data) != frame["bytes"] or blake3(data).digest() != digest:
            raise ValueError("preview frame integrity check failed")
        for a, b in slices:
            value = data[a:b].decode("utf-8")
            needed = width + 1 - characters
            chunks.append(value[:needed])
            characters += min(needed, len(value))
            if characters > width:
                return "".join(chunks)[:width], True
    return "".join(chunks), False


def preview(catalog: Catalog, request: q.PreviewRequest) -> q.PreviewResponse:
    limit = request.limit if request.HasField("limit") else 3
    width = request.max_characters if request.HasField("max_characters") else 1024
    if limit > 1000 or width > 1_000_000:
        raise ValueError("preview supports at most 1000 documents and 1,000,000 characters")
    kind = request.WhichOneof("input")
    if kind is None:
        raise ValueError("preview requires a snapshot or query ID")
    identity = _requests._id(getattr(request, kind), 32)
    resource = (
        catalog.GetQuery(q.GetQueryRequest(id=identity)).query
        if kind == "query_id"
        else catalog.GetSnapshot(s.GetSnapshotRequest(id=identity)).snapshot
    )
    if resource.status != status.STATUS_COMPLETED:
        raise ValueError("preview requires a completed resource")
    count = (
        resource.profile.output_documents
        if isinstance(resource, q.Query)
        else resource.profile.documents
    )
    result = q.PreviewResponse()
    if not limit or request.offset >= count:
        return result
    end = min(request.offset + limit, count)
    if width <= 1024 and end <= len(resource.preview.documents):
        for doc in resource.preview.documents[request.offset : end]:
            item = result.preview.documents.add(
                id=doc.id,
                text=doc.text,
                truncated=doc.truncated,
                source_key=doc.source_key,
                corpus_id=doc.corpus_id,
                ordinal=doc.ordinal,
            )
            item.truncated |= len(item.text) > width
            item.text = item.text[:width]
        return result
    namespace = kind.removesuffix("_id")
    store = catalog._storage
    index = store.load(namespace, identity, d.DocumentIndex, suffix=".preview-index")
    if index.resource_id != identity or index.documents != count:
        raise ValueError("preview index belongs to a different resource")
    cursor, payload_bytes = 0, 0
    for chunk in index.shards:
        if chunk.first != cursor or not chunk.count:
            raise ValueError("preview index is incomplete or out of order")
        cursor += chunk.count
        if chunk.first >= end or cursor <= request.offset:
            continue
        try:
            data = store.read_object(namespace, chunk.object)
        except KeyError as exc:
            raise ValueError("stored selection shard is missing") from exc
        rows = parse(d.SelectionShard, data).rows
        if len(rows) != chunk.count:
            raise ValueError("preview shard coverage differs")
        for ordinal in range(max(request.offset, chunk.first), min(end, cursor)):
            row = rows[ordinal - chunk.first]
            source = _decode_document(json_object(load_json(row.source_record)))
            text, truncated = _text(store, row, width)
            item = result.preview.documents.add(
                id=row.document_id,
                corpus_id=row.corpus_id,
                source_key=source["key"],
                ordinal=ordinal,
                text=text,
                truncated=truncated,
            )
            payload_bytes += item.ByteSize() + 6
            if payload_bytes > 3 * 1024 * 1024:
                raise ValueError(
                    "preview exceeds the response limit; reduce limit or max_characters"
                )
    if cursor != count or len(result.preview.documents) != end - request.offset:
        raise ValueError("preview index is incomplete")
    return result
