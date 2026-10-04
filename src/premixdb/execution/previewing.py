"""Published document descriptors let readers fetch only requested preview frames."""

from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING, Iterable, Sequence

from .._protobuf import parse
from .._typing import Interval, json_object, load_json
from ..engine.records import decode_document
from .storage import ObjectStore

if TYPE_CHECKING:
    from ..engine.contracts import DocumentRecord
    from ..engine.queries import Row
    from .catalog_reader import Catalog

from blake3 import blake3

from .. import _requests
from ..internal import derivation_pb2 as d
from ..v1 import query_pb2 as q
from ..v1 import snapshot_pb2 as s
from ..v1 import status_pb2 as status

SHARD_BYTES = 1024 * 1024


def bounded_text(store: ObjectStore, row: Row, width: int) -> tuple[str, bool]:
    """Read a verified text prefix, including any retained source ranges."""
    from .selections import stored_selection

    source, ranges = stored_selection(row)
    return _text(store, source, width, ranges)


def inline(
    store: ObjectStore, preview: q.QueryPreview | s.SnapshotPreview, rows: Iterable[Row]
) -> None:
    """Fill the first ten examples using only the frames needed for their prefix."""
    for row in islice(rows, 10):
        text, truncated = bounded_text(store, row, 1024)
        preview.documents.add(
            id=bytes.fromhex(row.id),
            text=text,
            truncated=truncated,
            source_key=row.source_key,
            corpus_id=bytes.fromhex(row.corpus_id),
            ordinal=row.ordinal,
        )


def publish(store: ObjectStore, kind: str, identity: bytes, rows: Iterable[Row]) -> None:
    from .selections import selection_record

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
        record = selection_record(row)
        width = record.ByteSize() + 10
        if shard.rows and size + width > SHARD_BYTES:
            flush()
            size = 0
        shard.rows.append(record)
        size += width
    if shard.rows:
        flush()
    store.save(kind, identity, index, suffix=".preview-index")


def _text(
    store: ObjectStore,
    source: DocumentRecord,
    width: int,
    ranges: Sequence[Interval] | None = None,
) -> tuple[str, bool]:
    intervals = list(ranges) if ranges is not None else [(0, source["bytes"])]
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
    selected = iter(intervals)
    interval = next(selected, None)
    for frame in source["frames"]:
        first, end = cursor, cursor + frame["bytes"]
        cursor = end
        while interval is not None and interval[1] <= first:
            interval = next(selected, None)
        if interval is None:
            break
        if interval[0] >= end:
            continue
        digest = bytes(frame["digest"])
        data = store._get("snapshot/objects/" + digest.hex(), frame["bytes"])
        if len(data) != frame["bytes"] or blake3(data).digest() != digest:
            raise ValueError("preview frame integrity check failed")
        while interval is not None and interval[0] < end:
            a, b = interval
            value = data[max(first, a) - first : min(end, b) - first].decode("utf-8")
            needed = width + 1 - characters
            chunks.append(value[:needed])
            characters += min(needed, len(value))
            if characters > width:
                return "".join(chunks)[:width], True
            if b > end:
                break
            interval = next(selected, None)
    return "".join(chunks), False


def preview(catalog: Catalog, request: q.PreviewRequest) -> q.PreviewResponse:
    limit, offset, width = _requests._preview_options(
        request.limit if request.HasField("limit") else 3,
        request.offset,
        request.max_characters if request.HasField("max_characters") else 1024,
        unit="documents",
    )
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
    if not limit or offset >= count:
        return result
    end = min(offset + limit, count)
    if width <= 1024 and end <= len(resource.preview.documents):
        for doc in resource.preview.documents[offset:end]:
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
        if chunk.first >= end or cursor <= offset:
            continue
        try:
            data = store.read_object(namespace, chunk.object)
        except KeyError as exc:
            raise ValueError("stored selection shard is missing") from exc
        rows = parse(d.SelectionShard, data).rows
        if len(rows) != chunk.count:
            raise ValueError("preview shard coverage differs")
        for ordinal in range(max(offset, chunk.first), min(end, cursor)):
            row = rows[ordinal - chunk.first]
            source = decode_document(json_object(load_json(row.source_record)))
            ranges = [(r.start, r.end) for r in row.ranges] if row.transformed else None
            text, truncated = _text(store, source, width, ranges)
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
    if cursor != count or len(result.preview.documents) != end - offset:
        raise ValueError("preview index is incomplete")
    return result
