"""Persist completed selections so eviction and restart never rerun their kernels."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Literal

from .._lineage import decode_lineage
from .._typing import Interval, json_object, load_json
from ..engine.contracts import Counts, DocumentRecord, Frame, QuerySummary
from ..v1 import query_pb2 as q
from .storage import ObjectStore

if TYPE_CHECKING:
    from .coordinator import Coordinator

from blake3 import blake3

from ..engine.curation import RetainedDocument, SelectedDocument
from ..engine.identity import CodeVersion
from ..engine.queries import CompletedQuery, Query, Row
from ..engine.records import decode_document, text_profile, validate_document
from ..engine.snapshots import StoredDocument
from ..internal import derivation_pb2 as d
from .catalog import wire

SHARD_BYTES = 8 * 1024 * 1024


def stored_selection(row: Row) -> tuple[DocumentRecord, tuple[Interval, ...] | None]:
    """Return captured source metadata and optional retained byte ranges."""
    original = row.document.original if isinstance(row.document, RetainedDocument) else row.document
    if not isinstance(original, StoredDocument):
        raise ValueError("document publication requires stored source documents")
    ranges = row.document.source_ranges if isinstance(row.document, RetainedDocument) else None
    return original.record, ranges


def selection_record(row: Row) -> d.SelectedDocument:
    import json

    source, ranges = stored_selection(row)
    record = d.SelectedDocument(
        document_id=bytes.fromhex(row.id),
        corpus_id=bytes.fromhex(row.corpus_id),
        source_record=json.dumps(source, sort_keys=True, separators=(",", ":")).encode(),
        transformed=ranges is not None,
    )
    if ranges is not None:
        record.ranges.extend(d.ByteRange(start=a, end=b) for a, b in ranges)
    return record


def publish(storage: ObjectStore, handle: Query) -> None:
    manifest = d.QuerySelection(
        query_id=bytes.fromhex(handle.id),
        repository=handle.code.repository,
        commit=handle.code.commit,
        environment=bytes.fromhex(handle.code.environment),
        occurrences=handle.row_count,
    )
    shard, size = d.SelectionShard(), 0

    def flush() -> None:
        manifest.shards.append(storage.put("query", wire(shard)))
        shard.Clear()

    for row in handle:
        record = selection_record(row)
        width = record.ByteSize() + 10
        if shard.rows and size + width > SHARD_BYTES:
            flush()
            size = 0
        shard.rows.append(record)
        size += width
    if shard.rows:
        flush()
    storage.save("query", manifest.query_id, manifest, suffix=".selection")


def restore(service: Coordinator, resource: q.Query) -> Query:
    storage = service._storage
    manifest = storage.load("query", resource.id, d.QuerySelection, suffix=".selection")
    if (
        manifest.query_id != resource.id
        or bytes.fromhex(manifest.commit) != resource.git_commit
        or manifest.occurrences != resource.profile.output_documents
    ):
        raise ValueError("stored query selection belongs to a different recipe")
    code = CodeVersion(manifest.repository, manifest.commit, manifest.environment.hex())
    provenance = decode_lineage(storage.read_object("query", resource.lineage))
    frames = _Frames(storage)
    rows: list[Row] = []
    for ref in manifest.shards:
        try:
            data = storage.read_object("query", ref)
        except KeyError as exc:
            raise ValueError("stored query selection shard is missing") from exc
        shard = d.SelectionShard()
        shard.ParseFromString(data)
        for record in shard.rows:
            id = record.document_id.hex()
            origin = provenance.get(id)
            if origin is None or origin["selection"]["kind"] != "retained":
                raise ValueError("stored query selection differs from its lineage")
            source = decode_document(json_object(load_json(record.source_record)))
            validate_document(source, require_profiles=True)
            if (
                len(record.document_id) != 32
                or record.corpus_id.hex() != origin["corpus_id"]
                or source["key"] != origin["source_key"]
                or bytes(source["content"]).hex() != origin["content"]
                or not origin["snapshots"]
                or not set(origin["snapshots"]) <= {id.hex() for id in resource.snapshot_ids}
            ):
                raise ValueError("stored selection source differs from its lineage")
            document: SelectedDocument = StoredDocument(
                record.corpus_id.hex(), source["key"], source, frames, id
            )
            ranges = tuple((r.start, r.end) for r in record.ranges)
            if record.transformed:
                if list(ranges) != origin.get("retained_ranges"):
                    raise ValueError("stored query ranges differ from their lineage")
                if any(
                    not 0 <= a < b <= document.size or i and ranges[i - 1][1] > a
                    for i, (a, b) in enumerate(ranges)
                ):
                    raise ValueError("stored query ranges are outside their source")
                content = document.text.encode()
                text = b"".join(content[a:b] for a, b in ranges).decode()
                document = RetainedDocument(document, text, ranges)
            elif record.ranges or "retained_ranges" in origin:
                raise ValueError("stored query omitted its retained ranges")
            rows.append(Row(len(rows), document))
    expected = Counter(
        {
            id: len(record.get("occurrences", [0]))
            for id, record in provenance.items()
            if record["selection"]["kind"] == "retained"
        }
    )
    if (
        len(rows) != manifest.occurrences
        or Counter(row.id for row in rows) != expected
        or [row.id for row in rows] != sorted(row.id for row in rows)
        or sum(row.document.size for row in rows) != resource.profile.output_content_bytes
    ):
        raise ValueError("stored query selection has incomplete coverage")
    handle = Query(
        CompletedQuery(
            id=resource.id.hex(),
            inputs=tuple(id.hex() for id in resource.snapshot_ids),
            field_snapshot_ids=tuple(resource.field_snapshot_ids),
            rows=rows,
            provenance=provenance,
            summary=_summary(resource.profile),
        ),
        (),
        code,
    )
    handle._encoding_provider = service._encodings
    return handle


class _Frames:
    """Fetch and verify only the selected source frames when their text is read."""

    def __init__(self, storage: ObjectStore) -> None:
        self.storage = storage

    def _frame(self, frame: Frame) -> bytes:
        digest = bytes(frame["digest"])
        data = self.storage._get("snapshot/objects/" + digest.hex(), frame["bytes"])
        if (
            len(data) != frame["bytes"]
            or blake3(data).digest() != digest
            or text_profile(data) != frame["profile"]
        ):
            raise ValueError("stored selection text frame integrity check failed")
        return data


def _summary(profile: q.QueryProfile) -> QuerySummary:
    def counts(
        value: q.QueryProfile | q.QueryStepProfile, prefix: Literal["input", "output"]
    ) -> Counts:
        if prefix == "input":
            return dict(
                documents=value.input_documents,
                bytes=value.input_content_bytes,
                characters=value.input_characters,
            )
        return dict(
            documents=value.output_documents,
            bytes=value.output_content_bytes,
            characters=value.output_characters,
        )

    result: QuerySummary = QuerySummary(
        input=counts(profile, "input"),
        output=counts(profile, "output"),
        steps=[
            dict(before=counts(step, "input"), after=counts(step, "output"))
            for step in profile.steps
        ],
    )
    if profile.HasField("sampling"):
        value = profile.sampling
        result["sampling"] = dict(
            unit=value.unit,
            requested=value.requested,
            realized=value.realized,
            overshoot=value.overshoot,
            unique_documents=value.unique_documents,
            document_occurrences=value.document_occurrences,
            requested_domains=dict(value.requested_domains),
            realized_domains=dict(value.realized_domains),
        )
    if profile.HasField("decontamination"):
        value = profile.decontamination
        result["decontamination"] = dict(
            matched_documents=value.matched_documents,
            removed_documents=value.removed_documents,
            removed_spans=value.removed_spans,
            removed_bytes=value.removed_bytes,
            reference_snapshot_ids=list(value.reference_snapshot_ids),
        )
    return result
