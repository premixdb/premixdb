"""Bounded execution for exact query and packed-sequence previews."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generator, Iterator, cast

from premixdb.contracts import ExecutionError, PreviewSequence
from premixdb.engine import execution
from premixdb.engine.dataset_plan import PackingPlan
from premixdb.engine.datasets import ByteTokens, TokenList
from premixdb.runtime.planner import compile_query, copy_fields, execution_steps
from premixdb.schemas.ids import _encode_id
from premixdb.storage.catalog import Catalog
from premixdb.v1 import data_mixture_pb2 as datasets
from premixdb.v1 import query_pb2 as queries
from premixdb.v1 import status_pb2 as status

if TYPE_CHECKING:
    from premixdb.runtime.coordinator import Coordinator


def query_rows(
    service: Coordinator, resource: queries.Query
) -> Generator[execution.Row, None, None]:
    """Yield exact final-order rows, streaming local filters when possible."""
    from premixdb.engine.queries import _OPS, Row, _value

    current = Catalog.GetQuery(service, queries.GetQueryRequest(id=resource.id)).query
    if current.status == status.STATUS_COMPLETED:
        yield from saved_rows(service, current)
        return
    if current.status == status.STATUS_ERROR:
        raise ExecutionError(current.error)
    steps = execution_steps(resource)
    local = [step for step in steps if step.kind != "Policy"]
    if (
        resource.HasField("sampling")
        or any(step.kind not in ("Filter", "FilterDocuments") for step in local)
        or any(
            step.kind == "Policy" and (step.payload is None or step.payload[0] != "decontaminate")
            for step in steps
        )
    ):
        # Winner selection, sampling and derived fields can require global evidence.
        yield from service._execute_query(compile_query(resource), publish=False)[1]
        return
    from premixdb.engine import curation
    from premixdb.engine.snapshots import Document

    index = service._query_index(resource.snapshot_ids)

    def selected() -> Iterator[Document]:
        for document in index.documents.values():
            if all(
                document.id in step.members
                if step.kind == "FilterDocuments"
                else _OPS[step.comparison](_value(document, step.field), step.value)
                for step in local
            ):
                yield document

    if not resource.HasField("decontaminate"):
        for ordinal, document in enumerate(selected()):
            yield Row(ordinal, document)
        return
    from premixdb.engine.contracts import Provenance
    from premixdb.engine.spill import references

    policy = resource.decontaminate
    reference = service._query_index(policy.snapshot_ids)
    provider = service.pipeline.references if service.pipeline is not None else references
    with provider(list(reference.documents.values()), policy) as evidence:
        ordinal = 0
        for document in selected():
            provenance: dict[str, Provenance] = {
                document.id: dict(
                    corpus_id=document.corpus_id,
                    source_key=document.source_key,
                    content=document.content.hex(),
                    snapshots=list(index.origins[document.id]),
                    selection=dict(kind="retained", ordinal=ordinal),
                )
            }
            for retained in curation.decontaminate(
                [document], (), policy, provenance, evidence=evidence
            ):
                yield Row(ordinal, retained)
                ordinal += 1


def dataset_preview(
    service: Coordinator,
    resource: datasets.Dataset,
    *,
    limit: int,
    offset: int,
    max_characters: int,
    split: str | None = None,
) -> list[PreviewSequence]:
    """Consume a packing prefix without publishing a completed dataset/profile."""
    from itertools import islice

    from premixdb.engine.datasets import encoded_tokens, pack_sequences
    from premixdb.storage.tokens import _regions
    from premixdb.training.sequences import decode_preview

    recipe = copy_fields(resource, datasets.CreateDatasetRequest())
    tokenizer = service._tokenizer(recipe)
    native_handle: execution.Dataset | None = None
    if recipe.HasField("sampling") and split not in ("validation", "test"):
        if recipe.HasField("splits") and split is None:
            from premixdb.runtime.split_datasets import build

            handle = build(service, recipe)
        else:
            pool, args = service._sampling(recipe)
            handle = pool.dataset(*args, *service._packing(recipe), stream=True)
        stream = cast(Generator[execution.Sequence, None, None], handle.iter_sequences())
        regions_handle = handle
        native_handle = handle
    elif recipe.HasField("splits"):
        from contextlib import closing

        from premixdb.schemas.splits import SPLIT_NAMES, content_split

        query = Catalog.GetQuery(service, queries.GetQueryRequest(id=recipe.query_id)).query
        occurrences: list[execution.Row] = []

        class SplitOccurrences:
            def occurrence_document(self, ordinal: int) -> str:
                return occurrences[ordinal].id

        regions_handle = SplitOccurrences()

        def split_stream() -> Generator[execution.Sequence, None, None]:
            ordinal = 0
            for name in (split,) if split else SPLIT_NAMES:
                with closing(service._preview_rows(query)) as selected_rows:

                    def encoded_split() -> Iterator[tuple[execution.Row, ByteTokens | TokenList]]:
                        for row in selected_rows:
                            if content_split(row.document.content, recipe.splits) != name:
                                continue
                            tokens = (
                                service._encodings(row, tokenizer)
                                if tokenizer
                                else encoded_tokens(row)
                            )
                            occurrences.append(row)
                            yield row, tokens

                    packed = pack_sequences(
                        encoded_split(),
                        PackingPlan(*service._packing(recipe)),
                        ordinal_start=ordinal,
                        occurrence_start=len(occurrences),
                    )
                    with closing(packed):
                        for sequence in packed:
                            yield sequence
                            ordinal += 1

        stream = split_stream()
    else:
        query = Catalog.GetQuery(service, queries.GetQueryRequest(id=recipe.query_id)).query
        rows = service._preview_rows(query)
        occurrences: list[execution.Row] = []

        class Occurrences:
            def occurrence_document(self, ordinal: int) -> str:
                return occurrences[ordinal].id

        regions_handle = Occurrences()

        def encoded() -> Iterator[tuple[execution.Row, ByteTokens | TokenList]]:
            for row in rows:
                tokens = service._encodings(row, tokenizer) if tokenizer else encoded_tokens(row)
                occurrences.append(row)
                yield row, tokens

        stream = pack_sequences(encoded(), PackingPlan(*service._packing(recipe)))
    result = []
    try:
        for sequence in islice(stream, offset, offset + limit):
            tokens, mask = sequence._preview(256)
            regions = list(_regions(regions_handle, sequence))
            text = (
                decode_preview(tokens, regions, tokenizer.decode if tokenizer else None)
                if max_characters
                else ""
            )
            result.append(
                PreviewSequence(
                    ordinal=sequence.ordinal,
                    text=text[:max_characters],
                    tokens=tokens,
                    mask=[bool(value) for value in mask],
                    attention_mask=[bool(value) for value in mask],
                    document_ids=list(
                        dict.fromkeys(
                            _encode_id(region.document_id)
                            for region in regions
                            if region.kind == datasets.TokenRegion.KIND_CONTENT
                            and region.start < len(tokens)
                        )
                    ),
                    truncated=resource.sequence_length > len(tokens)
                    or len(text) > max_characters
                    or (not max_characters and bool(tokens)),
                )
            )
    finally:
        stream.close()
        if native_handle is not None:
            native_handle.close()
        elif not recipe.HasField("splits"):
            rows.close()
    return result


def saved_rows(
    service: Coordinator, resource: queries.Query
) -> Generator[execution.Row, None, None]:
    """Load completed selection rows lazily, including retained text transformations."""
    from premixdb.contracts import json_object, load_json
    from premixdb.engine.curation import RetainedDocument, SelectedDocument
    from premixdb.engine.queries import Row
    from premixdb.engine.records import decode_document, validate_document
    from premixdb.engine.snapshots import StoredDocument
    from premixdb.internal import derivation_pb2 as d
    from premixdb.schemas.protobuf import parse
    from premixdb.storage.selections import _Frames

    if not service._storage.metadata.contains("query", resource.id, suffix=".preview-index"):
        yield from service._query(resource.id)  # Compatibility for older published selections.
        return
    index = service._storage.load("query", resource.id, d.DocumentIndex, suffix=".preview-index")
    if index.resource_id != resource.id or index.documents != resource.profile.output_documents:
        raise ValueError("preview index belongs to a different query")
    cursor = 0
    for chunk in index.shards:
        if chunk.first != cursor or not chunk.count:
            raise ValueError("preview index is incomplete or out of order")
        cursor += chunk.count
    if cursor != index.documents:
        raise ValueError("preview index is incomplete")
    frames = _Frames(service._storage)
    for chunk in index.shards:
        shard = parse(d.SelectionShard, service._storage.read_object("query", chunk.object))
        if len(shard.rows) != chunk.count:
            raise ValueError("preview shard coverage differs")
        for offset, record in enumerate(shard.rows):
            source = decode_document(json_object(load_json(record.source_record)))
            validate_document(source, require_profiles=True)
            if len(record.document_id) != 32 or len(record.corpus_id) != 16:
                raise ValueError("preview row has invalid document or corpus IDs")
            document: SelectedDocument = StoredDocument(
                record.corpus_id.hex(), source["key"], source, frames, record.document_id.hex()
            )
            if record.transformed:
                ranges = tuple((r.start, r.end) for r in record.ranges)
                if any(
                    not 0 <= a < b <= document.size or i and ranges[i - 1][1] > a
                    for i, (a, b) in enumerate(ranges)
                ):
                    raise ValueError("stored query ranges are outside their source")
                content = document.text.encode()
                document = RetainedDocument(
                    document, b"".join(content[a:b] for a, b in ranges).decode(), ranges
                )
            elif record.ranges:
                raise ValueError("stored query omitted its retained ranges")
            yield Row(chunk.first + offset, document)
