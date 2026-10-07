"""Publish packing streams into verified token shards and bounded metadata batches."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator, Mapping, Protocol

from premixdb.engine.contracts import Provenance, SourceRange
from premixdb.engine.datasets import Dataset, HuggingFaceTokenizer, Sequence
from premixdb.storage.objects import ObjectStore
from premixdb.storage.ranges import MAX_RANGE_BYTES
from premixdb.training.sequences import INDEX_PAGE_SIZE, MAX_INDEX_PAGE_BYTES, decode_preview
from premixdb.v1 import data_mixture_pb2 as datasets
from premixdb.v1.storage_pb2 import (
    COMPRESSION_UNSPECIFIED,
    COMPRESSION_ZSTANDARD,
    ObjectProfile,
    SpanProfile,
    SpanRef,
)


class _Occurrences(Protocol):
    def occurrence_document(self, ordinal: int) -> str: ...


def _regions(handle: _Occurrences, sequence: Sequence) -> Iterator[datasets.TokenRegion]:
    ranges: dict[int, list[SourceRange]] = {}
    for source in sequence.source_ranges:
        ranges.setdefault(source["occurrence"], []).append(source)
    kinds = {
        "content": datasets.TokenRegion.KIND_CONTENT,
        "separator": datasets.TokenRegion.KIND_SEPARATOR,
        "padding": datasets.TokenRegion.KIND_PADDING,
    }
    for span in sequence.spans:
        region = datasets.TokenRegion(
            start=span["start"], end=span["end"], kind=kinds[span["kind"]]
        )
        if span["kind"] != "padding":
            occurrence = span["occurrence"]
            region.query_ordinal = occurrence
            region.document_id = bytes.fromhex(handle.occurrence_document(occurrence))
            region.document_token_start = span.get("offset", 0)
            if span["kind"] == "content":
                region.source_ranges.extend(
                    datasets.TokenByteRange(
                        token=source["token"],
                        token_end=source["token_end"],
                        start=source["start"],
                        end=source["end"],
                    )
                    for source in ranges.get(occurrence, ())
                    if span["start"] <= source["token"] < span["end"]
                )
        yield region


def publish(
    store: ObjectStore,
    handle: Dataset,
    *,
    profile: datasets.DatasetProfile | None = None,
    lineage: Mapping[str, Provenance] | None = None,
    tokenizer: HuggingFaceTokenizer | None = None,
) -> tuple[list[SpanRef], list[SpanRef], datasets.DatasetPreview]:
    sequence_length = handle.plan.packing.length
    batch = datasets.SequenceBatch()
    batches: list[SpanRef] = []
    token_spans: list[SpanRef] = []
    preview = datasets.DatasetPreview()
    source_tokens, geometry = {}, {}

    def flush() -> None:
        size = batch.ByteSize()
        if size > MAX_INDEX_PAGE_BYTES:
            raise ValueError("oversized decoded sequence index")
        data = batch.SerializeToString(deterministic=True)
        compression = COMPRESSION_UNSPECIFIED
        if len(data) > MAX_RANGE_BYTES:
            import pyarrow as pa

            encoded = pa.compress(data, codec="zstd").to_pybytes()
            assert isinstance(encoded, bytes)
            data = encoded
            compression = COMPRESSION_ZSTANDARD
        if len(data) > MAX_RANGE_BYTES:
            raise ValueError("oversized encoded sequence index")
        ref = store.put("dataset", data)
        batches.append(
            SpanRef(
                object=ref,
                start=0,
                end=len(data),
                blake3_digest=ref.blake3_digest,
                compression=compression,
                profile=SpanProfile(content_bytes=size),
            )
        )
        batch.Clear()

    with TemporaryDirectory(prefix="premixdb-tokens-") as directory:
        for first, token_path, mask_path, digests, chunk in handle.write_token_shards(
            Path(directory)
        ):
            count = len(digests) * sequence_length
            token_object = store.put(
                "dataset",
                token_path.read_bytes(),
                profile=ObjectProfile(content_bytes=count * 4, spans=len(digests), tokens=count),
            )
            mask_object = store.put(
                "dataset",
                mask_path.read_bytes(),
                profile=ObjectProfile(content_bytes=count, spans=len(digests)),
            )
            token_spans.append(
                SpanRef(
                    object=token_object,
                    start=0,
                    end=count * 4,
                    blake3_digest=token_object.blake3_digest,
                    profile=SpanProfile(content_bytes=count * 4, tokens=count),
                )
            )
            for offset, ((token_digest, mask_digest), native) in enumerate(zip(digests, chunk)):
                start = offset * sequence_length
                mask = SpanRef(
                    object=mask_object,
                    start=start,
                    end=start + sequence_length,
                    blake3_digest=bytes.fromhex(mask_digest),
                    profile=SpanProfile(content_bytes=sequence_length),
                )
                seq = batch.sequences.add(
                    ordinal=first + offset,
                    attention_mask=mask,
                    loss_mask=mask,
                    tokens=SpanRef(
                        object=token_object,
                        start=start * 4,
                        end=(start + sequence_length) * 4,
                        blake3_digest=bytes.fromhex(token_digest),
                        profile=SpanProfile(
                            content_bytes=sequence_length * 4, tokens=sequence_length
                        ),
                    ),
                )
                seq.regions.extend(_regions(handle, native))
                if profile is not None:
                    documents = {r.document_id for r in seq.regions if r.document_id}
                    geometry[len(documents)] = geometry.get(len(documents), 0) + 1
                    for region in seq.regions:
                        if region.kind == datasets.TokenRegion.KIND_CONTENT and lineage is not None:
                            corpus = lineage[region.document_id.hex()]["corpus_id"]
                            source_tokens[corpus] = (
                                source_tokens.get(corpus, 0) + region.end - region.start
                            )
                if native.ordinal < 10:
                    tokens_preview, mask_preview = native._preview(256)
                    example = preview.sequences.add(
                        ordinal=native.ordinal,
                        tokens=tokens_preview,
                        attention_mask=mask_preview,
                        loss_mask=mask_preview,
                        truncated=sequence_length > 256,
                    )
                    for region in seq.regions:
                        if region.start >= len(example.tokens):
                            continue
                        clipped = example.regions.add()
                        clipped.CopyFrom(region)
                        clipped.end = min(clipped.end, len(example.tokens))
                        clipped.ClearField("source_ranges")
                        for alignment in region.source_ranges:
                            if alignment.token >= clipped.end:
                                continue
                            retained = clipped.source_ranges.add()
                            retained.CopyFrom(alignment)
                            if retained.token_end > clipped.end:
                                retained.end -= retained.token_end - clipped.end
                                retained.token_end = clipped.end
                    example.text = decode_preview(
                        example.tokens,
                        example.regions,
                        tokenizer.decode if tokenizer is not None else None,
                    )
                if len(batch.sequences) == INDEX_PAGE_SIZE:
                    flush()
        if batch.sequences:
            flush()
    if profile is not None:
        if (
            dict(profile.documents_per_sequence) != geometry
            or any(
                source_tokens.get(key, 0) != count for key, count in profile.source_tokens.items()
            )
            or any(
                profile.source_tokens.get(key, 0) != count for key, count in source_tokens.items()
            )
        ):
            raise ValueError("published token geometry does not match planned profile")
    return token_spans, batches, preview
