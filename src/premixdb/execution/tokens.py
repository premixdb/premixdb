"""Publish packing streams into verified token shards and bounded metadata batches."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Mapping, Sequence

from ..engine.contracts import Provenance
from ..engine.datasets import Dataset, HuggingFaceTokenizer
from ..v1 import dataset_pb2 as datasets
from ..v1.storage_pb2 import ObjectProfile, SpanProfile, SpanRef
from .storage import ObjectStore


def decode_preview(
    tokens: Sequence[int],
    regions: Iterable[datasets.TokenRegion],
    tokenizer: HuggingFaceTokenizer | None = None,
) -> str:
    if tokenizer is not None:
        return tokenizer.decode(list(tokens))
    text = []
    for region in regions:
        values = tokens[region.start : region.end]
        if not values:
            continue
        if region.kind == datasets.TokenRegion.KIND_CONTENT:
            text.append(bytes(values).decode("utf-8", errors="replace"))
        else:
            label = "separator" if region.kind == datasets.TokenRegion.KIND_SEPARATOR else "padding"
            text.append(f"<{label}:{values[0]} ×{len(values)}>")
    return "".join(text)


def publish(
    store: ObjectStore,
    handle: Dataset,
    sequence_length: int,
    *,
    profile: datasets.DatasetProfile | None = None,
    lineage: Mapping[str, Provenance] | None = None,
    tokenizer: HuggingFaceTokenizer | None = None,
) -> tuple[list[SpanRef], list[SpanRef], datasets.DatasetPreview]:
    batch = datasets.SequenceBatch()
    batches: list[SpanRef] = []
    token_spans: list[SpanRef] = []
    preview = datasets.DatasetPreview()
    source_tokens, geometry = {}, {}

    def flush() -> None:
        data = batch.SerializeToString(deterministic=True)
        ref = store.put("dataset", data)
        batches.append(
            SpanRef(
                object=ref,
                start=0,
                end=len(data),
                blake3_digest=ref.blake3_digest,
                profile=SpanProfile(content_bytes=len(data)),
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
                for span in native.spans:
                    kind = {
                        "content": datasets.TokenRegion.KIND_CONTENT,
                        "separator": datasets.TokenRegion.KIND_SEPARATOR,
                        "padding": datasets.TokenRegion.KIND_PADDING,
                    }[span["kind"]]
                    region = seq.regions.add(start=span["start"], end=span["end"], kind=kind)
                    if span["kind"] != "padding":
                        region.query_ordinal = span["occurrence"]
                        region.document_id = bytes.fromhex(
                            handle.occurrence_document(span["occurrence"])
                        )
                        region.document_token_start = span.get("offset", 0)
                        if span["kind"] == "content":
                            region.source_ranges.extend(
                                datasets.TokenByteRange(
                                    token=r["token"],
                                    start=r["start"],
                                    end=r["end"],
                                    token_end=r["token_end"],
                                )
                                for r in native.source_ranges
                                if r["occurrence"] == span["occurrence"]
                                and span["start"] <= r["token"] < span["end"]
                            )
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
                    mask_preview = [int(v) for v in native.mask[:256]]
                    example = preview.sequences.add(
                        ordinal=native.ordinal,
                        tokens=native.tokens[:256],
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
                    example.text = decode_preview(example.tokens, example.regions, tokenizer)
                if len(batch.sequences) == 128:
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
