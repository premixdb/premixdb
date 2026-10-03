"""Bounded packed-sequence previews, reusing inline examples when available."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import _requests
from ._ids import _encode_id
from ._types import PreviewSequence
from .v1 import dataset_pb2 as d
from .v1.storage_pb2 import SpanRef

if TYPE_CHECKING:
    from ._resources import Dataset


def preview(
    dataset: Dataset, *, limit: int, offset: int, max_characters: int
) -> list[PreviewSequence]:
    limit = _requests._uint(limit, 32, "limit")
    offset = _requests._uint(offset, 64, "offset")
    width = _requests._uint(max_characters, 32, "max_characters")
    if limit > 1000 or width > 1_000_000:
        raise ValueError("preview supports at most 1000 sequences and 1,000,000 characters")
    if not limit:
        return []
    ready = dataset.wait()._resource
    end = min(offset + limit, ready.profile.sequences)
    result = []
    page = {}
    decoder = None
    for ordinal in range(offset, end):
        if ordinal < len(ready.preview.sequences):
            example = ready.preview.sequences[ordinal]
            tokens = list(example.tokens)
            mask = [bool(value) for value in example.loss_mask]
            attention = [bool(value) for value in example.attention_mask]
            regions = example.regions
            text = example.text
            truncated = example.truncated
        else:
            if ordinal not in page:
                page = {sequence.ordinal: sequence for sequence in dataset._page(ordinal)}
            sequence = page[ordinal]
            tokens = sequence.tokens[:256]
            mask = sequence.mask[:256]
            attention = sequence.attention_mask[:256]
            regions = sequence.spans
            if width:
                from .execution.tokens import decode_preview

                if decoder is None and ready.tokenizer.HasField("hugging_face"):
                    from tokenizers import Tokenizer

                    asset = ready.tokenizer.hugging_face.asset
                    data = dataset._db._object_reader.read(
                        SpanRef(
                            object=asset, end=asset.size_bytes, blake3_digest=asset.blake3_digest
                        )
                    )
                    decoder = Tokenizer.from_str(data.decode())

                text = (
                    decoder.decode(tokens, skip_special_tokens=False)
                    if decoder is not None
                    else decode_preview(tokens, regions)
                )
            else:
                text = ""
            truncated = ready.sequence_length > len(tokens) or (not width and bool(tokens))
        result.append(
            PreviewSequence(
                ordinal=ordinal,
                text=text[:width],
                tokens=tokens,
                mask=mask,
                attention_mask=attention,
                document_ids=list(
                    dict.fromkeys(
                        _encode_id(region.document_id)
                        for region in regions
                        if region.kind == d.TokenRegion.KIND_CONTENT and region.start < len(tokens)
                    )
                ),
                truncated=truncated or len(text) > width,
            )
        )
    return result
