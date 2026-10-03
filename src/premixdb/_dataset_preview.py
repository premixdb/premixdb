"""Bounded packed-sequence previews, reusing inline examples when available."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from . import _requests
from ._ids import _encode_id
from ._sequences import decode_preview, preview_decoder
from ._types import PreviewSequence
from .v1 import dataset_pb2 as d

if TYPE_CHECKING:
    from ._resources import Dataset


def preview(
    dataset: Dataset, *, limit: int, offset: int, max_characters: int
) -> list[PreviewSequence]:
    limit, offset, width = _requests._preview_options(
        limit, offset, max_characters, unit="sequences"
    )
    if not limit:
        return []
    ready = dataset.wait()._resource
    end = min(offset + limit, ready.profile.sequences)
    result = []
    page = {}
    decoder: Callable[[list[int]], str] | None = None
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
            tokens, mask, attention = sequence._preview(256)
            regions = sequence.spans
            if width:
                if decoder is None and ready.tokenizer.HasField("hugging_face"):
                    decoder = preview_decoder(ready.tokenizer, dataset._db._object_reader)

                text = decode_preview(tokens, regions, decoder)
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
