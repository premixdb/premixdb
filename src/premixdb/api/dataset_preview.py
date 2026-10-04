"""Bounded packed-sequence previews, reusing inline examples when available."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from premixdb.contracts import PreviewSequence
from premixdb.schemas import requests as _requests
from premixdb.schemas.ids import _encode_id
from premixdb.training.sequences import decode_preview, preview_decoder
from premixdb.v1 import data_mixture_pb2 as d
from premixdb.v1 import status_pb2 as d_status

if TYPE_CHECKING:
    from premixdb.api import Dataset


def preview(
    dataset: Dataset, *, limit: int, offset: int, max_characters: int
) -> list[PreviewSequence]:
    limit, offset, width = _requests._preview_options(
        limit, offset, max_characters, unit="sequences"
    )
    if not limit:
        return []
    dataset._db._require_open()
    ready = dataset._db._get("Dataset", dataset._resource.id)
    if ready.status == d_status.STATUS_ERROR:
        from premixdb.contracts import ExecutionError

        raise ExecutionError(ready.error)
    if ready.status != d_status.STATUS_COMPLETED:
        if dataset._db._read_only:
            from premixdb.contracts import ExecutionError

            raise ExecutionError("dataset is not complete; preview it in a writable session first")
        from premixdb.runtime import Coordinator

        assert isinstance(dataset._db._executor, Coordinator)
        return dataset._db._executor._preview_dataset(
            ready, limit=limit, offset=offset, max_characters=width
        )
    dataset._resource = ready
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
