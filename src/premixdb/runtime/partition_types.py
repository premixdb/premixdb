"""Typed private worker boundaries and domain checks after protobuf decoding."""

from __future__ import annotations

from typing import Literal

from premixdb.engine.contracts import Span, TokenRange
from premixdb.internal import transport_pb2 as t
from premixdb.schemas.binary import CODEC_VERSION, known, parse


def request(data: bytes, kind: str) -> t.WorkerRequest:
    result = parse(data, t.WorkerRequest())
    if result.version != CODEC_VERSION or result.WhichOneof("request") != kind:
        raise ValueError("unsupported partition request")
    if kind == "pack":
        control = result.pack
        known(control)
        if not control.length or control.first + control.sequences > control.total_sequences:
            raise ValueError("invalid packing partition geometry")
        if any(a > b for a, b in zip(control.prefixes, control.prefixes[1:])):
            raise ValueError("invalid packing prefixes")
    elif kind == "features":
        known(result.features)
        if not result.features.HasField("producer") or not result.features.definition:
            raise ValueError("missing partition producer policy")
        for row in result.features.rows:
            known(row)
            if not row.id:
                raise ValueError("missing partition document identity")
    else:
        rows = result.tokenize.rows if kind == "tokenize" else result.evidence.rows
        known(result.tokenize if kind == "tokenize" else result.evidence)
        if kind == "tokenize" and not result.tokenize.HasField("tokenizer"):
            raise ValueError("missing tokenizer policy")
        for row in rows:
            known(row)
            if not row.id:
                raise ValueError("missing partition document identity")
            for interval in row.ranges:
                known(interval)
                if interval.start > interval.end:
                    raise ValueError("invalid retained byte interval")
            if sum(r.end - r.start for r in row.ranges) != len(row.text.encode()):
                raise ValueError("retained byte coverage differs from text")
    return result


def result(data: bytes, kind: str) -> t.WorkerResult:
    value = parse(data, t.WorkerResult())
    if value.version != CODEC_VERSION or value.WhichOneof("result") != kind:
        raise ValueError("unsupported partition output")
    return value


def spans(row: t.PackedRow) -> list[Span]:
    result: list[Span] = []
    cursor = 0
    for span in row.spans:
        known(span)
        kind: Literal["content", "separator", "padding"]
        if span.kind == "content":
            kind = "content"
        elif span.kind == "separator":
            kind = "separator"
        elif span.kind == "padding":
            kind = "padding"
        else:
            raise ValueError("invalid packing span kind")
        if not cursor == span.start < span.end <= len(row.tokens):
            raise ValueError("invalid packing span coverage")
        item = Span(start=span.start, end=span.end, kind=kind)
        if span.HasField("occurrence"):
            item["occurrence"] = span.occurrence
        if span.HasField("offset"):
            item["offset"] = span.offset
        if kind == "content" and ("occurrence" not in item or "offset" not in item):
            raise ValueError("missing content span provenance")
        if kind == "separator" and ("occurrence" not in item or "offset" in item):
            raise ValueError("invalid separator span provenance")
        if kind == "padding" and ("occurrence" in item or "offset" in item):
            raise ValueError("invalid padding span provenance")
        result.append(item)
        cursor = span.end
    if cursor != len(row.tokens):
        raise ValueError("incomplete packing span coverage")
    return result


def alignment(row: t.PackedRow) -> list[TokenRange]:
    result = []
    span_index = 0
    previous = 0
    for interval in row.alignment:
        known(interval)
        if (
            interval.token < previous
            or interval.token >= len(row.tokens)
            or interval.start > interval.end
        ):
            raise ValueError("invalid packed token interval")
        while span_index < len(row.spans) and row.spans[span_index].end <= interval.token:
            span_index += 1
        if span_index == len(row.spans):
            raise ValueError("packed interval has no source span")
        span = row.spans[span_index]
        if span.kind != "content" or interval.occurrence != span.occurrence:
            raise ValueError("packed interval provenance differs")
        result.append(
            TokenRange(
                token=interval.token,
                occurrence=interval.occurrence,
                start=interval.start,
                end=interval.end,
            )
        )
        previous = interval.token
    return result
