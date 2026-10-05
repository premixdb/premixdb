"""Private protobuf boundary helpers; durable identity codecs remain independent."""

from __future__ import annotations

import math
from typing import cast

from google.protobuf.message import DecodeError, Message
from google.protobuf.unknown_fields import UnknownFieldSet

from premixdb.contracts import FieldValue
from premixdb.internal import transport_pb2 as t

CODEC_VERSION = 1


def known(message: Message) -> None:
    if UnknownFieldSet(message):
        raise ValueError("unsupported protobuf fields")


def parse[M: Message](data: bytes, message: M) -> M:
    try:
        message.ParseFromString(data)
    except DecodeError as error:
        raise ValueError("malformed protobuf payload") from error
    known(message)
    return message


def encode_value(value: FieldValue) -> t.TransportValue:
    result = t.TransportValue()
    if value is None:
        result.field.null = True
    elif type(value) is bool:
        result.field.boolean = value
    elif isinstance(value, int):
        if -(1 << 63) <= value < 1 << 63:
            result.field.integer = value
        elif 0 <= value < 1 << 64:
            result.unsigned = value
        else:
            result.big_integer = str(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("field requires finite numeric values")
        result.field.number = value
    elif isinstance(value, str):
        result.field.text = value
    elif isinstance(value, list):
        if value and all(type(item) is float for item in value):
            vector = cast(list[float], value)
            if not all(math.isfinite(item) for item in vector):
                raise ValueError("field vector must contain finite numbers")
            result.field.vector.values.extend(vector)
        else:
            result.scalars.SetInParent()
            for item in value:
                if isinstance(item, list):
                    raise ValueError("expected scalar vector elements")
                result.scalars.values.add().CopyFrom(encode_value(item))
    else:
        raise ValueError("expected a scalar or scalar vector")
    return result


def decode_value(value: t.TransportValue) -> FieldValue:
    known(value)
    kind = value.WhichOneof("value")
    if kind == "unsigned":
        return value.unsigned
    if kind == "big_integer":
        try:
            number = int(value.big_integer)
        except ValueError as error:
            raise ValueError("invalid integer value") from error
        if str(number) != value.big_integer:
            raise ValueError("noncanonical integer value")
        return number
    if kind == "scalars":
        known(value.scalars)
        result = []
        for item in value.scalars.values:
            decoded = decode_value(item)
            if isinstance(decoded, list):
                raise ValueError("expected scalar vector elements")
            result.append(decoded)
        return result
    if kind != "field":
        raise ValueError("missing field outcome")
    row = value.field
    known(row)
    if row.document_id:
        raise ValueError("unexpected field document identity")
    field = row.WhichOneof("value")
    if field == "null":
        if not row.null:
            raise ValueError("invalid null marker")
        return None
    if field == "vector":
        known(row.vector)
        vector = list(row.vector.values)
        if not all(math.isfinite(item) for item in vector):
            raise ValueError("field vector must contain finite numbers")
        return cast(FieldValue, vector)
    if field == "number":
        if not math.isfinite(row.number):
            raise ValueError("field requires finite numeric values")
        return row.number
    if field == "integer":
        return row.integer
    if field == "text":
        return row.text
    if field == "boolean":
        return row.boolean
    raise ValueError("missing field outcome")
