"""Typed adapters for protobuf's dynamically generated runtime methods."""

from __future__ import annotations

from typing import Iterable, cast

from google.protobuf.descriptor import Descriptor
from google.protobuf.message import Message


def copy_message[T: Message](message: T) -> T:
    """Detach a message without a serialization round trip, retaining its type."""
    result = type(message)()
    result.CopyFrom(message)
    return result


def parse[T: Message](kind: type[T], data: bytes) -> T:
    value = kind()
    value.ParseFromString(data)
    return value


def descriptor_name(value: Message) -> str:
    descriptor = value.DESCRIPTOR
    if descriptor is None:
        raise TypeError("protobuf message has no descriptor")
    name: object = descriptor.full_name
    if not isinstance(name, str):
        raise TypeError("protobuf descriptor name must be a string")
    return name


def at[T](values: Iterable[T], index: int) -> T:
    # Protobuf's container indexing stubs erase the message element type.
    for ordinal, value in enumerate(values):
        if ordinal == index:
            return value
    raise IndexError(index)


def descriptor(value: Message | type[Message]) -> Descriptor:
    result = value.DESCRIPTOR
    if result is None:
        raise TypeError("protobuf message has no descriptor")
    return cast(Descriptor, result)
