"""Shared identity, integer, and preview-window validation."""

from __future__ import annotations

import builtins
from typing import Literal

from premixdb.schemas.ids import _decode_id


def _id(value: bytes | str, size: int) -> bytes:
    if isinstance(value, str):
        value = _decode_id(value)
    if not isinstance(value, bytes) or len(value) != size:
        raise ValueError(f"ID must contain {size} bytes from a resource")
    return value


def _uint(value: builtins.object, bits: int, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or not int(positive) <= value < 2**bits:
        raise ValueError(
            f"{name} must be {'a positive' if positive else 'an unsigned'} {bits}-bit integer"
        )
    return value


def _preview_options(
    limit: int,
    offset: int,
    max_characters: int,
    *,
    unit: Literal["documents", "sequences"],
) -> tuple[int, int, int]:
    limit = _uint(limit, 32, "limit")
    offset = _uint(offset, 64, "offset")
    max_characters = _uint(max_characters, 32, "max_characters")
    if limit > 1000 or max_characters > 1_000_000:
        raise ValueError(f"preview supports at most 1000 {unit} and 1,000,000 characters")
    return limit, offset, max_characters
