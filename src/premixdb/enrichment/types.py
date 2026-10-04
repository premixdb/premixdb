"""Inputs and validation shared by worker adapters."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from importlib.metadata import version
from typing import Sequence

from blake3 import blake3

from premixdb.contracts import FieldValue
from premixdb.v1 import field_pb2 as fields


@dataclass(frozen=True)
class Document:
    """An immutable source occurrence; IDs must be unique across all partitions."""

    id: str
    text: str
    url: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("document id must be a nonempty string")
        if not isinstance(self.text, str):
            raise TypeError("document text must be a string")
        if self.url is not None and not isinstance(self.url, str):
            raise TypeError("document url must be a string or None")


@dataclass(frozen=True)
class ModelPin:
    repository: str
    revision: str

    def __post_init__(self) -> None:
        if not self.repository or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("model pin requires a repository and full lowercase Hub commit SHA")


def check_documents(documents: Sequence[Document]) -> None:
    if len({doc.id for doc in documents}) != len(documents):
        raise ValueError("duplicate document IDs in batch")


def positive(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def matrix(values: object, rows: int, width: int) -> list[list[float]]:
    """Validate the untyped tensor/list boundary before admitting model values."""
    if not isinstance(values, (list, tuple)):
        raise ValueError("model output must be a matrix")
    result: list[list[float]] = []
    for row in values:
        if not isinstance(row, (list, tuple)) or any(
            not isinstance(value, (int, float)) for value in row
        ):
            raise ValueError("model output must contain numeric rows")
        result.append([float(value) for value in row])
    if len(result) != rows or any(len(row) != width for row in result):
        raise ValueError("model output shape does not match documents and field width")
    if any(not math.isfinite(x) for row in result for x in row):
        raise ValueError("model returned non-finite values")
    return result


type ComputedRow = dict[str, FieldValue]


def package_versions(*packages: str) -> dict[str, str]:
    return {name: version(name) for name in packages}


def field(
    name: str,
    *,
    width: int = 0,
    classes: Sequence[str] = (),
    element_type: fields.ValueType = fields.VALUE_FLOAT32,
) -> fields.Field:
    result = fields.Field(name=name, version=1, element_type=element_type, length=width)
    if classes:
        if len(classes) != width:
            raise ValueError("classes must match vector width")
        result.classification.classes.extend(classes)
        result.classification.transform = fields.PROBABILITY_TRANSFORM_SOFTMAX
    result.id = blake3(result.SerializeToString(deterministic=True)).digest()
    return result
