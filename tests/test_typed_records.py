"""Compiled record checks retain strict types, optional keys and recursive shapes."""

from __future__ import annotations

from copy import deepcopy
from typing import Literal, NotRequired, Required, TypedDict

import pytest

from premixdb.contracts import JSON, checked_record


class OptionalFields(TypedDict, total=False):
    name: Required[str]
    mode: Literal["one", "two"]
    tag: Literal[1]
    weights: dict[str, float]
    point: tuple[int, float]
    aliases: tuple[str, ...]
    payload: JSON


class Record(OptionalFields):
    ids: list[int]
    enabled: NotRequired[bool]


class Node(TypedDict):
    name: str
    children: NotRequired[list[Node]]


VALID: dict[str, object] = {
    "name": "root",
    "ids": [1, 2],
    "mode": "one",
    "tag": 1,
    "weights": {"a": 1, "b": 2.5},
    "point": (2, 3),
    "aliases": ("a", "b"),
    "payload": {"nested": [None, True, 2.5, {"x": []}]},
}


def test_record_validation_keeps_values_and_resolves_required_optional_inherited_fields() -> None:
    for value in (VALID, {"name": "minimal", "ids": []}, {**VALID, "enabled": False}):
        result = checked_record(value, Record)
        assert result is value
        assert result["ids"] == value["ids"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", 1),
        ("ids", [True]),
        ("ids", [1.0]),
        ("enabled", 1),
        ("mode", "three"),
        ("tag", True),
        ("weights", {"a": True}),
        ("weights", {1: 2}),
        ("point", (1, True)),
        ("point", (1, 2, 3)),
        ("point", [1, 2]),
        ("aliases", ("a", 2)),
        ("payload", {1: "x"}),
        ("payload", {"nested": [object()]}),
        ("extra", None),
    ],
)
def test_record_validation_rejects_wrong_types_and_unknown_keys(field: str, value: object) -> None:
    changed = deepcopy(VALID)
    changed[field] = value
    with pytest.raises(ValueError, match="invalid Record record"):
        checked_record(changed, Record)


@pytest.mark.parametrize("missing", ["name", "ids"])
def test_required_keys_remain_required(missing: str) -> None:
    changed = deepcopy(VALID)
    del changed[missing]
    with pytest.raises(ValueError):
        checked_record(changed, Record)


def test_recursive_record_schemas_validate_every_child() -> None:
    value: Node = {"name": "root", "children": [{"name": "child", "children": [{"name": "leaf"}]}]}
    assert checked_record(value, Node) is value
    with pytest.raises(ValueError):
        checked_record({"name": "root", "children": [{"name": 1}]}, Node)
