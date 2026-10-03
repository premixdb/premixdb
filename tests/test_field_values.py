"""Field outcomes preserve their wire types and reject malformed stored evidence."""

from __future__ import annotations

import math
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from premixdb._protobuf import copy_message
from premixdb._typing import FieldValue
from premixdb.execution.enrichment import decode_value, encode_value
from premixdb.internal import derivation_pb2 as e
from premixdb.v1 import field_pb2 as f

DOCUMENT_ID = b"x" * 32
UNKNOWN_FIELD = b"\xf8\x07\x01"


@pytest.mark.parametrize(
    "element_type,width,value,outcome",
    [
        (f.VALUE_FLOAT32, 0, True, e.FieldValue(number=1)),
        (f.VALUE_FLOAT32, 0, -0.0, e.FieldValue(number=-0.0)),
        (f.VALUE_FLOAT64, 0, 1e300, e.FieldValue(number=1e300)),
        (f.VALUE_FLOAT64, 0, 2**63 - 1, e.FieldValue(number=float(2**63 - 1))),
        (f.VALUE_INT64, 0, -(2**63), e.FieldValue(integer=-(2**63))),
        (f.VALUE_INT64, 0, 2**63 - 1, e.FieldValue(integer=2**63 - 1)),
        (f.VALUE_STRING, 0, "é🌍", e.FieldValue(text="é🌍")),
        (f.VALUE_STRING, 0, "", e.FieldValue(text="")),
        (f.VALUE_BOOL, 0, False, e.FieldValue(boolean=False)),
        (
            f.VALUE_FLOAT32,
            3,
            [True, 2, -0.0],
            e.FieldValue(vector=e.NumericVector(values=[1, 2, -0.0])),
        ),
        (f.VALUE_TYPE_UNSPECIFIED, 0, None, e.FieldValue(null=True)),
        (f.VALUE_FLOAT32, 1024, None, e.FieldValue(null=True)),
    ],
)
def test_field_outcomes_keep_exact_wire_representations(
    element_type: f.ValueType, width: int, value: FieldValue, outcome: e.FieldValue
) -> None:
    spec = f.Field(element_type=element_type, length=width)
    expected = copy_message(outcome)
    expected.document_id = DOCUMENT_ID
    row = encode_value(spec, DOCUMENT_ID, value)
    assert row.SerializeToString(deterministic=True) == expected.SerializeToString(
        deterministic=True
    )
    result = decode_value(spec, row)
    if row.WhichOneof("value") == "number":
        assert type(result) is float
        assert result == row.number
        assert math.copysign(1, result) == math.copysign(1, row.number)
    else:
        assert result == value
    if isinstance(result, list):
        result.append(4.0)
        assert row == expected


@pytest.mark.parametrize(
    "element_type,width,outcome",
    [
        (f.VALUE_FLOAT64, 0, e.FieldValue(integer=1)),
        (f.VALUE_FLOAT64, 0, e.FieldValue(boolean=True)),
        (f.VALUE_INT64, 0, e.FieldValue(number=1)),
        (f.VALUE_INT64, 0, e.FieldValue(boolean=True)),
        (f.VALUE_BOOL, 0, e.FieldValue(integer=1)),
        (f.VALUE_STRING, 0, e.FieldValue(boolean=False)),
        (f.VALUE_TYPE_UNSPECIFIED, 0, e.FieldValue(number=1)),
        (f.VALUE_FLOAT32, 0, e.FieldValue(number=math.nan)),
        (f.VALUE_FLOAT64, 0, e.FieldValue(number=math.inf)),
        (f.VALUE_FLOAT64, 0, e.FieldValue(number=-math.inf)),
        (f.VALUE_FLOAT32, 0, e.FieldValue(vector=e.NumericVector())),
        (f.VALUE_FLOAT32, 2, e.FieldValue(number=1)),
        (f.VALUE_FLOAT32, 2, e.FieldValue(vector=e.NumericVector(values=[1]))),
        (f.VALUE_FLOAT32, 2, e.FieldValue(vector=e.NumericVector(values=[1, math.nan]))),
        (f.VALUE_FLOAT32, 2, e.FieldValue(vector=e.NumericVector(values=[1, math.inf]))),
        (f.VALUE_BOOL, 0, e.FieldValue(null=False)),
        (f.VALUE_BOOL, 0, e.FieldValue()),
    ],
)
def test_malformed_field_outcomes_are_rejected(
    element_type: f.ValueType, width: int, outcome: e.FieldValue
) -> None:
    with pytest.raises(ValueError):
        decode_value(f.Field(element_type=element_type, length=width), outcome)


@pytest.mark.parametrize(
    "outcome,nested",
    [
        (e.FieldValue(number=1), False),
        (e.FieldValue(null=True), False),
        (e.FieldValue(vector=e.NumericVector(values=[1, 2])), False),
        (e.FieldValue(vector=e.NumericVector(values=[1, 2])), True),
    ],
)
def test_unknown_fields_are_rejected_on_values_and_nulls(
    outcome: e.FieldValue, nested: bool
) -> None:
    row = copy_message(outcome)
    width = 2 if row.HasField("vector") else 0
    (row.vector if nested else row).MergeFromString(UNKNOWN_FIELD)
    with pytest.raises(ValueError, match="stored value"):
        decode_value(f.Field(element_type=f.VALUE_FLOAT64, length=width), row)


@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=16))
def test_finite_vectors_round_trip_without_changing_values(values: list[float]) -> None:
    spec = f.Field(element_type=f.VALUE_FLOAT64, length=len(values))
    row = encode_value(spec, DOCUMENT_ID, cast(FieldValue, values))
    result = decode_value(spec, row)
    assert isinstance(result, list)
    assert result == values
    for actual, expected in zip(result, values, strict=True):
        assert type(actual) is float
        assert math.copysign(1, actual) == math.copysign(1, expected)


@pytest.mark.parametrize("value", [[1], [1, math.inf], ["x", 1], "x"])
def test_invalid_computed_vectors_are_rejected(value: FieldValue) -> None:
    with pytest.raises(ValueError):
        encode_value(f.Field(element_type=f.VALUE_FLOAT32, length=2), DOCUMENT_ID, value)
