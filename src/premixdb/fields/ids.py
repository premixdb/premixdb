"""Names and stable IDs for the complete protobuf built-in field catalog."""

from typing import cast

from premixdb.v1 import query_pb2 as q

# Enum names encode the namespace followed by the field's original spelling.
# Keep IDs in the schema; never derive them from list position or Python hashes.
FIELD_IDS = {
    symbol.removeprefix("FIELD_").lower().replace("_", ".", 1): value
    for symbol, value in cast(list[tuple[str, q.IntrinsicField]], q.IntrinsicField.items())
    if value != q.FIELD_UNSPECIFIED
}
FIELD_NAMES = {value: name for name, value in FIELD_IDS.items()}
DERIVED_FIELD_NAMES = frozenset(name for value, name in FIELD_NAMES.items() if value > 4)


def field_id(name: str) -> q.IntrinsicField:
    try:
        return FIELD_IDS[name]
    except KeyError:
        raise ValueError(f"unknown built-in field: {name}") from None


def field_name(value: q.IntrinsicField) -> str:
    try:
        return FIELD_NAMES[value]
    except KeyError:
        raise ValueError(f"unknown built-in field ID: {value}") from None


def selector_field(selector: q.FieldComparison) -> q.IntrinsicField:
    """Accept legacy names, rejecting conflicts instead of silently choosing one."""
    value = selector.field
    if selector.field_name:
        legacy = field_id(selector.field_name)
        if value and value != legacy:
            raise ValueError("field enum and legacy field name disagree")
        value = legacy
    field_name(value)  # Reject unspecified and unknown enum numbers.
    return value
