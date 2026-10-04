"""Validate query policies and compile identities without loading the execution engine."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import TYPE_CHECKING, Callable, Iterable, Literal

from premixdb.contracts import Edge, Orders
from premixdb.engine.identity import Canonical, CodeVersion, digest, unsigned
from premixdb.v1 import query_pb2 as q

if TYPE_CHECKING:
    from premixdb.engine.curation import SelectedDocument
    from premixdb.engine.datasets import HuggingFaceTokenizer
    from premixdb.engine.snapshots import Document

type PolicyPayload = (
    tuple[Literal["decontaminate"], q.Decontaminate, list[Document] | tuple[()]]
    | tuple[Literal["sample"], q.QuerySampling, HuggingFaceTokenizer | None]
    | tuple[
        Literal["similarity"],
        Orders,
        Iterable[Edge] | Callable[[list[SelectedDocument]], Iterable[Edge]] | None,
    ]
)


_FIELDS = {
    "bytes": ("Bytes", "text.utf8_bytes/u64/v1"),
    "characters": ("Characters", "text.unicode_scalars/u64/v1"),
    "object_uri": ("ObjectUri", "object.uri/utf8/v1"),
}
_OPERATORS = dict(
    eq="Equal", ne="NotEqual", lt="Less", le="LessOrEqual", gt="Greater", ge="GreaterOrEqual"
)


def field_definition(name: str) -> str:
    if name.startswith("external:"):
        return Canonical("query-field").string(name).finish().hex()
    if name not in _FIELDS:
        raise ValueError(f"unknown field: {name}")
    return Canonical("query-field").string(_FIELDS[name][1]).finish().hex()


@dataclass(frozen=True)
class Step:
    """Immutable validated policy plus its kernel metadata representation."""

    kind: str
    encoding: bytes
    field: str = ""
    comparison: str = ""
    value: int | str = 0
    orders: tuple[tuple[str, bool], ...] = ()
    unit: str = "Document"
    separator: str | None = None
    definition: bytes = b""
    members: frozenset[str] = frozenset()
    payload: PolicyPayload | None = dataclass_field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        # Constructors below own validation and canonical encoding. Reconstructing
        # at admission also rejects forged or mutated policies on empty inputs.
        if self.kind not in (
            "Filter",
            "DedupeExact",
            "Dedupe",
            "FilterIds",
            "FilterDocuments",
            "DedupeIndexed",
            "Policy",
        ):
            raise ValueError("unknown query step")


def filter(name: str, comparison: str, value: int | str) -> Step:
    definition = digest(field_definition(name))
    if comparison not in _OPERATORS:
        raise ValueError("unknown comparison")
    if name == "object_uri":
        if not isinstance(value, str):
            raise TypeError("URI filters require a string")
        encoded = _string(value)
    else:
        encoded = unsigned(value).to_bytes(8, "big")
    return Step(
        "Filter",
        _string("filter/v1") + definition + _string(comparison) + encoded,
        field=name,
        comparison=comparison,
        value=value,
    )


def _string(value: str) -> bytes:
    data = value.encode("utf-8")
    return len(data).to_bytes(8, "big") + data


def dedupe(
    order_by: Iterable[tuple[str, bool]], comparison: str = "document", separator: str | None = None
) -> Step:
    if comparison not in ("document", "line"):
        raise ValueError("unknown comparison unit")
    if separator is not None and (
        not isinstance(separator, str)
        or len(separator) != 1
        or separator == "\0"
        or 0xD800 <= ord(separator) <= 0xDFFF
    ):
        raise ValueError("group separator must be one non-NUL Unicode scalar")
    orders = list(order_by)
    encoded = _string("dedupe-exact-text/v1") + len(orders).to_bytes(8, "big")
    for name, descending in orders:
        if type(descending) is not bool:
            raise TypeError("descending must be a bool")
        encoded += digest(field_definition(name)) + _string("desc" if descending else "asc")
    if comparison == "document" and separator is None:
        kind = "DedupeExact"
    else:
        prefix = _string("dedupe-units/simultaneous/v1")
        prefix += _string("document" if comparison == "document" else "nonempty-lf-line")
        prefix += (
            _string("document")
            if separator is None
            else _string("corpus-source-prefix") + ord(separator).to_bytes(8, "big")
        )
        encoded = prefix + encoded
        kind = "Dedupe"
    return Step(
        kind,
        encoded,
        orders=tuple(orders),
        unit="Document" if comparison == "document" else "Line",
        separator=separator,
    )


def external_filter(definition: bytes, members: Iterable[str | bytes] = ()) -> Step:
    if not isinstance(definition, bytes) or len(definition) != 32:
        raise ValueError("external definition must be a 32-byte digest")
    members = frozenset(digest(v.hex() if isinstance(v, bytes) else v).hex() for v in members)
    return Step(
        "FilterIds",
        _string("external-filter/v1") + definition,
        definition=definition,
        members=members,
    )


def filter_documents(members: Iterable[str | bytes] = ()) -> Step:
    members = frozenset(digest(v.hex() if isinstance(v, bytes) else v).hex() for v in members)
    encoded = Canonical("document-selection/v1").u64(len(members))
    for member in sorted(members):
        encoded.fixed(digest(member))
    return Step("FilterDocuments", encoded.finish(), members=members)


def external_dedupe(definition: bytes, orders: Iterable[tuple[str, bool]]) -> Step:
    if not isinstance(definition, bytes) or len(definition) != 32:
        raise ValueError("external definition must be a 32-byte digest")
    policy = dedupe(orders)
    return Step(
        "DedupeIndexed",
        _string("external-dedupe/v1") + definition,
        definition=definition,
        orders=policy.orders,
    )


def validate(step: Step) -> Step:
    if not isinstance(step, Step):
        raise TypeError("query operations must be Step values")
    if step.kind == "Filter":
        expected = filter(step.field, step.comparison, step.value)
    elif step.kind == "FilterIds":
        expected = external_filter(step.definition, step.members)
    elif step.kind == "FilterDocuments":
        expected = filter_documents(step.members)
    elif step.kind == "Policy":
        expected = policy(step.definition, step.payload)
    elif step.kind == "DedupeIndexed":
        expected = external_dedupe(step.definition, step.orders)
    else:
        if step.unit not in ("Document", "Line"):
            raise ValueError("unknown comparison unit")
        expected = dedupe(
            step.orders, "document" if step.unit == "Document" else "line", step.separator
        )
    if expected != step:
        raise ValueError("query policy does not match its canonical encoding")
    return step


def query_identity(
    snapshots: Iterable[str],
    steps: tuple[Step, ...] | list[Step],
    code: CodeVersion,
    fields: tuple[bytes, ...] | list[bytes] = (),
) -> str:
    inputs = sorted({digest(value) for value in snapshots})
    hash_ = Canonical("query").u64(len(inputs))
    for value in inputs:
        hash_.fixed(value)
    hash_.fixed(code.canonical_digest()).u64(len(steps))
    for step in steps:
        hash_.fixed(step.encoding)
    if fields:
        hash_.string("query-fields/v1").u64(len(fields))
        for value in fields:
            hash_.fixed(value)
    return hash_.finish().hex()


def policy(definition: bytes, payload: PolicyPayload | None = None) -> Step:
    if not isinstance(definition, bytes) or len(definition) != 32:
        raise ValueError("policy definition must be a 32-byte digest")
    return Step("Policy", _string("policy/v1") + definition, definition=definition, payload=payload)
