"""Typed intrinsic and derived field expressions; no inference or execution runtime imports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from ._field_ids import field_id
from .v1 import query_pb2 as q


@runtime_checkable
class FieldProjection(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def projection(self) -> q.FieldComparison.Projection: ...
    @property
    def class_name(self) -> str: ...
    @property
    def component_index(self) -> int | None: ...


@dataclass(frozen=True)
class FieldPredicate:
    field: "FieldProjection"
    operator: q.Comparison.Operator
    value: str | int | float | bool | Enum

    def __bool__(self) -> bool:
        raise TypeError("use separate where() steps instead of chained comparisons or and/or")

    def _operation(self) -> q.Operation:
        selector = self.field
        identity = field_id(selector.name)
        if identity <= q.FIELD_SOURCE_CORPUS_ID and selector.projection == q.FieldComparison.SCALAR:
            if identity == q.FIELD_SOURCE_CORPUS_ID:
                raise NotImplementedError(
                    "source.corpus_id is currently available for mix strata only"
                )
            comparison = q.Comparison(field=identity, operator=self.operator)
            if identity == q.FIELD_OBJECT_URI:
                if not isinstance(self.value, str):
                    raise TypeError("object.uri comparisons require a string")
                comparison.text = self.value
            else:
                if type(self.value) is not int or not 0 <= self.value < 2**64:
                    raise ValueError("count must be an unsigned 64-bit integer")
                comparison.count = self.value
            return q.Operation(where=comparison)
        result = q.FieldComparison(
            field=field_id(selector.name),
            projection=selector.projection,
            class_name=selector.class_name,
            operator=self.operator,
        )
        if selector.component_index is not None:
            result.component = selector.component_index
        value = self.value.value if isinstance(self.value, Enum) else self.value
        if type(value) is bool:
            result.boolean = value
        elif type(value) is int:
            if not -(2**63) <= value < 2**63:
                raise ValueError("integer field comparisons require int64")
            result.integer = value
        elif type(value) is float:
            if not math.isfinite(value):
                raise ValueError("field comparisons must be finite")
            result.number = value
        elif isinstance(value, str):
            result.text = value
        else:
            raise TypeError("unsupported field comparison value")
        return q.Operation(field_where=result)


@dataclass(frozen=True, eq=False)
class ScalarField[T: str | int | float | bool | Enum]:
    name: str
    value_type: type[T]
    projection: q.FieldComparison.Projection = q.FieldComparison.SCALAR
    class_name: str = ""
    component_index: int | None = None

    def _compare(self, operator: q.Comparison.Operator, value: T) -> FieldPredicate:
        valid = (
            type(value) in (float, int)
            if self.value_type is float
            else isinstance(value, self.value_type)
        )
        if not valid or (isinstance(value, bool) and self.value_type is not bool):
            raise TypeError(f"{self.name} expects {self.value_type.__name__}")
        return FieldPredicate(self, operator, value)

    def __eq__(self, value: T) -> FieldPredicate:  # ty: ignore[invalid-method-override]
        return self._compare(q.Comparison.OPERATOR_EQ, value)

    def __ne__(self, value: T) -> FieldPredicate:  # ty: ignore[invalid-method-override]
        return self._compare(q.Comparison.OPERATOR_NE, value)

    def __lt__(self, value: T) -> FieldPredicate:
        return self._compare(q.Comparison.OPERATOR_LT, value)

    def __le__(self, value: T) -> FieldPredicate:
        return self._compare(q.Comparison.OPERATOR_LE, value)

    def __gt__(self, value: T) -> FieldPredicate:
        return self._compare(q.Comparison.OPERATOR_GT, value)

    def __ge__(self, value: T) -> FieldPredicate:
        return self._compare(q.Comparison.OPERATOR_GE, value)

    def asc(self) -> q.OrderBy:
        """Order documents by this field from smallest to largest."""
        return self._order(q.OrderBy.DIRECTION_ASC)

    def desc(self) -> q.OrderBy:
        """Order documents by this field from largest to smallest."""
        return self._order(q.OrderBy.DIRECTION_DESC)

    def _order(self, direction: q.OrderBy.Direction) -> q.OrderBy:
        from ._curation import selector

        identity = field_id(self.name)
        if identity <= q.FIELD_SOURCE_CORPUS_ID and self.projection == q.FieldComparison.SCALAR:
            return q.OrderBy(field=identity, direction=direction)
        return q.OrderBy(selector=selector(self), direction=direction)

    def is_null(self) -> FieldPredicate:
        """Select documents whose field value is missing."""
        return FieldPredicate(
            ScalarField(self.name, bool, q.FieldComparison.IS_NULL),
            q.Comparison.OPERATOR_EQ,
            True,
        )


@dataclass(frozen=True)
class VectorField:
    name: str

    def component(self, index: int) -> ScalarField[float]:
        """Select one zero-based vector component for filtering or ordering."""
        if type(index) is not int or not 0 <= index < 2**32:
            raise ValueError("vector component must be uint32")
        return ScalarField(
            self.name, float, q.FieldComparison.VECTOR_COMPONENT, component_index=index
        )
