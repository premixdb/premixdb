"""Lossless Roaring selection and bounded profiles over prepared scalar columns."""

from __future__ import annotations

import math
from array import array
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray
from pyroaring import BitMap

from premixdb.internal import analytics_pb2 as a
from premixdb.storage.profiles import Histogram, ProfileKind, endpoint, value
from premixdb.v1 import profile_pb2 as p
from premixdb.v1 import query_pb2 as q

type Scalar = int | float | str | bool
type Numbers = NDArray[np.int64 | np.uint64 | np.float64]
type Ordinals = NDArray[np.uint32]

BLOCK_ROWS = 1_000_000


def bitmap(mask: NDArray[np.bool_]) -> BitMap:
    """Bulk construction avoids converting each NumPy scalar through Python."""
    return BitMap(array("I", np.flatnonzero(mask).astype(np.uint32).tobytes()))


def ordinals(selected: BitMap) -> Ordinals:
    return np.frombuffer(selected.to_array(), dtype=np.uint32)


@dataclass(frozen=True)
class Moments:
    count: int
    total: float
    mean: float
    squared_deviations: float

    def without(self, other: Moments) -> Moments:
        count = self.count - other.count
        if not count:
            return Moments(0, 0.0, 0.0, 0.0)
        delta = other.mean - self.mean
        mean = self.mean - delta * other.count / count
        squared = (
            self.squared_deviations
            - other.squared_deviations
            - delta * delta * self.count * other.count / count
        )
        return Moments(count, self.total - other.total, mean, max(0.0, squared))


def moments(values: Numbers, origin: int | float) -> Moments:
    if not len(values):
        return Moments(0, 0.0, 0.0, 0.0)
    if values.dtype.kind in "iu":
        # Subtract before conversion: 2**60 and 2**60+1 must remain distinct.
        maximum, minimum = int(np.max(values)), int(np.min(values))
        if minimum >= origin and maximum - origin <= (
            2**64 - 1 if values.dtype.kind == "u" else 2**63 - 1
        ):
            offsets = (values - np.array(origin, dtype=values.dtype)).astype(np.float64)
        elif values.dtype.kind == "i" and minimum - origin >= -(2**63) and maximum - origin < 2**63:
            offsets = (values - np.array(origin, dtype=values.dtype)).astype(np.float64)
        else:
            offsets = np.array([int(v) - origin for v in values], dtype=np.float64)
        if len(values) * max(abs(maximum), abs(minimum)) <= np.iinfo(np.int64).max:
            total = float(np.sum(values, dtype=np.int64))
        else:
            total = float(sum(int(v) for v in values))
    else:
        offsets = values - origin
        total = float(np.sum(values))
    mean = float(np.mean(offsets))
    squared = float(np.sum((offsets - mean) ** 2))
    return Moments(len(values), total, mean, squared)


class Column:
    """One immutable block. Numeric dictionaries retain the original scalar type.

    Dictionary ranks are used only for comparisons, never numeric aggregation.
    Nulls do not belong to any ordinary comparison, including not-equal.
    """

    def __init__(
        self,
        kind: ProfileKind,
        values: Numbers,
        dictionary: pa.Array,
        ranks: Ordinals,
        bucket_codes: Ordinals,
        present: BitMap,
        planes: list[BitMap],
        buckets: list[BitMap],
        *,
        full: p.FieldDistribution | None = None,
        origin: int | float = 0,
        population_moments: Moments | None = None,
        bounds: list[tuple[Scalar, Scalar]] | None = None,
    ) -> None:
        self.kind = kind
        self.values = values
        self.dictionary = dictionary
        self.ranks = ranks
        self.bucket_codes = bucket_codes
        self.present = present
        self.planes = planes
        self.buckets = buckets
        self.origin = origin
        self.population_moments = population_moments
        self.bounds = (
            bounds
            if bounds is not None
            else [
                (
                    self.scalar((i * len(dictionary) + len(buckets) - 1) // len(buckets)),
                    self.scalar(((i + 1) * len(dictionary) + len(buckets) - 1) // len(buckets) - 1),
                )
                for i in range(len(buckets))
            ]
        )
        self.full = full or self.distribution(BitMap(range(len(values))))

    def __sizeof__(self) -> int:
        # NumPy views keep mapped Arrow buffers alive; include that native storage.
        return object.__sizeof__(self) + sum(
            a.nbytes for a in (self.values, self.ranks, self.bucket_codes)
        )

    @classmethod
    def build(
        cls,
        values: Sequence[Scalar | None],
        kind: ProfileKind,
        layout: p.FieldDistribution | None = None,
    ) -> Column:
        present_mask = np.array([v is not None for v in values], dtype=np.bool_)
        present = bitmap(present_mask)
        if kind == "text":
            labels = sorted({cast(str, v) for v in values if v is not None})
            dictionary = pa.array(labels, type=pa.string())
            lookup = {label: rank for rank, label in enumerate(labels)}
            numbers: Numbers = np.array(
                [lookup[cast(str, v)] if v is not None else 0 for v in values], dtype=np.int64
            )
            ranks = numbers.astype(np.uint32)
        else:
            dtype = np.uint64 if kind == "count" else np.float64 if kind == "number" else np.int64
            numbers = np.array([0 if v is None else v for v in values], dtype=dtype)
            distinct, inverse = np.unique(numbers[present_mask], return_inverse=True)
            dictionary = pa.array(distinct)
            ranks = np.zeros(len(values), dtype=np.uint32)
            ranks[present_mask] = inverse.astype(np.uint32)
        groups = min(64, len(dictionary))
        codes = (ranks.astype(np.uint64) * groups // max(1, len(dictionary))).astype(np.uint32)
        bounds = None
        if layout is not None:
            bounds = [(value(b.lower), value(b.upper)) for b in layout.buckets]
            groups = len(bounds)
            dtype = object if kind == "text" else numbers.dtype
            upper = np.array([b[1] for b in bounds], dtype=dtype)
            memberships = np.searchsorted(upper, dictionary.to_numpy(zero_copy_only=False)).astype(
                np.uint32
            )
            if len(memberships) and int(np.max(memberships)) >= groups:
                raise ValueError("histogram layout does not cover its column")
            codes = (
                memberships[ranks] if len(memberships) else np.zeros(len(values), dtype=np.uint32)
            )
        planes = [
            bitmap(present_mask & ((ranks & (1 << bit)) != 0))
            for bit in range(max(0, len(dictionary) - 1).bit_length())
        ]
        buckets = [bitmap(present_mask & (codes == group)) for group in range(groups)]
        origin: int | float = 0
        numeric = None
        if kind in ("count", "integer", "number") and len(present):
            raw = dictionary[0].as_py()
            assert isinstance(raw, (int, float))
            origin = raw
            numeric = moments(numbers[present_mask], origin)
        return cls(
            kind,
            numbers,
            dictionary,
            ranks,
            codes,
            present,
            planes,
            buckets,
            origin=origin,
            population_moments=numeric,
            bounds=bounds,
        )

    def scalar(self, rank: int) -> Scalar:
        result = self.dictionary[rank].as_py()
        if not isinstance(result, (int, float, str, bool)):
            raise ValueError("invalid scalar dictionary")
        return bool(result) if self.kind == "boolean" else result

    def rank(self, threshold: Scalar, side: Literal["left", "right"]) -> int:
        # Python's mixed integer/float comparisons are exact even above 2**53.
        search = bisect_left if side == "left" else bisect_right
        return search(range(len(self.dictionary)), threshold, key=self.scalar)

    def ge_rank(self, rank: int, selected: BitMap) -> BitMap:
        equal = selected & self.present
        if rank == 0:
            return equal
        if rank >= len(self.dictionary):
            return BitMap()
        greater = BitMap()
        for bit in range(len(self.planes) - 1, -1, -1):
            plane = self.planes[bit]
            if rank & (1 << bit):
                equal &= plane
            else:
                greater |= equal & plane
                equal -= plane
        return greater | equal

    def select(
        self, selected: BitMap, operator: q.Comparison.Operator, threshold: Scalar
    ) -> BitMap:
        lower = self.rank(threshold, "left")
        upper = self.rank(threshold, "right")
        if operator == q.Comparison.OPERATOR_GE:
            return self.ge_rank(lower, selected)
        if operator == q.Comparison.OPERATOR_GT:
            return self.ge_rank(upper, selected)
        if operator == q.Comparison.OPERATOR_LT:
            return (selected & self.present) - self.ge_rank(lower, selected)
        if operator == q.Comparison.OPERATOR_LE:
            return (selected & self.present) - self.ge_rank(upper, selected)
        equal = self.ge_rank(lower, selected) - self.ge_rank(upper, selected)
        if operator == q.Comparison.OPERATOR_EQ:
            return equal
        if operator == q.Comparison.OPERATOR_NE:
            return (selected & self.present) - equal
        raise ValueError("invalid comparison operator")

    def bound(self, selected: BitMap, maximum: bool = False) -> Scalar:
        candidates = selected & self.present
        if not candidates:
            raise ValueError("empty scalar population")
        for plane in reversed(self.planes):
            ones = candidates & plane
            zeros = candidates - plane
            candidates = (ones or zeros) if maximum else (zeros or ones)
        return self.scalar(int(self.ranks[candidates.min()]))

    def selected_moments(self, selected: BitMap) -> Moments:
        assert self.population_moments is not None
        chosen = selected & self.present
        removed = self.present - chosen
        if len(removed) < len(chosen):
            return self.population_moments.without(
                moments(self.values[ordinals(removed)], self.origin)
            )
        return moments(self.values[ordinals(chosen)], self.origin)

    def distribution(self, selected: BitMap) -> p.FieldDistribution:
        chosen = selected & self.present
        result = p.FieldDistribution()
        if not chosen:
            return result
        # Preserve exact categories and the existing bounded histogram for small slices.
        if len(chosen) <= 4096:
            histogram = Histogram(self.kind)
            for ordinal in chosen:
                histogram.add(self.scalar(int(self.ranks[ordinal])))
            return histogram.proto()
        if len(chosen) < len(self.values) // 20:
            counts = np.bincount(self.bucket_codes[ordinals(chosen)], minlength=len(self.buckets))
        else:
            counts = [chosen.intersection_cardinality(bucket) for bucket in self.buckets]
        occupied = [i for i, count in enumerate(counts) if count]
        for index in occupied:
            lower, upper = self.bounds[index]
            if index == occupied[0]:
                lower = self.bound(chosen & self.buckets[index])
            if index == occupied[-1]:
                upper = self.bound(chosen & self.buckets[index], maximum=True)
            result.buckets.add(
                lower=endpoint(self.kind, lower),
                upper=endpoint(self.kind, upper),
                documents=int(counts[index]),
            )
        if self.population_moments is not None:
            numeric = self.selected_moments(chosen)
            result.numeric.CopyFrom(
                p.NumericSummary(
                    documents=numeric.count,
                    total=numeric.total,
                    mean=self.origin + numeric.mean,
                    standard_deviation=math.sqrt(numeric.squared_deviations / numeric.count),
                    minimum=result.buckets[0].lower,
                    maximum=result.buckets[-1].upper,
                )
            )
        return result

    def bitmap_proto(self) -> a.BitmapBlock:
        return a.BitmapBlock(
            present=self.present.serialize(),
            planes=[plane.serialize() for plane in self.planes],
            buckets=[bucket.serialize() for bucket in self.buckets],
        )
