"""Bounded, deterministic field indexes and metadata-only cardinality planning.

Profiles describe document populations. Text span counts supply exact document
values; summing those values is never a substitute for their distribution.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import Counter
from typing import TYPE_CHECKING, Literal, Mapping, NotRequired, Sequence, TypedDict

from .._typing import FieldValue
from ..v1.storage_pb2 import ObjectRef

if TYPE_CHECKING:
    from ..engine.queries import Query
    from .catalog_reader import Catalog
    from .coordinator import Coordinator

from .._field_ids import field_id
from ..enrichment.classification import probabilities
from ..internal import derivation_pb2 as d
from ..v1 import field_pb2 as f
from ..v1 import profile_pb2 as p
from ..v1 import query_pb2 as q
from ..v1 import snapshot_pb2 as s

type Comparable = int | float | str | bool
type ProfileKind = Literal["count", "integer", "number", "boolean", "text"]


class ProjectionOptions(TypedDict):
    projection: NotRequired[p.FieldDistribution.Projection]
    class_name: NotRequired[str]


def compare(left: Comparable, right: Comparable) -> int:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return (left > right) - (left < right)
    if isinstance(left, str) and isinstance(right, str):
        return (left > right) - (left < right)
    raise TypeError("cannot compare numeric and text values")


def endpoint(kind: ProfileKind, item: Comparable) -> p.ProfileValue:
    if kind == "text" and isinstance(item, str):
        return p.ProfileValue(text=item)
    if kind == "boolean" and isinstance(item, bool):
        return p.ProfileValue(boolean=item)
    if kind == "count" and isinstance(item, int):
        return p.ProfileValue(count=item)
    if kind == "integer" and isinstance(item, int):
        return p.ProfileValue(integer=item)
    if kind == "number" and isinstance(item, (int, float)):
        return p.ProfileValue(number=item)
    raise ValueError("profile value does not match its kind")


def comparison_value(selector: q.Comparison | q.FieldComparison) -> Comparable:
    kind = selector.WhichOneof("value")
    if isinstance(selector, q.Comparison):
        if kind == "count":
            return selector.count
        if kind == "text":
            return selector.text
    else:
        if kind == "integer":
            return selector.integer
        if kind == "number":
            return selector.number
        if kind == "boolean":
            return selector.boolean
        if kind == "text":
            return selector.text
    raise ValueError("comparison has no value")


MAX_BUCKETS = 64


class TextProfiler:
    """Share one document pass for the intrinsic text and source distributions."""

    def __init__(self) -> None:
        self.documents = 0
        self.histograms = [
            (field, Histogram(kind, projection=p.FieldDistribution.SCALAR))
            for field, kind in (
                (q.FIELD_TEXT_BYTES, "count"),
                (q.FIELD_TEXT_CHARACTERS, "count"),
                (q.FIELD_OBJECT_URI, "text"),
                (q.FIELD_SOURCE_CORPUS_ID, "text"),
            )
        ]

    def add(self, content_bytes: int, characters: int, uri: str, corpus: str) -> None:
        self.documents += 1
        for (_, histogram), item in zip(
            self.histograms, (content_bytes, characters, uri, corpus), strict=True
        ):
            histogram.add(item)

    def proto(self) -> list[p.FieldProfile]:
        return [
            p.FieldProfile(field=field, documents=self.documents, distributions=[histogram.proto()])
            for field, histogram in self.histograms
        ]


def output_profiles(service: Coordinator, recipe: q.Query, handle: Query) -> list[p.FieldProfile]:
    """Profile the selected occurrences, retaining original external feature values."""
    from .enrichment import decode_value, load_build, read_rows

    occurrences = Counter()
    text = TextProfiler()
    for row in handle:
        occurrences[row.id] += 1
        text.add(row.document.size, row.document.characters, row.source_key, row.corpus_id)
    result = text.proto()
    for identity in recipe.field_snapshot_ids:
        build, manifest = load_build(service, "field", identity, recipe.snapshot_ids)
        profiler = FieldProfiler(build.field)
        for value_row in read_rows(service, "field", manifest):
            if count := occurrences[value_row.document_id.hex()]:
                profiler.add(decode_value(build.field, value_row), occurrences=count)
        result.append(profiler.proto())
    return result


def value(endpoint: p.ProfileValue) -> Comparable:
    kind = endpoint.WhichOneof("value")
    if kind == "count":
        return endpoint.count
    if kind == "integer":
        return endpoint.integer
    if kind == "number":
        return endpoint.number
    if kind == "boolean":
        return endpoint.boolean
    if kind == "text":
        return endpoint.text
    raise ValueError("profile endpoint has no value")


class Histogram:
    """Keep exact frequencies until full, then merge adjacent sparse ranges.

    Memory is bounded independently of population size. Insertion order is
    canonical document order, so changing worker shard size changes no totals.
    """

    def __init__(
        self,
        kind: ProfileKind,
        *,
        projection: p.FieldDistribution.Projection | None = None,
        class_name: str = "",
    ) -> None:
        self.kind = kind
        self.projection: ProjectionOptions = (
            {} if projection is None else dict(projection=projection)
        )
        if class_name:
            self.projection["class_name"] = class_name
        self.buckets: list[tuple[Comparable, Comparable, int]] = []
        self.documents = 0
        self.total = 0
        self.origin: int | float | None = None
        self.mean_offset = 0.0
        self.squared_deviations = 0.0

    def add(self, item: FieldValue) -> None:
        if item is None or isinstance(item, list):
            raise ValueError("histograms require nonnull scalar values")
        if self.kind in ("count", "integer", "number"):
            if not isinstance(item, (int, float)):
                raise ValueError("numeric histograms require numeric values")
            if not math.isfinite(item):
                raise ValueError("numeric profiles require finite values")
            self.documents += 1
            self.total += item
            if self.origin is None:
                self.origin = item
            # Subtract in the original type before converting to float. This
            # preserves small differences between integers above 2^53.
            offset = item - self.origin
            delta = offset - self.mean_offset
            self.mean_offset += delta / self.documents
            self.squared_deviations += delta * (offset - self.mean_offset)
        index = bisect_right(self.buckets, item, key=lambda bucket: bucket[0])
        if index:
            lower, upper, count = self.buckets[index - 1]
            if compare(item, upper) <= 0:
                self.buckets[index - 1] = (lower, upper, count + 1)
                return
        self.buckets.insert(index, (item, item, 1))
        if len(self.buckets) > MAX_BUCKETS:
            index = min(
                range(len(self.buckets) - 1),
                key=lambda i: self.buckets[i][2] + self.buckets[i + 1][2],
            )
            lower, _, left = self.buckets[index]
            _, upper, right = self.buckets[index + 1]
            self.buckets[index : index + 2] = [(lower, upper, left + right)]

    def proto(self) -> p.FieldDistribution:
        result = p.FieldDistribution(
            **self.projection,
            buckets=[
                p.ProfileBucket(
                    lower=endpoint(self.kind, lower),
                    upper=endpoint(self.kind, upper),
                    documents=count,
                )
                for lower, upper, count in self.buckets
            ],
        )
        if self.documents:
            assert self.origin is not None
            result.numeric.CopyFrom(
                p.NumericSummary(
                    documents=self.documents,
                    minimum=endpoint(self.kind, self.buckets[0][0]),
                    maximum=endpoint(self.kind, self.buckets[-1][1]),
                    total=self.total,
                    mean=self.origin + self.mean_offset,
                    standard_deviation=math.sqrt(
                        max(0.0, self.squared_deviations / self.documents)
                    ),
                )
            )
        return result


class FieldProfiler:
    def __init__(self, spec: f.Field) -> None:
        self.profile = p.FieldProfile(field=field_id(spec.name))
        self.spec = spec
        self.histograms = []
        if spec.HasField("classification"):
            if spec.classification.transform == f.PROBABILITY_TRANSFORM_SOFTMAX:
                self.histograms.append(Histogram("text", projection=p.FieldDistribution.TOP_CLASS))
            self.histograms.extend(
                Histogram("number", projection=p.FieldDistribution.CLASS_PROBABILITY, class_name=c)
                for c in spec.classification.classes
            )
        elif not spec.length:
            kinds: dict[int, ProfileKind] = {
                f.VALUE_INT64: "integer",
                f.VALUE_FLOAT32: "number",
                f.VALUE_FLOAT64: "number",
                f.VALUE_BOOL: "boolean",
                f.VALUE_STRING: "text",
            }
            self.histograms.append(
                Histogram(kinds[spec.element_type], projection=p.FieldDistribution.SCALAR)
            )

    def add(self, item: FieldValue, *, occurrences: int = 1) -> None:
        """Project once; repeated occurrences keep the original histogram arithmetic."""
        if type(occurrences) is not int or occurrences < 0:
            raise ValueError("profile occurrences must be a nonnegative integer")
        if not occurrences:
            return
        self.profile.documents += occurrences
        if item is None:
            self.profile.null_documents += occurrences
            return
        if self.spec.HasField("classification"):
            if not isinstance(item, list) or not all(isinstance(v, (int, float)) for v in item):
                raise ValueError("classification requires a numeric vector")
            scores = probabilities(
                self.spec, [float(v) for v in item if isinstance(v, (float, int))]
            )
            for histogram in self.histograms:
                value = (
                    max(scores, key=scores.__getitem__)
                    if histogram.projection["projection"] == p.FieldDistribution.TOP_CLASS
                    else scores[histogram.projection["class_name"]]
                )
                for _ in range(occurrences):
                    histogram.add(value)
        elif self.histograms:
            for _ in range(occurrences):
                self.histograms[0].add(item)

    def proto(self) -> p.FieldProfile:
        result = p.FieldProfile()
        result.CopyFrom(self.profile)
        result.distributions.extend(h.proto() for h in self.histograms)
        return result


def text_profiles(objects: Mapping[str, ObjectRef], corpus_id: bytes) -> list[p.FieldProfile]:
    """Aggregate exact object values; logical URIs differ from storage URIs."""
    text = TextProfiler()
    for key, obj in sorted(objects.items()):
        text.add(obj.profile.content_bytes, obj.profile.characters, key, corpus_id.hex())
    return text.proto()


def bucket_bounds(
    bucket: p.ProfileBucket, op: q.Comparison.Operator, expected: Comparable
) -> tuple[int, int]:
    lower, upper = value(bucket.lower), value(bucket.upper)
    if op == q.Comparison.OPERATOR_EQ:
        return int(lower == upper == expected), int(
            compare(lower, expected) <= 0 and compare(expected, upper) <= 0
        )
    if op == q.Comparison.OPERATOR_NE:
        minimum, maximum = bucket_bounds(bucket, q.Comparison.OPERATOR_EQ, expected)
        return 1 - maximum, 1 - minimum
    if op == q.Comparison.OPERATOR_LT:
        return int(compare(upper, expected) < 0), int(compare(lower, expected) < 0)
    if op == q.Comparison.OPERATOR_LE:
        return int(compare(upper, expected) <= 0), int(compare(lower, expected) <= 0)
    if op == q.Comparison.OPERATOR_GT:
        return int(compare(lower, expected) > 0), int(compare(upper, expected) > 0)
    if op == q.Comparison.OPERATOR_GE:
        return int(compare(lower, expected) >= 0), int(compare(upper, expected) >= 0)
    raise ValueError("invalid comparison operator")


def _possible(
    bucket: p.ProfileBucket, selectors: Sequence[q.FieldComparison | q.Comparison]
) -> bool:
    """Intersect predicates within a bucket, preserving strict endpoints."""
    lower, upper = value(bucket.lower), value(bucket.upper)
    lower_closed = upper_closed = True
    excluded = set()
    for selector in selectors:
        expected = comparison_value(selector)
        op = selector.operator
        if op == q.Comparison.OPERATOR_NE:
            excluded.add(expected)
        elif op == q.Comparison.OPERATOR_EQ:
            if not (compare(lower, expected) <= 0 and compare(expected, upper) <= 0):
                return False
            if expected == lower and not lower_closed or expected == upper and not upper_closed:
                return False
            lower = upper = expected
            lower_closed = upper_closed = True
        elif op in (q.Comparison.OPERATOR_GT, q.Comparison.OPERATOR_GE):
            closed = op == q.Comparison.OPERATOR_GE
            if compare(expected, lower) > 0:
                lower, lower_closed = expected, closed
            elif expected == lower:
                lower_closed &= closed
        elif op in (q.Comparison.OPERATOR_LT, q.Comparison.OPERATOR_LE):
            closed = op == q.Comparison.OPERATOR_LE
            if compare(expected, upper) < 0:
                upper, upper_closed = expected, closed
            elif expected == upper:
                upper_closed &= closed
    if bucket.lower.WhichOneof("value") in ("integer", "count"):
        # Integer projections can be compared with floating thresholds.
        assert isinstance(lower, (int, float)) and isinstance(upper, (int, float))
        lower = math.ceil(lower) if lower_closed else math.floor(lower) + 1
        upper = math.floor(upper) if upper_closed else math.ceil(upper) - 1
        if lower <= upper and upper - lower + 1 <= len(excluded):
            return any(item not in excluded for item in range(lower, upper + 1))
        lower_closed = upper_closed = True
    return compare(lower, upper) < 0 or (
        lower == upper and lower_closed and upper_closed and lower not in excluded
    )


def predicate_bounds(
    profile: p.FieldProfile,
    selector: q.Comparison | q.FieldComparison,
    *,
    selectors: Sequence[q.FieldComparison | q.Comparison] | None = None,
) -> tuple[int, int] | None:
    """Conservative match counts, or None for an unprofiled projection."""
    if profile.null_documents > profile.documents:
        raise ValueError("invalid field profile null coverage")
    selectors = selectors or [selector]
    projection = (
        selector.projection
        if isinstance(selector, q.FieldComparison)
        else p.FieldDistribution.SCALAR
    )
    if projection == q.FieldComparison.IS_NULL:
        matches = sum(
            count
            for is_null, count in (
                (True, profile.null_documents),
                (False, profile.documents - profile.null_documents),
            )
            if all(
                bucket_bounds(
                    p.ProfileBucket(
                        lower=p.ProfileValue(boolean=is_null),
                        upper=p.ProfileValue(boolean=is_null),
                    ),
                    sel.operator,
                    comparison_value(sel),
                )[0]
                for sel in selectors
            )
        )
        return matches, matches
    nonnull = profile.documents - profile.null_documents
    if not nonnull:
        return 0, 0
    for distribution in profile.distributions:
        if distribution.projection != projection:
            continue
        if isinstance(selector, q.FieldComparison) and (
            distribution.class_name != selector.class_name
            or distribution.HasField("component") != selector.HasField("component")
            or distribution.component != selector.component
        ):
            continue
        if sum(bucket.documents for bucket in distribution.buckets) != nonnull:
            raise ValueError("incomplete field profile coverage")
        lower = upper = 0
        for bucket in distribution.buckets:
            lo = all(
                bucket_bounds(bucket, sel.operator, comparison_value(sel))[0] for sel in selectors
            )
            hi = _possible(bucket, selectors)
            lower += bucket.documents * lo
            upper += bucket.documents * hi
        return lower, upper
    return None


def document_estimate(lower: int, upper: int, *, known: bool = True) -> p.DocumentEstimate:
    result = p.DocumentEstimate(lower=lower, upper=upper)
    if known or lower == upper:
        result.expected = lower + (upper - lower) / 2
    return result


def estimate_query(service: Catalog, query: q.Query | q.CreateQueryRequest) -> p.QueryEstimate:
    """Use published profiles only: never open text, value shards or models.

    Marginal histograms do not establish cross-field independence. Sequential
    filters use intersection bounds. Overlapping snapshot unions have population
    bounds until a derivation publishes a profile for the complete union.
    """
    snapshots = [service._storage.load("snapshot", id, s.Snapshot) for id in query.snapshot_ids]
    totals = [snapshot.profile.documents for snapshot in snapshots]
    profiles = {}
    lower, upper = max(totals, default=0), sum(totals)
    if len(snapshots) == 1:
        profiles.update((profile.field, profile) for profile in snapshots[0].profile.fields)
    population = None
    for id in query.field_snapshot_ids:
        try:
            build = service._storage.load("field", id, d.FieldBuild)
        except KeyError:
            continue
        snapshot = build.snapshot
        if snapshot.id != id or list(snapshot.snapshot_ids) != list(query.snapshot_ids):
            raise ValueError("field profile belongs to a different query population")
        if not snapshot.HasField("profile"):
            continue
        profile = snapshot.profile
        if profile.field != field_id(build.field.name):
            raise ValueError("field profile does not match its build")
        if population is not None and population != profile.documents:
            raise ValueError("field profiles disagree on query population")
        population = profile.documents
        profiles[profile.field] = profile
    if population is not None:
        if not lower <= population <= upper:
            raise ValueError("field profile population is outside snapshot union bounds")
        lower = upper = population
    result = p.QueryEstimate(input=document_estimate(lower, upper, known=lower == upper))
    universe = upper
    missing = set()
    groups = {}
    group_bounds = {}
    opaque = False
    # After a line transform, byte/character marginals describe obsolete text.
    transformed = False
    for operation in query.operations:
        kind = operation.WhichOneof("kind")
        known = False
        if kind in ("where", "field_where"):
            selector = operation.field_where if kind == "field_where" else operation.where
            field = selector.field
            profile = profiles.get(field)
            if transformed and field in (q.FIELD_TEXT_BYTES, q.FIELD_TEXT_CHARACTERS):
                profile = None
            bounds = predicate_bounds(profile, selector) if profile is not None else None
            if bounds is None:
                missing.add(field)
                lower = 0
                opaque = True
            else:
                key = (
                    field,
                    selector.projection
                    if isinstance(selector, q.FieldComparison)
                    else p.FieldDistribution.SCALAR,
                    selector.class_name if isinstance(selector, q.FieldComparison) else "",
                    selector.component if isinstance(selector, q.FieldComparison) else None,
                )
                groups.setdefault(key, []).append(selector)
                assert profile is not None
                combined = predicate_bounds(profile, selector, selectors=groups[key])
                assert combined is not None
                group_bounds[key] = combined
                upper = min(upper, group_bounds[key][1])
                if not opaque:
                    lower = max(
                        0,
                        sum(lo for lo, hi in group_bounds.values())
                        - (len(group_bounds) - 1) * universe,
                    )
                known = not opaque
        else:
            # Winner/removal policies need joint evidence, not marginal counts.
            lower = 0
            opaque = True
            transformed |= kind == "dedupe" and (
                operation.dedupe.algorithm == q.Dedupe.ALGORITHM_EXACT_LINE
            )
        result.steps.append(document_estimate(lower, upper, known=known))
    result.output.CopyFrom(result.steps[-1] if result.steps else result.input)
    result.fields.extend(profiles[field] for field in sorted(profiles))
    result.unavailable_fields.extend(sorted(missing))
    return result
