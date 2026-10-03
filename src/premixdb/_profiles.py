"""Detached researcher summaries over persisted field profiles; no data reads."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field as dataclass_field

from ._curation import selector
from ._field_expr import FieldProjection, VectorField
from ._field_ids import field_name, selector_field
from ._requests import _Field
from .v1 import dataset_pb2 as d
from .v1 import profile_pb2 as p
from .v1 import query_pb2 as q
from .v1.snapshot_pb2 import SnapshotProfile as _SnapshotProfile

type ProfileScalar = int | float | str | bool
type ProfileSelector = (
    str | _Field[int] | _Field[str] | FieldProjection | VectorField | q.FieldComparison
)


def _value(endpoint: p.ProfileValue) -> ProfileScalar:
    kind = endpoint.WhichOneof("value")
    if kind == "integer":
        return endpoint.integer
    if kind == "count":
        return endpoint.count
    if kind == "number":
        return endpoint.number
    if kind == "boolean":
        return endpoint.boolean
    if kind == "text":
        return endpoint.text
    raise ValueError("profile endpoint has no value")


@dataclass(frozen=True)
class HistogramBucket:
    lower: ProfileScalar
    upper: ProfileScalar
    documents: int


@dataclass(frozen=True)
class QuantileRange:
    """Inclusive bounds on a nearest-rank quantile; equal bounds are exact."""

    lower: int | float
    upper: int | float


@dataclass(frozen=True)
class DistributionSummary:
    """One projection of a document population, excluding nulls from statistics.

    Numeric moments are unknown for legacy profiles. Category counts/fractions
    are unknown when any histogram bucket merges multiple values. Fractions use
    the non-null population; missing_fraction uses the complete population.
    """

    name: str
    projection: str
    class_name: str
    documents: int
    null_documents: int
    buckets: tuple[HistogramBucket, ...] = dataclass_field(repr=False)
    total: float | None = None
    mean: float | None = None
    standard_deviation: float | None = None

    @property
    def count(self) -> int:
        """Return the number of documents with a present field value."""
        return self.documents - self.null_documents

    @property
    def missing_fraction(self) -> float | None:
        """Return the fraction of documents with no value, or None for an empty population."""
        return self.null_documents / self.documents if self.documents else None

    @property
    def minimum(self) -> ProfileScalar | None:
        """Return the smallest observed value, or None when unavailable."""
        return self.buckets[0].lower if self.buckets else None

    @property
    def maximum(self) -> ProfileScalar | None:
        """Return the largest observed value, or None when unavailable."""
        return self.buckets[-1].upper if self.buckets else None

    @property
    def counts(self) -> dict[ProfileScalar, int] | None:
        """Return exact category counts when individual categories are retained."""
        if any(bucket.lower != bucket.upper for bucket in self.buckets):
            return None
        return {bucket.lower: bucket.documents for bucket in self.buckets}

    @property
    def fractions(self) -> dict[ProfileScalar, float] | None:
        """Return exact category fractions when individual categories are retained."""
        counts = self.counts
        if counts is None:
            return None
        return {label: count / self.count for label, count in counts.items()}

    def quantile(self, probability: float) -> QuantileRange | None:
        """Return bounds without interpolating unknown values within a bucket."""
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("quantile probability must be finite and in [0, 1]")
        if not self.buckets:
            return None
        if type(self.buckets[0].lower) not in (int, float):
            raise TypeError("quantiles require a numeric projection")
        rank = max(1, math.ceil(probability * self.count))
        cumulative = 0
        for bucket in self.buckets:
            cumulative += bucket.documents
            if cumulative >= rank:
                if not isinstance(bucket.lower, (int, float)) or not isinstance(
                    bucket.upper, (int, float)
                ):
                    raise TypeError("quantiles require numeric endpoints")
                return QuantileRange(bucket.lower, bucket.upper)
        raise ValueError("incomplete field profile coverage")


def _describe_field(
    profiles: Iterable[p.FieldProfile], field: ProfileSelector
) -> DistributionSummary:
    """Select a published projection, raising KeyError when it is unavailable."""
    selected = selector(field)
    identity = selector_field(selected)
    projection = selected.projection
    class_name = selected.class_name
    component = selected.component if selected.HasField("component") else None
    for profile in profiles:
        if profile.field != identity:
            continue
        for distribution in profile.distributions:
            if (
                distribution.projection != projection
                or distribution.class_name != class_name
                or (distribution.component if distribution.HasField("component") else None)
                != component
            ):
                continue
            if (
                profile.null_documents > profile.documents
                or sum(bucket.documents for bucket in distribution.buckets)
                != profile.documents - profile.null_documents
            ):
                raise ValueError("incomplete field profile coverage")
            numeric = distribution.numeric if distribution.HasField("numeric") else None
            return DistributionSummary(
                name=field_name(identity),
                projection=p.FieldDistribution.Projection.Name(distribution.projection),
                class_name=class_name,
                documents=profile.documents,
                null_documents=profile.null_documents,
                buckets=tuple(
                    HistogramBucket(_value(bucket.lower), _value(bucket.upper), bucket.documents)
                    for bucket in distribution.buckets
                ),
                total=numeric.total if numeric is not None else None,
                mean=numeric.mean if numeric is not None else None,
                standard_deviation=numeric.standard_deviation if numeric is not None else None,
            )
    raise KeyError(f"no published profile for {field_name(identity)} projection {projection}")


def _field_names(fields: list[p.FieldProfile]) -> str:
    names = list(dict.fromkeys(field_name(field.field) for field in fields))
    shown = ", ".join(names[:4]) or "none"
    return shown + (f", +{len(names) - 4} more" if len(names) > 4 else "")


def _snapshot_profile_text(self: object) -> str:
    # Match object.__repr__; validate before reading generated message fields.
    assert isinstance(self, _SnapshotProfile)
    profile = self
    return "\n".join(
        [
            "SnapshotProfile",
            f"  Documents: {profile.documents:,}",
            f"  Text: {profile.content_bytes:,} bytes; {profile.characters:,} characters",
            f"  Changes: {profile.added:,} added; {profile.changed:,} changed; {profile.removed:,} removed",
            f"  Unchanged: {profile.unchanged:,}; reused: {profile.reused:,}",
            f"  Fields: {_field_names(list(profile.fields))}",
        ]
    )


def _query_profile_text(self: object) -> str:
    # Match object.__repr__; validate before reading generated message fields.
    assert isinstance(self, q.QueryProfile)
    profile = self
    lines = [
        "QueryProfile",
        f"  Input: {profile.input_documents:,} documents in {profile.snapshots:,} snapshots",
        f"  Output: {profile.output_documents:,} document occurrences",
        f"  Text: {profile.output_content_bytes:,} bytes; {profile.output_characters:,} characters",
    ]
    if profile.input_documents:
        lines.append(f"  Retained: {profile.output_documents / profile.input_documents:.1%}")
    for index, step in enumerate(profile.steps[:4], 1):
        lines.append(
            f"  Step {index}: {step.input_documents:,} → {step.output_documents:,} documents"
        )
    if len(profile.steps) > 4:
        lines.append(f"  Further steps: {len(profile.steps) - 4}")
    if profile.HasField("decontamination"):
        policy = profile.decontamination
        lines.append(
            f"  Contamination: {policy.removed_documents:,} documents; {policy.removed_spans:,} spans removed"
        )
    if profile.HasField("sampling"):
        sampling = profile.sampling
        lines.append(
            f"  Sampling: {sampling.realized:,} / {sampling.requested:,} {sampling.unit}; overshoot {sampling.overshoot:,}"
        )
        lines.append(f"  Unique documents: {sampling.unique_documents:,}")
    lines.append(f"  Fields: {_field_names(list(profile.fields))}")
    return "\n".join(lines)


class _MixProfiles(list[d.DatasetProfile]):
    """Candidate data with a bounded display; indexing retains complete profiles."""

    def __repr__(self) -> str:
        lines = ["MixProfile", f"  Candidates: {len(self):,}"]
        if not self:
            return "\n".join(lines)
        rows = [("Candidate", "Content / budget", "Sequences", "Unique docs", "Repeats", "Padding")]
        for index, profile in enumerate(self[:10]):
            rows.append(
                (
                    str(index),
                    f"{profile.content_tokens:,} / {profile.planned_content_tokens:,}",
                    f"{profile.sequences:,}",
                    f"{profile.source_documents:,}",
                    f"{max(0, profile.document_occurrences - profile.source_documents):,}",
                    f"{profile.padding_tokens:,}",
                )
            )
        widths = [max(len(row[column]) for row in rows) for column in range(6)]
        lines.extend(
            "  "
            + "  ".join(
                value.ljust(width) for value, width in zip(row, widths, strict=True)
            ).rstrip()
            for row in rows
        )
        if len(self) > 10:
            lines.append(f"  +{len(self) - 10:,} more candidates; index this result for details")
        return "\n".join(lines)

    def __str__(self) -> str:
        return repr(self)


def _dataset_profile_text(self: object) -> str:
    # Match object.__repr__; validate before reading generated message fields.
    assert isinstance(self, d.DatasetProfile)
    profile = self
    return "\n".join(
        [
            "DatasetProfile",
            f"  Content tokens: {profile.content_tokens:,}; budget: {profile.planned_content_tokens:,}",
            f"  Sequences: {profile.sequences:,}; output tokens: {profile.output_tokens:,}",
            f"  Documents: {profile.source_documents:,} unique; {profile.document_occurrences:,} occurrences",
            f"  Repeats: {max(0, profile.document_occurrences - profile.source_documents):,}",
            f"  Special tokens: {profile.separator_tokens:,} separators; {profile.padding_tokens:,} padding",
            f"  Dropped tokens: {profile.dropped_tokens:,}",
        ]
    )


# These are our generated message classes. Keep their typed data and wire format
# intact while giving Python's print() and interactive display bounded output.

_SnapshotProfile.__str__ = _snapshot_profile_text
_SnapshotProfile.__repr__ = _snapshot_profile_text
q.QueryProfile.__str__ = _query_profile_text
q.QueryProfile.__repr__ = _query_profile_text

d.DatasetProfile.__str__ = _dataset_profile_text
d.DatasetProfile.__repr__ = _dataset_profile_text
