"""Compose lazy queries over one snapshot or a union of snapshots."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from ._policies import DecontaminateDefault, SamplerDefault
from .v1 import query_pb2 as queries

if TYPE_CHECKING:
    from ._resources import Query, Snapshot


class _SnapshotOperations:
    def union(self, *others: Snapshot) -> SnapshotUnion:
        """Combine snapshots into one population, counting each document once."""
        return SnapshotUnion((cast("Snapshot", self), *others))

    def query(
        self,
        *,
        steps: Iterable[queries.Operation] = (),
        decontaminate: queries.Decontaminate | DecontaminateDefault = DecontaminateDefault(),
        sampling: queries.QuerySampling | SamplerDefault = SamplerDefault(),
    ) -> Query:
        """Plan a lazy query; defaults retain all documents without resampling.

        Filters and dedupe steps infer the fields they need. Decontamination
        requires explicit references: decontaminate=p.decontaminate(reference).
        Decontamination and sampling policies are separate from steps.
        wait(), profile(), preview(), and reading results start execution.
        """
        return self.union().query(steps=steps, decontaminate=decontaminate, sampling=sampling)


@dataclass(frozen=True)
class SnapshotUnion:
    _snapshots: tuple[Snapshot, ...]

    def __repr__(self) -> str:
        from ._display import _id, _items

        inputs = _items([_id(s.id) for s in self._snapshots])
        return f"SnapshotUnion\n  Snapshots: {inputs}"

    def __post_init__(self) -> None:
        if not self._snapshots:
            raise ValueError("snapshot union cannot be empty")
        client = self._snapshots[0]._db
        if any(s._db is not client for s in self._snapshots):
            raise ValueError("union resources must belong to the same PremixDB session")

    def union(self, *others: Snapshot) -> SnapshotUnion:
        """Add snapshots without capturing or querying documents."""
        return SnapshotUnion((*self._snapshots, *others))

    def query(
        self,
        *,
        steps: Iterable[queries.Operation] = (),
        decontaminate: queries.Decontaminate | DecontaminateDefault = DecontaminateDefault(),
        sampling: queries.QuerySampling | SamplerDefault = SamplerDefault(),
    ) -> Query:
        """Plan the union once; infer required fields from its ordered steps."""
        return self._snapshots[0]._query_union(
            self._snapshots,
            steps=steps,
            decontaminate=None
            if isinstance(decontaminate, DecontaminateDefault)
            else decontaminate,
            sampling=None if isinstance(sampling, SamplerDefault) else sampling,
        )
