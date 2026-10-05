"""Lazy query rows and provenance over shared analytical population descriptors."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterator, Sequence
from copy import deepcopy
from typing import overload

import numpy as np
from pyroaring import BitMap

from premixdb.engine.analytics import ordinals
from premixdb.engine.contracts import Counts, Provenance, Selection
from premixdb.engine.identity import CodeVersion
from premixdb.engine.queries import Query, Row
from premixdb.internal import analytics_pb2 as a
from premixdb.storage.analytics import Population, load_column, local_selection
from premixdb.storage.selections import _summary
from premixdb.v1 import query_pb2 as q

LINEAGE_MAGIC = b"premixdb/indexed-lineage/v1\0"


class IndexedRows(Sequence[Row]):
    def __init__(self, population: Population, selected: BitMap) -> None:
        self.population, self.selected = population, selected

    def __len__(self) -> int:
        return len(self.selected)

    @overload
    def __getitem__(self, index: int) -> Row: ...
    @overload
    def __getitem__(self, index: slice) -> list[Row]: ...
    def __getitem__(self, index: int | slice) -> Row | list[Row]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if not 0 <= index < len(self):
            raise IndexError("row index out of range")
        return self.population.row(self.selected[index], index)

    def __iter__(self) -> Iterator[Row]:
        for index, ordinal in enumerate(self.selected):
            yield self.population.row(ordinal, index)


def total(population: Population, field: q.IntrinsicField, selected: BitMap) -> int:
    value = 0
    for block in population.intrinsic(field).projections[0].blocks:
        column = load_column(population.catalog, block)
        chosen = local_selection(selected, block)
        if len(chosen) > block.documents // 2:
            removed = BitMap(range(block.documents)) - chosen
            value += int(np.sum(column.values, dtype=np.uint64)) - int(
                np.sum(column.values[ordinals(removed)], dtype=np.uint64)
            )
        else:
            value += int(np.sum(column.values[ordinals(chosen)], dtype=np.uint64))
    return value


def counts(population: Population, selected: BitMap) -> Counts:
    return dict(
        documents=len(selected),
        bytes=total(population, q.FIELD_TEXT_BYTES, selected),
        characters=total(population, q.FIELD_TEXT_CHARACTERS, selected),
    )


class IndexedQuery(Query):
    """Ordinary Query with lazy rows and provenance backed by an immutable receipt."""

    def __init__(
        self, population: Population, selection: a.IndexedSelection, resource: q.Query
    ) -> None:
        self.population, self.receipt = population, selection
        self.selected = BitMap.deserialize(selection.selected)
        self.survivors = [BitMap.deserialize(data) for data in selection.steps]
        self.code = CodeVersion(selection.repository, selection.commit, selection.environment.hex())
        self._id = resource.id.hex()
        self.inputs = tuple(id.hex() for id in resource.snapshot_ids)
        self.field_snapshot_ids = tuple(resource.field_snapshot_ids)
        self._selection_rows = IndexedRows(population, self.selected)
        self._rows = self._selection_rows
        self._summary = _summary(resource.profile)
        self.steps = ()
        self.elapsed_seconds = 0.0
        self._encoding_provider = None
        previous = BitMap(range(population.manifest.documents))
        if (
            selection.query_id != resource.id
            or selection.population_id != population.manifest.id
            or list(selection.snapshot_ids) != list(resource.snapshot_ids)
            or list(population.manifest.snapshot_ids) != list(resource.snapshot_ids)
            or bytes.fromhex(self.code.commit) != resource.git_commit
            or len(self.survivors) != len(resource.operations)
            or len(self.selected) != resource.profile.output_documents
            or resource.profile.input_documents != population.manifest.documents
        ):
            raise ValueError("indexed selection belongs to a different recipe")
        for step, profile in zip(self.survivors, resource.profile.steps, strict=True):
            if (
                not step.issubset(previous)
                or len(step) != profile.output_documents
                or len(previous) != profile.input_documents
            ):
                raise ValueError("indexed selection has invalid step coverage")
            previous = step
        if previous != self.selected:
            raise ValueError("indexed selection differs from its final step")

    def source_counts(self) -> Counts:
        return (
            super().source_counts()
            if self._rows is not self._selection_rows
            else deepcopy(self._summary["output"])
        )

    def lengths(self) -> list[int]:
        if self._rows is not self._selection_rows:
            return super().lengths()
        result: list[int] = []
        for block in self.population.intrinsic(q.FIELD_TEXT_BYTES).projections[0].blocks:
            column = load_column(self.population.catalog, block)
            result.extend(column.values[ordinals(local_selection(self.selected, block))].tolist())
        return result

    def provenance(self) -> dict[str, Provenance]:
        result: dict[str, Provenance] = {}
        for ordinal in range(self.population.manifest.documents):
            row = self.population.row(ordinal, ordinal)
            decision: Selection
            if ordinal in self.selected:
                decision = dict(kind="retained", ordinal=self.selected.rank(ordinal) - 1)
            else:
                step = next(
                    i for i, selected in enumerate(self.survivors) if ordinal not in selected
                )
                decision = dict(kind="filtered", step=step)
            result[row.id] = dict(
                corpus_id=row.corpus_id,
                source_key=row.source_key,
                content=row.document.content.hex(),
                snapshots=self.population.origins(ordinal),
                selection=decision,
            )
        return result

    def provenance_for(self, row: Row) -> Provenance:
        from premixdb.engine.analytics import BLOCK_ROWS

        def document_id(ordinal: int) -> bytes:
            raw = self.population.part(ordinal // BLOCK_ROWS)["id"][ordinal % BLOCK_ROWS].as_py()
            assert isinstance(raw, bytes)
            return raw

        identity = bytes.fromhex(row.id)
        ordinal = self.selected[row.ordinal]
        if document_id(ordinal) != identity:
            ordinal = bisect_left(
                range(self.population.manifest.documents), identity, key=document_id
            )
        if ordinal not in self.selected or document_id(ordinal) != identity:
            raise ValueError("row is outside indexed selection")
        return dict(
            corpus_id=row.corpus_id,
            source_key=row.source_key,
            content=row.document.content.hex(),
            snapshots=self.population.origins(ordinal),
            selection=dict(kind="retained", ordinal=self.selected.rank(ordinal) - 1),
        )
