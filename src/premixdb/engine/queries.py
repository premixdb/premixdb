"""Reusable exact evidence indexes and deterministic ordered query selection."""

from __future__ import annotations

import operator
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, Mapping, Sequence

from premixdb.contracts import EvidenceRange, FieldValue
from premixdb.engine.contracts import (
    Contamination,
    Counts,
    PackingProfile,
    Provenance,
    QuerySummary,
    Range,
    SamplingStatistics,
    Selection,
)
from premixdb.engine.curation import ReferenceProvider, SelectedDocument
from premixdb.engine.identity import CodeVersion
from premixdb.engine.snapshots import Snapshot

if TYPE_CHECKING:
    from premixdb.engine.datasets import ByteTokens, Dataset, HuggingFaceTokenizer, TokenList
    from premixdb.engine.mixtures import MixturePool

from premixdb.engine import plans
from premixdb.engine.curation import RetainedDocument
from premixdb.engine.dataset_plan import PackingPlan
from premixdb.engine.snapshots import Document, counts

_OPS = dict(
    eq=operator.eq, ne=operator.ne, lt=operator.lt, le=operator.le, gt=operator.gt, ge=operator.ge
)


def _value(document: SelectedDocument, field: str) -> int | str:
    if field == "bytes":
        return document.size
    if field == "characters":
        return document.characters
    return document.source_key


@dataclass(frozen=True)
class Row:
    ordinal: int
    document: Document | RetainedDocument

    @property
    def id(self) -> str:
        return self.document.id

    @property
    def corpus_id(self) -> str:
        return self.document.corpus_id

    @property
    def source_key(self) -> str:
        return self.document.source_key

    @property
    def text(self) -> str:
        return self.document.text


def _range(id: str, start: int, end: int) -> Range:
    return Range(document=id, start=start, end=end)


class CorpusIndex:
    def __init__(
        self,
        snapshots: Iterable[Snapshot],
        document_hashes: Iterable[tuple[str, str]] | None = None,
    ) -> None:
        snapshot_map = {s.id: s for s in snapshots}
        self.inputs = sorted(snapshot_map)
        documents: dict[str, Document] = {}
        origins: dict[str, list[str]] = {}
        for sid, snapshot in sorted(snapshot_map.items()):
            for document in snapshot.documents.values():
                documents[document.id] = document
                origins.setdefault(document.id, []).append(sid)
        self.documents = dict(sorted(documents.items()))
        self.origins = origins
        self._classes: dict[str, list[list[EvidenceRange]]] = {}
        self.field_values: dict[str, Mapping[str, FieldValue]] = {}
        self.class_provider: (
            Callable[[CorpusIndex, str], Iterable[Iterable[EvidenceRange]]] | None
        ) = None
        self.reference_provider: ReferenceProvider | None = None
        self.external = document_hashes is not None
        if document_hashes is not None:
            hashes = dict(document_hashes)
            if hashes != {id: d.content.hex() for id, d in self.documents.items()}:
                raise ValueError("external hashes do not cover the immutable corpus")

    def classes(self, unit: str) -> Iterable[Iterable[EvidenceRange]]:
        if self.class_provider is not None:
            return self.class_provider(self, unit)
        if sum(document.size for document in self.documents.values()) > 8 * 1024 * 1024:
            from premixdb.engine.spill import classes

            return classes(self, unit)
        if unit not in self._classes:
            classes: dict[bytes, list[EvidenceRange]] = {}
            for id, document in self.documents.items():
                data = document.text.encode()
                if unit == "Document":
                    classes.setdefault(data, []).append((id, 0, len(data)))
                else:
                    start = 0
                    for line in data.split(b"\n"):
                        if line:
                            classes.setdefault(line, []).append((id, start, start + len(line)))
                        start += len(line) + 1
            self._classes[unit] = list(classes.values())
        return self._classes[unit]

    def reference_matches(
        self, targets: list[str], references: Sequence[str], unit: str = "Document"
    ) -> list[tuple[EvidenceRange, EvidenceRange]]:
        if unit not in ("Document", "Line"):
            raise ValueError("unknown comparison unit")
        if (
            not targets
            or not references
            or not (set(targets) | set(references)) <= set(self.inputs)
        ):
            raise ValueError("reference lookup requires indexed target and reference snapshots")
        target_ids = {id for id, origins in self.origins.items() if set(origins) & set(targets)}
        reference_ids = {
            id for id, origins in self.origins.items() if set(origins) & set(references)
        }
        matches = []
        for group_values in self.classes(unit):
            group = list(group_values)
            refs = [r for r in group if r[0] in reference_ids]
            if refs:
                reference = min(refs)
                matches.extend((r, reference) for r in group if r[0] in target_ids)
        return sorted(matches)

    def execute(
        self, plan: Iterable[plans.Step], version: CodeVersion, fields: tuple[bytes, ...] = ()
    ) -> Query:
        return Query(self, plan, version, fields)


@dataclass(frozen=True)
class CompletedQuery:
    """Validated selection state; constructing a Query from it never executes a recipe."""

    id: str
    inputs: tuple[str, ...]
    field_snapshot_ids: tuple[bytes, ...]
    rows: list[Row]
    provenance: dict[str, Provenance]
    summary: QuerySummary


class Query:
    def __init__(
        self,
        index: CorpusIndex | CompletedQuery,
        plan: Iterable[plans.Step],
        code: CodeVersion,
        fields: tuple[bytes, ...] = (),
    ) -> None:
        start = time.monotonic()
        self.code = code
        self._encoding_provider: (
            Callable[[Row, HuggingFaceTokenizer], ByteTokens | TokenList] | None
        ) = None
        if isinstance(index, CompletedQuery):
            self.steps: tuple[plans.Step, ...] = ()
            self._id = index.id
            self.inputs = index.inputs
            self.field_snapshot_ids = index.field_snapshot_ids
            self._rows = list(index.rows)
            self._provenance = deepcopy(index.provenance)
            self._summary = deepcopy(index.summary)
            self.elapsed_seconds = 0.0
            return
        steps = tuple(plans.validate(step) for step in plan)
        self.steps = steps
        self._id = plans.query_identity(index.inputs, steps, code, fields)
        self.inputs = tuple(index.inputs)
        self.field_snapshot_ids = ()
        documents: list[SelectedDocument] = list(index.documents.values())
        provenance: dict[str, Provenance] = {
            id: dict(
                corpus_id=d.corpus_id,
                source_key=d.source_key,
                content=d.content.hex(),
                snapshots=list(index.origins[id]),
                selection=dict(kind="retained", ordinal=0),
            )
            for id, d in index.documents.items()
        }
        for step in steps:
            if step.kind == "Policy":
                continue
            if step.kind == "FilterIds":
                if not step.members <= index.documents.keys():
                    raise ValueError("field selection is outside query population")
            elif step.kind == "DedupeIndexed" and not index.external:
                raise ValueError("external dedupe requires verified evidence")
            elif step.kind.startswith("Dedupe"):
                index.classes(step.unit)
        summary: QuerySummary = QuerySummary(
            input=counts(documents), steps=[], output=counts(documents)
        )
        for ordinal, step in enumerate(steps):
            before = counts(documents)
            if step.kind in ("Filter", "FilterIds", "FilterDocuments"):

                def keep(d: SelectedDocument) -> bool:
                    return (
                        d.id in step.members
                        if step.kind in ("FilterIds", "FilterDocuments")
                        else bool(_OPS[step.comparison](_value(d, step.field), step.value))
                    )

                retained: list[SelectedDocument] = []
                for d in documents:
                    if keep(d):
                        retained.append(d)
                    else:
                        provenance[d.id]["selection"] = dict(kind="filtered", step=ordinal)
                documents = retained
            elif step.kind == "Policy":
                from premixdb.engine import curation

                assert step.payload is not None
                if step.payload[0] == "decontaminate":
                    _, policy, payload = step.payload
                    matched = {d.id for d in documents}
                    documents = curation.decontaminate(
                        documents, payload, policy, provenance, provider=index.reference_provider
                    )
                    matches: list[list[Contamination]] = [
                        provenance[id]["contamination"]
                        for id in matched
                        if "contamination" in provenance[id]
                    ]
                    summary["decontamination"] = dict(
                        matched_documents=len(matches),
                        removed_documents=before["documents"] - len(documents),
                        removed_spans=sum(
                            len(curation.coalesce((m["start"], m["end"]) for m in group))
                            for group in matches
                        ),
                        removed_bytes=before["bytes"] - sum(d.size for d in documents),
                        reference_snapshot_ids=list(policy.snapshot_ids),
                    )
                elif step.payload[0] == "sample":
                    _, sampling, tokenizer = step.payload
                    statistics = SamplingStatistics(
                        unit="",
                        requested=0,
                        realized=0,
                        overshoot=0,
                        unique_documents=0,
                        document_occurrences=0,
                    )
                    documents = curation.sample(
                        documents, sampling, index.field_values, tokenizer, provenance, statistics
                    )
                    summary["sampling"] = statistics
                else:
                    assert step.payload[0] == "similarity"
                    _, orders, edge_source = step.payload
                    if edge_source is None:
                        raise ValueError("similarity policy has no evidence")
                    edges = (
                        edge_source if isinstance(edge_source, Iterable) else edge_source(documents)
                    )
                    documents = curation.greedy(
                        documents, edges, orders, index.field_values, provenance, ordinal
                    )
            else:
                documents = self._dedupe(index, documents, provenance, step, ordinal)
            summary["steps"].append(dict(before=before, after=counts(documents)))
        documents.sort(key=lambda d: d.id)
        self._rows = [Row(i, d) for i, d in enumerate(documents)]
        occurrences = {}
        for row in self._rows:
            ordinals = occurrences.setdefault(row.id, [])
            ordinals.append(row.ordinal)
            provenance[row.id]["selection"] = dict(kind="retained", ordinal=ordinals[0])
        for id, ordinals in occurrences.items():
            if len(ordinals) > 1:
                provenance[id]["occurrences"] = ordinals
        self._provenance = provenance
        summary["output"] = counts(documents)
        self._summary = summary
        self.elapsed_seconds = time.monotonic() - start

    @staticmethod
    def _dedupe(
        index: CorpusIndex,
        documents: list[SelectedDocument],
        provenance: dict[str, Provenance],
        policy: plans.Step,
        ordinal: int,
    ) -> list[SelectedDocument]:
        from premixdb.engine.curation import ordering

        def key(d: SelectedDocument) -> str | tuple[str, str]:
            if policy.separator is None:
                return d.id
            return d.corpus_id, d.source_key.split(policy.separator)[0]

        groups = {}
        for d in ordering(documents, policy.orders, index.field_values):
            groups.setdefault(key(d), d)
        ranks = {
            key(d): rank
            for rank, d in enumerate(ordering(groups.values(), policy.orders, index.field_values))
        }
        active = {d.id: key(d) for d in documents}
        removed = {}
        for evidence in index.classes(policy.unit):
            group = list(evidence)
            candidates = ((ranks[active[r[0]]], active[r[0]], r) for r in group if r[0] in active)
            winner = min(candidates, key=lambda u: (u[0], u[2]), default=None)
            if winner is None:
                continue
            winner_rank, _, kept = winner
            for matched in group:
                if matched[0] not in active:
                    continue
                group_key = active[matched[0]]
                rank = ranks[group_key]
                if rank == winner_rank:
                    continue
                decision = (winner_rank, matched[0], matched[1], kept[0], kept[1])
                if group_key not in removed or decision < removed[group_key][0]:
                    removed[group_key] = decision, matched, kept
        retained = []
        for d in documents:
            if key(d) not in removed:
                retained.append(d)
                continue
            _, matched, kept = removed[key(d)]
            selection: Selection = (
                dict(kind="duplicate", step=ordinal, kept=kept[0])
                if policy.kind != "Dedupe"
                else dict(
                    kind="duplicate_unit",
                    step=ordinal,
                    matched=_range(*matched),
                    kept=_range(*kept),
                )
            )
            provenance[d.id]["selection"] = selection
        return retained

    @property
    def id(self) -> str:
        return self._id

    @property
    def row_count(self) -> int:
        return len(self._rows)

    def row(self, index: int) -> Row:
        if not 0 <= index < len(self._rows):
            raise IndexError("row index out of range")
        return self._rows[index]

    def summary(self) -> QuerySummary:
        return deepcopy(self._summary)

    def provenance(self) -> dict[str, Provenance]:
        return deepcopy(self._provenance)

    def __iter__(self) -> Iterator[Row]:
        return iter(self._rows)

    def rows(self) -> list[Row]:
        return list(self._rows)

    def profile(
        self, sequence_length: int, separator: int | None, padding: int | None
    ) -> PackingProfile:
        return PackingPlan(sequence_length, separator, padding).profile(
            self.source_counts(), self.lengths()
        )

    def source_counts(self) -> Counts:
        return counts({r.id: r.document for r in self._rows}.values())

    def lengths(self) -> list[int]:
        return [r.document.size for r in self._rows]

    def dataset(
        self,
        length: int,
        separator: int | None,
        padding: int | None,
        tokenizer: HuggingFaceTokenizer | None = None,
        *,
        stream: bool = False,
    ) -> Dataset:
        from premixdb.engine.datasets import Dataset

        return Dataset.from_query(self, length, separator, padding, tokenizer, stream=stream)

    def mixture_pool(self, field: str, assignments: Mapping[str, str]) -> MixturePool:
        from premixdb.engine.mixtures import MixturePool

        return MixturePool(self, field, assignments)
