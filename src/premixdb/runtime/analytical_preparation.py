"""Prepare immutable analytical populations and enrichment columns once."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

import pyarrow as pa

from premixdb.engine.analytics import BLOCK_ROWS, Column, Scalar
from premixdb.engine.queries import Row
from premixdb.internal import analytics_pb2 as a
from premixdb.internal import derivation_pb2 as d
from premixdb.schemas.protobuf import wire
from premixdb.storage.analytics import (
    FIELD_SUFFIX,
    POPULATION_SUFFIX,
    Population,
    load_population,
    population_id,
    put_table,
    save_column,
    validate_index,
)
from premixdb.storage.profiles import ProfileKind
from premixdb.storage.selections import selection_record
from premixdb.v1 import profile_pb2 as p
from premixdb.v1 import query_pb2 as q

if TYPE_CHECKING:
    from premixdb.runtime.coordinator import Coordinator


def prepare_population(service: Coordinator, snapshot_ids: Iterable[bytes]) -> Population:
    inputs = tuple(sorted(set(snapshot_ids)))
    identity_ = population_id(inputs)
    try:
        return load_population(service, identity_)
    except KeyError:
        pass
    index = service._query_index(inputs)
    if len(index.documents) > 2**32:
        raise ValueError("analytical populations require at most 2**32 documents")
    manifest = a.PopulationIndex(id=identity_, snapshot_ids=inputs, documents=len(index.documents))
    intrinsic = [
        (q.FIELD_TEXT_BYTES, "count", "bytes"),
        (q.FIELD_TEXT_CHARACTERS, "count", "characters"),
        (q.FIELD_OBJECT_URI, "text", "uri"),
        (q.FIELD_SOURCE_CORPUS_ID, "text", "corpus"),
    ]
    from premixdb.storage.profiles import TextProfiler

    text = TextProfiler()
    for document in index.documents.values():
        text.add(document.size, document.characters, document.source_key, document.corpus_id)
    full_profiles = {profile.field: profile for profile in text.proto()}
    for field, _, _ in intrinsic:
        manifest.intrinsic.add(id=identity_, population_id=identity_, full=full_profiles[field])
    records, size = d.SelectionShard(), 0
    registry: dict[str, list[object]] = {
        name: []
        for name in ("id", "origins", "chunk", "position", "bytes", "characters", "uri", "corpus")
    }
    first = 0

    def descriptors() -> None:
        nonlocal size
        if records.rows:
            manifest.descriptors.append(service._storage.put("index", wire(records)))
            records.Clear()
            size = 0

    def columns() -> None:
        nonlocal first
        table = pa.table({name: pa.array(values) for name, values in registry.items()})
        manifest.columns.append(put_table(service._storage, table))
        for item, (_, kind, name) in zip(manifest.intrinsic, intrinsic, strict=True):
            column = Column.build(
                cast(list[Scalar | None], registry[name]),
                cast(ProfileKind, kind),
                item.full.distributions[0],
            )
            projection = (
                item.projections[0]
                if item.projections
                else item.projections.add(projection=p.FieldDistribution.SCALAR)
            )
            projection.blocks.append(save_column(service._storage, column, first))
        first += len(registry["id"])
        for values in registry.values():
            values.clear()

    for ordinal, document in enumerate(index.documents.values()):
        record = selection_record(Row(ordinal, document))
        width = record.ByteSize() + 10
        if size + width > 1024 * 1024 and records.rows:
            descriptors()
        registry["id"].append(bytes.fromhex(document.id))
        registry["origins"].append([bytes.fromhex(sid) for sid in index.origins[document.id]])
        registry["chunk"].append(len(manifest.descriptors))
        registry["position"].append(len(records.rows))
        registry["bytes"].append(document.size)
        registry["characters"].append(document.characters)
        registry["uri"].append(document.source_key)
        registry["corpus"].append(document.corpus_id)
        records.rows.append(record)
        size += width
        if len(registry["id"]) == BLOCK_ROWS:
            columns()
    descriptors()
    if registry["id"]:
        columns()
    if not manifest.documents:
        for item, (_, kind, _) in zip(manifest.intrinsic, intrinsic, strict=True):
            item.projections.add(projection=p.FieldDistribution.SCALAR).blocks.append(
                save_column(service._storage, Column.build([], cast(ProfileKind, kind)), 0)
            )
    population = Population(service, manifest)
    for item in manifest.intrinsic:
        manifest.profiles.append(item.full)
    service._storage.save("index", identity_, manifest, suffix=POPULATION_SUFFIX)
    return population


def prepare_field(
    service: Coordinator, population: Population, item: d.FieldBuild, manifest: d.EnrichmentManifest
) -> a.FieldIndex:
    """Create reusable columns once, validating every outcome against canonical IDs."""
    try:
        index = service._storage.load("field", item.snapshot.id, a.FieldIndex, suffix=FIELD_SUFFIX)
    except KeyError:
        pass
    else:
        if (
            index.id != item.snapshot.id
            or index.population_id != population.manifest.id
            or index.field_snapshot_id != item.snapshot.id
            or index.logical_digest != manifest.logical_digest
            or index.full != item.snapshot.profile
            or index.source_manifest_digest != item.snapshot.manifest.blake3_digest
        ):
            raise ValueError("analytical field identity or population mismatch")
        return index
    from premixdb.enrichment.classification import probabilities, top_class
    from premixdb.runtime.enrichment import decode_value, numeric_vector, read_rows
    from premixdb.v1 import field_pb2 as f

    spec = item.field
    index = a.FieldIndex(
        id=item.snapshot.id,
        field_snapshot_id=item.snapshot.id,
        population_id=population.manifest.id,
        logical_digest=manifest.logical_digest,
        full=item.snapshot.profile,
        source_manifest_digest=item.snapshot.manifest.blake3_digest,
    )
    kinds: list[ProfileKind] = []
    exclusive = spec.classification.transform == f.PROBABILITY_TRANSFORM_SOFTMAX
    if spec.HasField("classification"):
        if exclusive:
            index.projections.add(projection=p.FieldDistribution.TOP_CLASS)
            kinds.append("text")
        for label in spec.classification.classes:
            index.projections.add(
                projection=p.FieldDistribution.CLASS_PROBABILITY, class_name=label
            )
            kinds.append("number")
    elif spec.length:
        index.projections.add(projection=0)
        kinds.append("boolean")
    else:
        index.projections.add(projection=p.FieldDistribution.SCALAR)
        scalar_kinds: dict[int, ProfileKind] = {
            f.VALUE_FLOAT32: "number",
            f.VALUE_FLOAT64: "number",
            f.VALUE_INT64: "integer",
            f.VALUE_STRING: "text",
            f.VALUE_BOOL: "boolean",
        }
        kinds.append(scalar_kinds[spec.element_type])
    columns: list[list[Scalar | None]] = [[] for _ in kinds]
    first = count = 0

    def flush() -> None:
        nonlocal first
        for projection, values, kind in zip(index.projections, columns, kinds, strict=True):
            layout = next(
                (
                    distribution
                    for distribution in item.snapshot.profile.distributions
                    if distribution.projection == projection.projection
                    and distribution.class_name == projection.class_name
                ),
                None,
            )
            projection.blocks.append(
                save_column(service._storage, Column.build(values, kind, layout), first)
            )
        first += len(columns[0])
        for values in columns:
            values.clear()

    for row in read_rows(service, "field", manifest):
        if count >= population.manifest.documents:
            raise ValueError("field build must cover the entire query union")
        table = population.part(count // BLOCK_ROWS)
        if table["id"][count % BLOCK_ROWS].as_py() != row.document_id:
            raise ValueError("field build must cover the entire query union")
        outcome = decode_value(spec, row)
        if outcome is None:
            projected: list[Scalar | None] = [None] * len(columns)
        elif spec.HasField("classification"):
            assert isinstance(outcome, list)
            vector = numeric_vector(outcome)
            scores = probabilities(spec, vector)
            projected = ([top_class(spec, vector)[0]] if exclusive else []) + [
                scores[label] for label in spec.classification.classes
            ]
        elif spec.length:
            projected = [True]
        else:
            assert isinstance(outcome, (bool, int, float, str))
            projected = [outcome]
        for column, scalar in zip(columns, projected, strict=True):
            column.append(scalar)
        count += 1
        if len(columns[0]) == BLOCK_ROWS:
            flush()
    if count != population.manifest.documents:
        raise ValueError("field build must cover the entire query union")
    if columns[0] or not count:
        flush()
    service._storage.save("field", item.snapshot.id, index, suffix=FIELD_SUFFIX)
    return index


def prepared_field(
    service: Coordinator, population: Population, pin: bytes
) -> tuple[d.FieldBuild, a.FieldIndex]:
    from premixdb.runtime.enrichment import load_build

    item = service._storage.load("field", pin, d.FieldBuild)
    try:
        index = service._storage.load("field", pin, a.FieldIndex, suffix=FIELD_SUFFIX)
    except KeyError:
        item, manifest = load_build(service, "field", pin, population.manifest.snapshot_ids)
        return item, prepare_field(service, population, item, manifest)
    if (
        item.snapshot.id != pin
        or list(item.snapshot.snapshot_ids) != list(population.manifest.snapshot_ids)
        or index.id != pin
        or index.field_snapshot_id != pin
        or index.population_id != population.manifest.id
        or index.full != item.snapshot.profile
        or index.full.documents != population.manifest.documents
        or index.source_manifest_digest != item.snapshot.manifest.blake3_digest
    ):
        raise ValueError("analytical field identity or population mismatch")
    validate_index(index, population.manifest.documents)
    return item, index
