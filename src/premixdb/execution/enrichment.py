"""Single-host enrichment workers and verified, immutable build storage."""

from __future__ import annotations

import json
import math
import operator
from collections.abc import Mapping
from itertools import islice
from typing import (
    TYPE_CHECKING,
    Iterable,
    Iterator,
    Literal,
    Protocol,
    Sequence,
    TypedDict,
    cast,
    overload,
)

from google.protobuf.message import Message

from .._protobuf import at, parse
from .._typing import Edge, FieldValue, checked_record, field_value
from ..engine.curation import SelectedDocument
from ..engine.plans import Step
from ..enrichment import (
    DataTroveFields,
    DupekitIndex,
    Embeddings,
    LanguageScores,
    QuRating,
    WebOrganizer,
)
from ..enrichment.types import ComputedRow
from .materialization import SingleFlight
from .storage import ObjectStore

if TYPE_CHECKING:
    from .coordinator import Coordinator
    from .pipeline import PartitionPipeline
from urllib.parse import urlsplit

from blake3 import blake3

from .. import _runtime
from .._field_ids import field_name, selector_field
from ..engine import execution
from ..enrichment.classification import probabilities, top_class
from ..enrichment.types import Document
from ..internal import derivation_pb2 as e
from ..v1 import field_pb2 as f
from ..v1 import index_pb2 as ix
from ..v1 import query_pb2 as q
from ..v1 import storage_pb2 as storage
from .catalog import build_id as planned_build_id
from .catalog import identity, wire
from .profiles import FieldProfiler, predicate_bounds

type Worker = DataTroveFields | LanguageScores | DupekitIndex | QuRating | WebOrganizer | Embeddings


class FieldComputer(Protocol):
    def compute(self, documents: Sequence[Document], /) -> list[ComputedRow]: ...


class CacheService(Protocol):
    @property
    def _storage(self) -> ObjectStore: ...
    @property
    def _submissions(self) -> SingleFlight: ...
    @property
    def pipeline(self) -> PartitionPipeline | None: ...


class DedupeRow(TypedDict):
    id: str
    exact_hash: bytes
    minhash: list[int] | None
    lsh_buckets: list[int] | None


def require_field(schema: f.Field | ix.Index) -> f.Field:
    if not isinstance(schema, f.Field):
        raise TypeError("expected a field schema")
    return schema


def require_index(schema: f.Field | ix.Index) -> ix.Index:
    if not isinstance(schema, ix.Index):
        raise TypeError("expected an index schema")
    return schema


def row_id(row: Mapping[str, object]) -> str:
    id = row["id"]
    if not isinstance(id, str):
        raise ValueError("computed row must have a string id")
    return id


def numeric_vector(value: FieldValue) -> list[float]:
    if not isinstance(value, list) or any(not isinstance(x, (float, int)) for x in value):
        raise ValueError("expected a numeric vector")
    return [float(x) for x in value if isinstance(x, (float, int))]


SHARD_ROWS = 64


def hash_rows(digest: blake3, rows: Iterable[Message]) -> None:
    for row in rows:
        data = wire(row)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)


def producer(spec: e.EnrichmentProducer) -> Worker:
    kind = spec.WhichOneof("kind")
    if kind == "datatrove":
        from ..enrichment import DataTroveFields

        return DataTroveFields(language=spec.datatrove.language or "en")
    if kind == "language":
        from ..enrichment.language import LanguageScores

        return LanguageScores(spec.language.languages, model_digest=spec.language.model_digest)
    if kind == "dupekit":
        from ..enrichment import DupekitIndex

        p = spec.dupekit
        return DupekitIndex(p.num_perms, p.num_bands, p.ngram_size, p.seed)
    if kind == "model":
        from ..enrichment import Embeddings, ModelPin, QuRating, WebOrganizer

        p = spec.model
        pin = ModelPin(p.repository, p.revision)
        device, batch_size = p.device or "cpu", p.batch_size or 8
        if p.kind == e.ModelProducer.QUALITY:
            if p.max_length not in (0, 512):
                raise ValueError("QuRating window size is fixed at 512")
            return QuRating(pin, device=device, batch_size=batch_size)
        max_length = p.max_length or 8192
        if p.kind in (e.ModelProducer.TOPIC, e.ModelProducer.CONTENT_TYPE):
            from ..fields import ContentType, Topic

            return WebOrganizer(
                pin,
                task="topic" if p.kind == e.ModelProducer.TOPIC else "content_type",
                classes=tuple(Topic if p.kind == e.ModelProducer.TOPIC else ContentType),
                device=device,
                batch_size=batch_size,
                max_length=max_length,
            )
        if p.kind == e.ModelProducer.HARRIER:
            return Embeddings(
                pin,
                family="harrier",
                width=1024,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
            )
    raise ValueError("unsupported enrichment producer")


def encode_value(spec: f.Field, id: bytes, value: FieldValue) -> e.FieldValue:
    row = e.FieldValue(document_id=id)
    if value is None:
        row.null = True
    elif spec.length:
        if not isinstance(value, (tuple, list)) or len(value) != spec.length:
            raise ValueError("field vector width mismatch")
        vector = numeric_vector(value)
        if not all(math.isfinite(x) for x in vector):
            raise ValueError("field vector must contain finite numbers")
        row.vector.values.extend(vector)
    elif spec.element_type in (f.VALUE_FLOAT32, f.VALUE_FLOAT64):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("field requires finite numeric values")
        row.number = value
    elif spec.element_type == f.VALUE_INT64:
        if type(value) is not int or not -(2**63) <= value < 2**63:
            raise ValueError("field requires int64 values")
        row.integer = value
    elif spec.element_type == f.VALUE_STRING and isinstance(value, str):
        row.text = value
    elif spec.element_type == f.VALUE_BOOL and type(value) is bool:
        row.boolean = value
    else:
        raise ValueError("value does not match field schema")
    return row


def decode_value(spec: f.Field, row: e.FieldValue) -> FieldValue:
    kind = row.WhichOneof("value")
    if kind == "null":
        if not row.null:
            raise ValueError("invalid null marker")
        return None
    if kind is None:
        raise ValueError("missing field outcome")
    value: FieldValue = (
        list(row.vector.values)
        if kind == "vector"
        else row.number
        if kind == "number"
        else row.integer
        if kind == "integer"
        else row.text
        if kind == "text"
        else row.boolean
    )
    if encode_value(spec, row.document_id, value) != row:
        raise ValueError("stored value does not match field schema")
    return value


# Computation cohorts are semantic; storage shards may be repacked independently.
COHORT_ROWS = 64


def cached_rows(
    service: CacheService,
    request: e.DerivationPlan,
    worker: FieldComputer | DupekitIndex,
    documents: Iterable[Document],
    schemas: Sequence[f.Field] | Sequence[ix.Index],
    definition: bytes,
) -> Iterator[e.DedupeEvidence | list[e.FieldValue]]:
    """Reuse independent document maps or exact cohorts for batch-sensitive models."""
    is_index = request.producer.HasField("dupekit")
    scope = getattr(worker, "cache_scope", "batch")
    if scope not in ("document", "batch"):
        raise ValueError("unknown derivation cache scope")
    schema_ids = [spec.id for spec in schemas]
    recipe = identity(
        "premixdb-computation/v1",
        request.git_commit,
        request.catalog_version.to_bytes(4, "big"),
        request.runtime_digest,
        wire(request.producer),
        definition,
        scope.encode(),
        *schema_ids,
    )

    def key(docs: Sequence[Document]) -> bytes:
        return identity(
            "premixdb-computation-inputs/v1", recipe, *(bytes.fromhex(doc.id) for doc in docs)
        )

    def load(docs: Sequence[Document]) -> e.DerivationCache:
        cache_id = key(docs)
        result = service._storage.load("derivation", cache_id, e.DerivationCache, suffix=".cache")
        if (
            result.id != cache_id
            or result.definition_json != definition
            or list(result.schema_ids) != schema_ids
            or list(result.document_ids) != [bytes.fromhex(doc.id) for doc in docs]
        ):
            raise ValueError("cached derivation belongs to different inputs or producer")
        if is_index:
            if result.fields or [row.document_id for row in result.evidence.rows] != list(
                result.document_ids
            ):
                raise ValueError("incomplete cached evidence coverage")
            for doc, row in zip(docs, result.evidence.rows):
                if (
                    len(row.exact_hash) != 32
                    or row.exact_hash != blake3(doc.text.encode()).digest()
                ):
                    raise ValueError("cached evidence does not match its document")
        else:
            if result.HasField("evidence") or len(result.fields) != len(schemas):
                raise ValueError("cached derivation schema mismatch")
            for spec, column in zip(schemas, result.fields):
                if [row.document_id for row in column.rows] != list(result.document_ids):
                    raise ValueError("incomplete cached field coverage")
                for row in column.rows:
                    decode_value(require_field(spec), row)
        return result

    def compute(
        docs: Sequence[Document], values: list[ComputedRow] | list[DedupeRow] | None = None
    ) -> e.DerivationCache:
        if values is None:
            if isinstance(worker, DupekitIndex):
                values = [
                    checked_record(row, DedupeRow) for row in worker.compute(docs).to_pylist()
                ]
            else:
                values = worker.compute(docs)
        if [row["id"] for row in values] != [doc.id for doc in docs]:
            raise ValueError("producer changed document coverage")
        result = e.DerivationCache(
            id=key(docs),
            definition_json=definition,
            document_ids=[bytes.fromhex(doc.id) for doc in docs],
            schema_ids=schema_ids,
        )
        if is_index:
            result.evidence.rows.extend(
                e.DedupeEvidence(
                    document_id=bytes.fromhex(row_id(row)),
                    exact_hash=checked_record(row, DedupeRow)["exact_hash"],
                    minhash=checked_record(row, DedupeRow)["minhash"] or (),
                    lsh_buckets=checked_record(row, DedupeRow)["lsh_buckets"] or (),
                )
                for row in values
            )
            if any(
                row.exact_hash != blake3(doc.text.encode()).digest()
                for doc, row in zip(docs, result.evidence.rows)
            ):
                raise ValueError("producer returned evidence for different text")
        else:
            for spec in schemas:
                result.fields.add(
                    rows=[
                        encode_value(
                            require_field(spec),
                            bytes.fromhex(row_id(row)),
                            field_value(cast(Mapping[str, object], row)[spec.name]),
                        )
                        for row in values
                    ]
                )
        return result

    def save(result: e.DerivationCache) -> None:
        service._storage.save("derivation", result.id, result, suffix=".cache")

    def batch(docs: Sequence[Document]) -> e.DerivationCache:
        try:
            return load(docs)
        except KeyError:
            result = compute(docs)
            save(result)
            return result

    iterator = iter(documents)
    if service.pipeline is not None:
        # Bound outstanding text and preserve the fixed semantic cohort policy.
        while window := list(islice(iterator, COHORT_ROWS * 8)):
            cohorts = (
                [window[i : i + COHORT_ROWS] for i in range(0, len(window), COHORT_ROWS)]
                if scope == "batch"
                else [[doc] for doc in window]
            )
            available: dict[bytes, e.DerivationCache] = {}
            missing: list[Sequence[Document]] = []
            for cohort in cohorts:
                try:
                    available[key(cohort)] = load(cohort)
                except KeyError:
                    missing.append(cohort)
            for cohort, values in zip(
                missing,
                service.pipeline.features(request.producer, definition, missing),
                strict=True,
            ):
                result = compute(cohort, values)
                save(result)
                available[key(cohort)] = result
            for cohort in cohorts:
                result = available[key(cohort)]
                for i in range(len(cohort)):
                    yield (
                        at(result.evidence.rows, i)
                        if is_index
                        else [at(column.rows, i) for column in result.fields]
                    )
        return
    while docs := list(islice(iterator, COHORT_ROWS)):
        if scope == "batch":
            result = service._submissions.run(("derivation-cache", key(docs)), lambda: batch(docs))
            results = [(result, i) for i in range(len(docs))]
        else:
            results, missing = {}, []
            for doc in docs:
                try:
                    results[doc.id] = load([doc])
                except KeyError:
                    missing.append(doc)
            if missing:
                computed = compute(missing)
                for i, doc in enumerate(missing):
                    result = e.DerivationCache(
                        id=key([doc]),
                        definition_json=definition,
                        document_ids=[bytes.fromhex(doc.id)],
                        schema_ids=schema_ids,
                    )
                    if is_index:
                        result.evidence.rows.append(computed.evidence.rows[i])
                    else:
                        for column in computed.fields:
                            result.fields.add(rows=[at(column.rows, i)])
                    save(result)
                    results[doc.id] = result
            results = [(results[doc.id], 0) for doc in docs]
        for result, i in results:
            yield (
                at(result.evidence.rows, i)
                if is_index
                else [at(column.rows, i) for column in result.fields]
            )


def build(service: Coordinator, request: e.DerivationPlan) -> e.Materialization:
    """Materialize one planner-owned recipe, once for its immutable population."""
    try:
        return service._storage.load("derivation", request.id, e.Materialization)
    except KeyError:
        pass
    code = _runtime.resolve_code(request.git_commit)
    handles = [service._snapshot(id) for id in request.snapshot_ids]
    population = execution.execute(handles, [], code)
    result = e.Materialization(plan=request)
    names = set()
    for policy in (request.producer,):
        worker = producer(policy)
        definition = json.dumps(
            worker.definition, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        schemas = worker.indexes if isinstance(worker, DupekitIndex) else worker.fields
        if any(spec.name in names for spec in schemas):
            raise ValueError("duplicate enrichment field/index names")
        names.update(spec.name for spec in schemas)
        manifests = [
            e.EnrichmentManifest(
                producer=policy,
                definition_json=definition,
                documents=population.row_count,
                plan=request,
            )
            for _ in schemas
        ]
        logical_digests = [blake3() for _ in schemas]
        profilers = (
            []
            if policy.HasField("dupekit")
            else [FieldProfiler(require_field(spec)) for spec in schemas]
        )

        def documents() -> Iterator[Document]:
            for row in population.rows():
                url = (
                    row.source_key if urlsplit(row.source_key).scheme in ("http", "https") else None
                )
                yield Document(row.id, row.text, url)

        outcomes = iter(cached_rows(service, request, worker, documents(), schemas, definition))
        while rows := list(islice(outcomes, SHARD_ROWS)):
            if policy.HasField("dupekit"):
                shard = e.DedupeEvidenceShard(
                    rows=[r for r in rows if isinstance(r, e.DedupeEvidence)]
                )
                if len(shard.rows) != len(rows):
                    raise ValueError("index producer returned field outcomes")
                ref = service._storage.put("index", wire(shard))
                for manifest, digest in zip(manifests, logical_digests):
                    manifest.shards.append(ref)
                    hash_rows(digest, shard.rows)
            else:
                for column, (spec, manifest, digest, profiler) in enumerate(
                    zip(schemas, manifests, logical_digests, profilers)
                ):
                    shard = e.FieldValueShard(
                        rows=[row[column] for row in rows if isinstance(row, list)]
                    )
                    if len(shard.rows) != len(rows):
                        raise ValueError("field producer returned index outcomes")
                    shard_profiler = FieldProfiler(require_field(spec))
                    for row in shard.rows:
                        value = decode_value(require_field(spec), row)
                        profiler.add(value)
                        shard_profiler.add(value)
                    data = wire(shard)
                    manifest.shards.append(
                        service._storage.put(
                            "field",
                            data,
                            profile=storage.ObjectProfile(
                                content_bytes=len(data), fields=[shard_profiler.proto()]
                            ),
                        )
                    )
                    hash_rows(digest, shard.rows)
        for i, (spec, manifest, digest) in enumerate(zip(schemas, manifests, logical_digests)):
            manifest.logical_digest = digest.digest()
            is_index = policy.HasField("dupekit")
            prefix = "index" if is_index else "field"
            ref = service._storage.put(prefix, wire(manifest))
            build_id = planned_build_id(request, spec.name)
            if is_index:
                item = e.IndexBuild(
                    index=require_index(spec),
                    snapshot=ix.IndexSnapshot(
                        id=build_id,
                        index_id=spec.id,
                        snapshot_id=request.snapshot_ids[0]
                        if len(request.snapshot_ids) == 1
                        else b"",
                        snapshot_ids=request.snapshot_ids,
                        manifest=ref,
                        git_commit=request.git_commit,
                    ),
                )
            else:
                item = e.FieldBuild(
                    field=require_field(spec),
                    snapshot=f.FieldSnapshot(
                        id=build_id,
                        field_id=spec.id,
                        snapshot_id=request.snapshot_ids[0]
                        if len(request.snapshot_ids) == 1
                        else b"",
                        snapshot_ids=request.snapshot_ids,
                        manifest=ref,
                        git_commit=request.git_commit,
                        profile=profilers[i].proto(),
                    ),
                )
            try:
                previous, _ = load_build(service, prefix, build_id, request.snapshot_ids)
            except KeyError:
                service._storage.save(prefix, build_id, item)
            else:
                item = previous
            if isinstance(item, e.IndexBuild):
                result.indexes.append(item)
            else:
                result.fields.append(item)
    service._storage.save("derivation", request.id, result)
    return result


@overload
def load_build(
    service: Coordinator, prefix: Literal["field"], id: bytes, population: Iterable[bytes]
) -> tuple[e.FieldBuild, e.EnrichmentManifest]: ...
@overload
def load_build(
    service: Coordinator, prefix: Literal["index"], id: bytes, population: Iterable[bytes]
) -> tuple[e.IndexBuild, e.EnrichmentManifest]: ...
@overload
def load_build(
    service: Coordinator, prefix: Literal["field", "index"], id: bytes, population: Iterable[bytes]
) -> tuple[e.FieldBuild | e.IndexBuild, e.EnrichmentManifest]: ...
def load_build(
    service: Coordinator, prefix: Literal["field", "index"], id: bytes, population: Iterable[bytes]
) -> tuple[e.FieldBuild | e.IndexBuild, e.EnrichmentManifest]:
    if len(id) != 32:
        raise ValueError("build IDs must be 32 bytes")
    item = (
        service._storage.load(prefix, id, e.FieldBuild)
        if prefix == "field"
        else service._storage.load(prefix, id, e.IndexBuild)
    )
    snapshot = item.snapshot
    if snapshot.id != id or list(snapshot.snapshot_ids) != sorted(set(population)):
        raise ValueError("enrichment build belongs to a different query population")
    raw = service._storage.read_object(prefix, snapshot.manifest)
    manifest = parse(e.EnrichmentManifest, raw)
    schema = item.field if isinstance(item, e.FieldBuild) else item.index
    planned = parse(e.DerivationPlan, wire(manifest.plan))
    claimed = planned.id
    planned.ClearField("id")
    if identity("premixdb-derivation/v1", wire(planned)) != claimed:
        raise ValueError("derivation recipe identity mismatch")
    if manifest.producer != planned.producer:
        raise ValueError("artifact producer does not match its derivation recipe")
    if (
        list(planned.snapshot_ids) != list(snapshot.snapshot_ids)
        or planned.git_commit != snapshot.git_commit
    ):
        raise ValueError("derivation population/revision mismatch")
    expected = planned_build_id(manifest.plan, schema.name)
    if expected != id:
        raise ValueError("enrichment build identity mismatch")
    return item, manifest


@overload
def read_rows(
    service: Coordinator,
    prefix: Literal["field"],
    manifest: e.EnrichmentManifest,
    *,
    selector: None = None,
) -> Iterator[e.FieldValue]: ...
@overload
def read_rows(
    service: Coordinator,
    prefix: Literal["field"],
    manifest: e.EnrichmentManifest,
    *,
    selector: q.FieldComparison,
) -> Iterator[tuple[e.FieldValue, bool | None]]: ...
@overload
def read_rows(
    service: Coordinator,
    prefix: Literal["index"],
    manifest: e.EnrichmentManifest,
    *,
    selector: None = None,
) -> Iterator[e.DedupeEvidence]: ...
def read_rows(
    service: Coordinator,
    prefix: Literal["field", "index"],
    manifest: e.EnrichmentManifest,
    *,
    selector: q.FieldComparison | None = None,
) -> Iterator[e.FieldValue | e.DedupeEvidence | tuple[e.FieldValue, bool | None]]:
    previous = b""
    count = 0
    digest = blake3()
    message = e.FieldValueShard if prefix == "field" else e.DedupeEvidenceShard
    for ref in manifest.shards:
        decision = None
        if selector is not None:
            profile = next((p for p in ref.profile.fields if p.field == selector.field), None)
            bounds = predicate_bounds(profile, selector) if profile is not None else None
            if bounds is not None and profile is not None:
                lower, upper = bounds
                if upper == 0:
                    decision = False
                elif lower == profile.documents:
                    decision = True
        shard = message()
        shard.ParseFromString(service._storage.read_object(prefix, ref))
        for row in shard.rows:
            if len(row.document_id) != 32 or row.document_id <= previous:
                raise ValueError("duplicate, unordered or malformed enrichment row")
            previous = row.document_id
            count += 1
            hash_rows(digest, (row,))
            if selector is None:
                yield row
            else:
                if not isinstance(row, e.FieldValue):
                    raise ValueError("index evidence cannot be filtered by a field selector")
                yield row, decision
    if count != manifest.documents:
        raise ValueError("incomplete enrichment coverage")
    if digest.digest() != manifest.logical_digest:
        raise ValueError("enrichment logical row digest mismatch")


def matches(spec: f.Field, selector: q.FieldComparison, value: FieldValue) -> bool:
    projection = selector.projection
    if projection == q.FieldComparison.IS_NULL:
        value = value is None
    elif value is None:
        return False  # All ordinary comparisons, including !=, exclude computed nulls.
    elif projection == q.FieldComparison.CLASS_PROBABILITY:
        value = probabilities(spec, numeric_vector(value))[selector.class_name]
    elif projection == q.FieldComparison.TOP_CLASS:
        value = top_class(spec, numeric_vector(value))[0]
    elif projection == q.FieldComparison.VECTOR_COMPONENT:
        value = numeric_vector(value)[selector.component]
    expected = getattr(selector, selector.WhichOneof("value"))
    return bool(
        {
            q.Comparison.OPERATOR_EQ: operator.eq,
            q.Comparison.OPERATOR_NE: operator.ne,
            q.Comparison.OPERATOR_LT: operator.lt,
            q.Comparison.OPERATOR_LE: operator.le,
            q.Comparison.OPERATOR_GT: operator.gt,
            q.Comparison.OPERATOR_GE: operator.ge,
        }[selector.operator](value, expected)
    )


def validate_selector(spec: f.Field, selector: q.FieldComparison) -> None:
    if field_name(selector_field(selector)) != spec.name:
        raise ValueError("field name does not match pinned field build")
    projection = selector.projection
    kind = selector.WhichOneof("value")
    expected = None
    if projection == q.FieldComparison.SCALAR:
        if spec.length:
            raise ValueError("vector fields require a component or classification projection")
        expected = {
            f.VALUE_FLOAT32: ("number", "integer"),
            f.VALUE_FLOAT64: ("number", "integer"),
            f.VALUE_INT64: ("integer",),
            f.VALUE_BOOL: ("boolean",),
            f.VALUE_STRING: ("text",),
        }.get(spec.element_type)
    elif projection == q.FieldComparison.IS_NULL:
        expected = ("boolean",)
    elif projection in (q.FieldComparison.CLASS_PROBABILITY, q.FieldComparison.TOP_CLASS):
        if not spec.HasField("classification"):
            raise ValueError("field is not a classifier")
        if projection == q.FieldComparison.TOP_CLASS:
            top_class(spec, [0.0] * spec.length)
            expected = ("text",)
            if selector.text not in spec.classification.classes:
                raise ValueError("unknown class label")
        else:
            if selector.class_name not in probabilities(spec, [0.0] * spec.length):
                raise ValueError("unknown class label")
            expected = ("number", "integer")
    elif projection == q.FieldComparison.VECTOR_COMPONENT:
        if not selector.HasField("component") or selector.component >= spec.length:
            raise ValueError("vector component is outside field width")
        expected = ("number", "integer")
    if expected is None or kind not in expected:
        raise ValueError("comparison value does not match field projection")
    if selector.HasField("component") != (projection == q.FieldComparison.VECTOR_COMPONENT):
        raise ValueError("component is only valid for vector component comparisons")
    if bool(selector.class_name) != (projection == q.FieldComparison.CLASS_PROBABILITY):
        raise ValueError("class_name is only valid for class probabilities")
    if kind in ("boolean", "text") and selector.operator not in (
        q.Comparison.OPERATOR_EQ,
        q.Comparison.OPERATOR_NE,
    ):
        raise ValueError("boolean and label fields support only equality comparisons")
    if kind == "number" and not math.isfinite(selector.number):
        raise ValueError("comparison value must be finite")


def query_inputs(service: Coordinator, query: q.Query) -> tuple[execution.CorpusIndex, list[Step]]:
    """Derive missing recipes, then verify complete coverage before selecting rows."""
    from .catalog import resolve

    for plan in resolve(query):
        service._once(plan, lambda plan=plan: build(service, plan))
    from ..engine import plans
    from .planner import execution_steps, external_definition

    handles = [service._snapshot(id) for id in query.snapshot_ids]
    union = execution.execute(handles, [], _runtime.resolve_code(query.git_commit))
    ids = {bytes.fromhex(row[0]) for row in union.metadata()}
    selections, document_hashes = {}, None
    from ..engine import curation
    from ..engine.value_cache import ValueCache
    from .catalog import selectors

    field_values: dict[str, Mapping[str, FieldValue]] = {}
    for selector in selectors(query):
        key = curation.selector_key(selector)
        if key in field_values:
            continue
        if selector.field in (1, 2, 3, 4):
            field_values[key] = ValueCache(
                (
                    (
                        row.id,
                        row.document.size
                        if selector.field == 1
                        else row.document.characters
                        if selector.field == 2
                        else row.source_key
                        if selector.field == 3
                        else row.corpus_id,
                    )
                    for row in union.rows()
                )
            )
            continue
        item, manifest = load_build(
            service, "field", selector.field_snapshot_id, query.snapshot_ids
        )
        covered, projected = set(), ValueCache()
        for row in read_rows(service, "field", manifest):
            value = decode_value(item.field, row)
            covered.add(row.document_id)
            projected[row.document_id.hex()] = project(item.field, selector, value)
        if covered != ids:
            raise ValueError("field build must cover the entire query union")
        field_values[key] = projected
    for operation in query.operations:
        if operation.HasField("field_where"):
            selector = operation.field_where
            item, manifest = load_build(
                service, "field", selector.field_snapshot_id, query.snapshot_ids
            )
            validate_selector(item.field, selector)
            covered, members = set(), []
            for row, decision in read_rows(service, "field", manifest, selector=selector):
                value = decode_value(item.field, row)
                covered.add(row.document_id)
                if decision is True or (decision is None and matches(item.field, selector, value)):
                    members.append(row.document_id)
            if covered != ids:
                raise ValueError("field build must cover the entire query union")
            key = external_definition(operation)
            selections[key] = plans.external_filter(key, members)
        elif operation.HasField("indexed_dedupe"):
            item, manifest = load_build(
                service, "index", operation.indexed_dedupe.index_snapshot_id, query.snapshot_ids
            )
            if item.index.name not in ("dupekit.exact_candidates", "dupekit.lsh"):
                raise NotImplementedError(
                    "LSH candidates require an explicit verification/clustering policy"
                )
            rows = list(read_rows(service, "index", manifest))
            if {row.document_id for row in rows} != ids or any(
                len(row.exact_hash) != 32 for row in rows
            ):
                raise ValueError(
                    "index must contain complete binary-hash evidence for the query union"
                )
            hashes = [(row.document_id.hex(), row.exact_hash.hex()) for row in rows]
            if document_hashes is not None and hashes != document_hashes:
                raise ValueError("conflicting exact evidence indexes")
            document_hashes = hashes
    steps = execution_steps(query)
    for i, operation in enumerate(query.operations):
        if operation.HasField("field_where"):
            steps[i] = selections[external_definition(operation)]
    index = execution.CorpusIndex(handles, document_hashes)
    index.field_values = field_values
    for i, operation in enumerate(query.operations):
        if operation.HasField("similarity_dedupe"):
            policy = operation.similarity_dedupe
            if policy.algorithm == 1:

                def edges(
                    documents: list[SelectedDocument], p: q.SimilarityDedupe = policy
                ) -> Iterable[Edge]:
                    return curation.jaccard_edges(documents, p.n, p.threshold)
            else:
                vectors = field_values[curation.selector_key(policy.embedding)]

                def edges(
                    documents: list[SelectedDocument],
                    vectors: Mapping[str, FieldValue] = vectors,
                    threshold: float = policy.threshold,
                ) -> Iterable[Edge]:
                    selected = {doc.id for doc in documents}
                    return curation.cosine_edges(
                        {
                            id: numeric_vector(vector) if vector is not None else None
                            for id, vector in vectors.items()
                        },
                        threshold,
                        selected=selected,
                    )

            payload = steps[i].payload
            assert payload is not None and payload[0] == "similarity"
            steps[i] = plans.policy(steps[i].definition, ("similarity", payload[1], edges))
        elif (
            operation.HasField("indexed_dedupe")
            and operation.indexed_dedupe.index_name == "dupekit.lsh"
        ):
            policy = operation.indexed_dedupe
            _, manifest = load_build(service, "index", policy.index_snapshot_id, query.snapshot_ids)

            def edges(
                documents: list[SelectedDocument],
                p: q.IndexedDedupe = policy,
                manifest: e.EnrichmentManifest = manifest,
            ) -> Iterable[Edge]:
                from ..engine.spill import candidate_pairs

                pairs = candidate_pairs(
                    (row.document_id.hex(), row.lsh_buckets)
                    for row in read_rows(service, "index", manifest)
                )
                return curation.jaccard_edges(documents, 5, p.threshold, pairs)

            payload = steps[i].payload
            assert payload is not None and payload[0] == "similarity"
            steps[i] = plans.policy(steps[i].definition, ("similarity", payload[1], edges))
    return index, steps


def project(spec: f.Field, selector: q.FieldComparison, value: FieldValue) -> FieldValue:
    if selector.projection == q.FieldComparison.IS_NULL:
        return value is None
    if value is None:
        return None
    if selector.projection == q.FieldComparison.CLASS_PROBABILITY:
        return probabilities(spec, numeric_vector(value))[selector.class_name]
    if selector.projection == q.FieldComparison.TOP_CLASS:
        return top_class(spec, numeric_vector(value))[0]
    if selector.projection == q.FieldComparison.VECTOR_COMPONENT:
        if not selector.HasField("component") or selector.component >= spec.length:
            raise ValueError("vector component is outside field width")
        return numeric_vector(value)[selector.component]
    if selector.projection != q.FieldComparison.SCALAR:
        raise ValueError("invalid value projection")
    return value


def projections(
    service: Coordinator, query: execution.Query, selectors: Iterable[q.FieldComparison]
) -> dict[str, Mapping[str, FieldValue]]:
    from ..engine.curation import selector_key
    from .catalog import build_id, plan

    result = {}
    rows = query.rows()
    for selector in selectors:
        if selector.field in (1, 2, 3, 4):
            result[selector_key(selector)] = {
                r.id: r.document.size
                if selector.field == 1
                else len(r.text)
                if selector.field == 2
                else r.source_key
                if selector.field == 3
                else r.corpus_id
                for r in rows
            }
            continue
        name = field_name(selector.field)
        snapshots = [bytes.fromhex(id) for id in query.inputs]
        definition = plan(snapshots, name, bytes.fromhex(query.code.commit))
        pin = build_id(definition, name)
        if selector.field_snapshot_id and selector.field_snapshot_id != pin:
            raise ValueError("stratum field pin does not match its built-in recipe")
        # A mixture can request a field after selection. Reuse its frozen input
        # snapshots and execution revision, without changing the parent query.
        service._once(definition, lambda definition=definition: build(service, definition))
        item, manifest = load_build(service, "field", pin, snapshots)
        values = {
            r.document_id.hex(): project(item.field, selector, decode_value(item.field, r))
            for r in read_rows(service, "field", manifest)
        }
        if not {r.id for r in rows} <= values.keys():
            raise ValueError("incomplete stratum field coverage")
        result[selector_key(selector)] = values
    return result
