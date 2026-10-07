"""Verified, immutable analytical columns and shared document descriptors."""

from __future__ import annotations

import weakref
from bisect import bisect_right
from collections.abc import Iterable, Iterator, MutableMapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, cast

import numpy as np
import pyarrow as pa
from blake3 import blake3
from pyroaring import BitMap

from premixdb.contracts import json_object, load_json
from premixdb.engine.analytics import BLOCK_ROWS, Column, Moments, Numbers, Scalar
from premixdb.engine.identity import identity
from premixdb.engine.queries import Row
from premixdb.engine.records import decode_document, validate_document
from premixdb.engine.snapshots import StoredDocument
from premixdb.internal import analytics_pb2 as a
from premixdb.internal import derivation_pb2 as d
from premixdb.schemas.protobuf import parse, wire
from premixdb.storage.objects import ObjectStore
from premixdb.storage.profiles import endpoint, value
from premixdb.storage.publication import publish
from premixdb.storage.selections import _Frames
from premixdb.v1 import profile_pb2 as p
from premixdb.v1 import query_pb2 as q
from premixdb.v1 import storage_pb2 as s

if TYPE_CHECKING:
    from premixdb.engine.indexed import IndexedQuery
    from premixdb.storage.catalog import Catalog

POPULATION_SUFFIX = ".population-index-v1"
FIELD_SUFFIX = ".field-index-v1"
SELECTION_SUFFIX = ".indexed-selection-v1"


def population_id(snapshot_ids: Iterable[bytes]) -> bytes:
    return identity("premixdb-analytical-population/v1", *sorted(set(snapshot_ids)))


def put_table(store: ObjectStore, table: pa.Table) -> s.ObjectRef:
    """Publish large Arrow files without making another complete byte copy."""
    with NamedTemporaryFile(suffix=".arrow") as temporary:
        with pa.OSFile(temporary.name, "wb") as output:
            with pa.ipc.new_file(output, table.schema) as writer:
                writer.write_table(table)
        digest, size = blake3(), 0
        with Path(temporary.name).open("rb") as stream:
            while data := stream.read(1024 * 1024):
                digest.update(data)
                size += len(data)
        relative = f"index/objects/{digest.hexdigest()}"

        def chunks() -> Iterator[bytes]:
            with Path(temporary.name).open("rb") as stream:
                while data := stream.read(1024 * 1024):
                    yield data

        publish(store.root / relative, chunks())
        return s.ObjectRef(
            blake3_digest=digest.digest(), size_bytes=size, uri=store.object_uri(relative)
        )


def read_table(store: ObjectStore, ref: s.ObjectRef) -> pa.Table:
    if len(ref.blake3_digest) != 32:
        raise ValueError("invalid analytical object reference")
    path = store.root / "index/objects" / ref.blake3_digest.hex()
    try:
        with pa.memory_map(str(path), "r") as source:
            if source.size() != ref.size_bytes:
                raise ValueError("analytical object integrity check failed: size mismatch")
            buffer = source.read_buffer()
            if blake3(buffer).digest() != ref.blake3_digest:
                raise ValueError("analytical object integrity check failed")
            return pa.ipc.open_file(buffer).read_all()
    except FileNotFoundError:
        raise ValueError("analytical object is missing") from None


def save_column(store: ObjectStore, column: Column, first: int) -> a.ColumnBlock:
    numeric = column.population_moments
    return a.ColumnBlock(
        first=first,
        documents=len(column.values),
        kind=column.kind,
        values=put_table(
            store,
            pa.table(
                {
                    "value": column.values,
                    "rank": column.ranks,
                    "bucket": column.bucket_codes,
                }
            ),
        ),
        dictionary=put_table(store, pa.table({"value": column.dictionary})),
        bitmaps=store.put("index", wire(column.bitmap_proto())),
        full=column.full,
        exact_origin=endpoint(
            column.kind if column.kind in ("count", "integer", "number") else "integer",
            column.origin,
        ),
        mean_offset=numeric.mean if numeric else 0,
        squared_deviations=numeric.squared_deviations if numeric else 0,
        bucket_bounds=[
            p.ProfileBucket(lower=endpoint(column.kind, low), upper=endpoint(column.kind, high))
            for low, high in column.bounds
        ],
    )


def load_column(catalog: Catalog, block: a.ColumnBlock) -> Column:
    cache: MutableMapping[bytes, Column] = catalog._cache.namespace("analytical_columns")
    key = identity("premixdb-analytical-column/v1", wire(block))
    cached = cache.get(key)
    if cached is not None:
        return cached
    arrays = read_table(catalog._storage, block.values)
    dictionary = read_table(catalog._storage, block.dictionary)
    if arrays.column_names != ["value", "rank", "bucket"] or dictionary.column_names != ["value"]:
        raise ValueError("invalid analytical column schema")
    values = arrays["value"].combine_chunks().to_numpy(zero_copy_only=True)
    ranks = arrays["rank"].combine_chunks().to_numpy(zero_copy_only=True)
    codes = arrays["bucket"].combine_chunks().to_numpy(zero_copy_only=True)
    distinct = dictionary["value"].combine_chunks()
    if (
        len(values) != block.documents
        or ranks.dtype != np.uint32
        or codes.dtype != np.uint32
        or values.dtype not in (np.dtype("int64"), np.dtype("uint64"), np.dtype("float64"))
        or block.kind not in ("count", "integer", "number", "boolean", "text")
    ):
        raise ValueError("invalid analytical column coverage or type")
    bits = parse(a.BitmapBlock, catalog._storage.read_object("index", block.bitmaps))
    present = BitMap.deserialize(bits.present)
    planes = [BitMap.deserialize(data) for data in bits.planes]
    buckets = [BitMap.deserialize(data) for data in bits.buckets]
    if (
        any(b and b.max() >= block.documents for b in [present, *planes, *buckets])
        or len(planes) != max(0, len(distinct) - 1).bit_length()
        or len(buckets) != len(block.bucket_bounds)
        or len(buckets) > 64
        or sum(len(b) for b in buckets) != len(present)
    ):
        raise ValueError("invalid analytical bitmap coverage")
    numeric = None
    if block.full.HasField("numeric"):
        numeric = Moments(
            block.full.numeric.documents,
            block.full.numeric.total,
            block.mean_offset,
            block.squared_deviations,
        )
    origin = value(block.exact_origin)
    assert isinstance(origin, (int, float))
    result = Column(
        block.kind,
        cast(Numbers, values),
        distinct,
        ranks,
        codes,
        present,
        planes,
        buckets,
        full=block.full,
        origin=origin,
        population_moments=numeric,
        bounds=[(value(b.lower), value(b.upper)) for b in block.bucket_bounds],
    )
    cache[key] = result
    return result


class Population:
    def __init__(self, catalog: Catalog, manifest: a.PopulationIndex) -> None:
        self._owner, self.manifest = weakref.ref(catalog), manifest
        # Iterators own their current block. Weak references let provenance reuse
        # that block without retaining memory outside the iterator or shared LRU.
        self._active_parts: weakref.WeakValueDictionary[int, pa.Table] = (
            weakref.WeakValueDictionary()
        )
        self.starts = list(range(0, manifest.documents, BLOCK_ROWS))
        if len(self.starts) != len(manifest.columns):
            raise ValueError("incomplete analytical population")
        for indexed in manifest.intrinsic:
            validate_index(indexed, manifest.documents)

    @property
    def catalog(self) -> Catalog:
        catalog = self._owner()
        if catalog is None:
            raise ValueError("analytical catalog is closed")
        return catalog

    def part(self, index: int) -> pa.Table:
        active = self._active_parts.get(index)
        if active is not None:
            return active
        cache: MutableMapping[bytes, pa.Table] = self.catalog._cache.namespace(
            "analytical_registry"
        )
        ref = self.manifest.columns[index]
        table = cache.get(ref.blake3_digest)
        if table is None:
            table = read_table(self.catalog._storage, ref)
            expected = min(BLOCK_ROWS, self.manifest.documents - self.starts[index])
            if len(table) != expected:
                raise ValueError("incomplete analytical registry")
            cache[ref.blake3_digest] = table
        self._active_parts[index] = table
        return table

    def row(self, ordinal: int, output_ordinal: int) -> Row:
        if not 0 <= ordinal < self.manifest.documents:
            raise IndexError("document ordinal out of range")
        part = bisect_right(self.starts, ordinal) - 1
        table, local = self.part(part), ordinal - self.starts[part]
        chunk, position = int(table["chunk"][local].as_py()), int(table["position"][local].as_py())
        return self._row(table, local, position, self._descriptor(chunk), output_ordinal)

    def rows(self, ordinals: Iterable[int]) -> Iterator[Row]:
        """Retain only the current registry block and descriptor while streaming."""
        current_part = current_chunk = -1
        table, shard = None, None
        for output_ordinal, ordinal in enumerate(ordinals):
            if not 0 <= ordinal < self.manifest.documents:
                raise IndexError("document ordinal out of range")
            part = bisect_right(self.starts, ordinal) - 1
            if part != current_part:
                table = self.part(part)
                current_part = part
            assert table is not None
            local = ordinal - self.starts[part]
            chunk = int(table["chunk"][local].as_py())
            if chunk != current_chunk:
                shard = self._descriptor(chunk)
                current_chunk = chunk
            assert shard is not None
            position = int(table["position"][local].as_py())
            yield self._row(table, local, position, shard, output_ordinal)

    def _descriptor(self, chunk: int) -> d.SelectionShard:
        if not 0 <= chunk < len(self.manifest.descriptors):
            raise ValueError("invalid analytical document descriptor")
        ref = self.manifest.descriptors[chunk]
        cache: MutableMapping[bytes, d.SelectionShard] = self.catalog._cache.namespace(
            "analytical_descriptors"
        )
        shard = cache.get(ref.blake3_digest)
        if shard is None:
            try:
                shard = parse(d.SelectionShard, self.catalog._storage.read_object("index", ref))
            except KeyError:
                raise ValueError("stored selection shard is missing") from None
            cache[ref.blake3_digest] = shard
        return shard

    def _row(
        self,
        table: pa.Table,
        local: int,
        position: int,
        shard: d.SelectionShard,
        output_ordinal: int,
    ) -> Row:
        if not 0 <= position < len(shard.rows):
            raise ValueError("invalid analytical descriptor position")
        record = shard.rows[position]
        identity = table["id"][local].as_py()
        if record.document_id != identity or record.transformed:
            raise ValueError("analytical descriptor identity mismatch")
        source = decode_document(json_object(load_json(record.source_record)))
        validate_document(source, require_profiles=True)
        document = StoredDocument(
            record.corpus_id.hex(),
            source["key"],
            source,
            _Frames(self.catalog._storage),
            record.document_id.hex(),
        )
        return Row(output_ordinal, document)

    def origins(self, ordinal: int) -> list[str]:
        part = bisect_right(self.starts, ordinal) - 1
        raw = self.part(part)["origins"][ordinal - self.starts[part]].as_py()
        if not isinstance(raw, list) or any(not isinstance(item, bytes) for item in raw):
            raise ValueError("invalid analytical origins")
        return [item.hex() for item in raw]

    def intrinsic(self, field: q.IntrinsicField) -> a.FieldIndex:
        for item in self.manifest.intrinsic:
            if item.full.field == field:
                return item
        raise ValueError("unindexed intrinsic field")


def load_population(catalog: Catalog, identity: bytes) -> Population:
    manifest = catalog._storage.load("index", identity, a.PopulationIndex, suffix=POPULATION_SUFFIX)
    if (
        manifest.id != identity
        or population_id(manifest.snapshot_ids) != identity
        or manifest.documents > 2**32
    ):
        raise ValueError("analytical population identity mismatch")
    return Population(catalog, manifest)


def local_selection(selected: BitMap, block: a.ColumnBlock) -> BitMap:
    return (selected & BitMap(range(block.first, block.first + block.documents))).shift(
        -block.first
    )


def validate_index(index: a.FieldIndex, documents: int) -> None:
    if not index.projections or index.full.documents != documents:
        raise ValueError("incomplete analytical field coverage")
    for projection in index.projections:
        cursor = 0
        if not projection.blocks:
            raise ValueError("incomplete analytical projection")
        for block in projection.blocks:
            if (
                block.first != cursor
                or block.documents > BLOCK_ROWS
                or (not block.documents and documents)
            ):
                raise ValueError("unordered or incomplete analytical column coverage")
            cursor += block.documents
        if cursor != documents:
            raise ValueError("incomplete analytical projection coverage")


def select_index(
    catalog: Catalog,
    index: a.FieldIndex,
    selected: BitMap,
    operator: q.Comparison.Operator,
    threshold: Scalar,
    projection: int = p.FieldDistribution.SCALAR,
    class_name: str = "",
) -> BitMap:
    null = projection == q.FieldComparison.IS_NULL
    projected = (
        index.projections[0]
        if null
        else next(
            (
                part
                for part in index.projections
                if part.projection == projection and part.class_name == class_name
            ),
            None,
        )
    )
    if projected is None:
        raise ValueError("unindexed field projection")
    result = BitMap()
    for block in projected.blocks:
        local = local_selection(selected, block)
        if not local:
            continue
        column = load_column(catalog, block)
        if null:
            is_null = bool(threshold) == (operator == q.Comparison.OPERATOR_EQ)
            kept = local - column.present if is_null else local & column.present
        else:
            kept = column.select(local, operator, threshold)
        result |= kept.shift(block.first)
    return result


def profile_index(
    catalog: Catalog, index: a.FieldIndex, selected: BitMap, *, use_full: bool = True
) -> p.FieldProfile:
    if use_full and len(selected) == index.full.documents:
        result = p.FieldProfile()
        result.CopyFrom(index.full)
        return result
    result = p.FieldProfile(field=index.full.field, documents=len(selected))
    for projection in index.projections:
        distributions = []
        centered: list[tuple[int | float, Moments]] = []
        nulls = 0
        for block in projection.blocks:
            selection = local_selection(selected, block)
            column = load_column(catalog, block)
            nulls += len(selection) - selection.intersection_cardinality(column.present)
            distribution = (
                column.full if len(selection) == block.documents else column.distribution(selection)
            )
            distributions.append(distribution)
            if len(projection.blocks) > 1 and column.population_moments is not None:
                moment = column.selected_moments(selection)
                if moment.count:
                    centered.append((column.origin, moment))
        result.null_documents = nulls
        if projection.projection == 0:
            continue  # Coverage-only vector column.
        merged = merge_distributions(distributions)
        if centered:
            origin = centered[0][0]
            count, mean, m2, total = 0, 0.0, 0.0, 0.0
            for center, moment in centered:
                adjusted = (center - origin) + moment.mean
                delta = adjusted - mean
                combined = count + moment.count
                m2 += moment.squared_deviations + delta**2 * count * moment.count / combined
                mean += delta * moment.count / combined
                total += moment.total
                count = combined
            merged.numeric.mean = origin + mean
            merged.numeric.total = total
            merged.numeric.standard_deviation = (m2 / count) ** 0.5
        merged.projection = cast(p.FieldDistribution.Projection, projection.projection)
        merged.class_name = projection.class_name
        if projection.HasField("component"):
            merged.component = projection.component
        result.distributions.append(merged)
    if not index.projections:
        # Vector profiles expose coverage without inventing scalar distributions.
        result.null_documents = (
            index.full.null_documents if len(selected) == index.full.documents else 0
        )
    return result


def merge_distributions(distributions: list[p.FieldDistribution]) -> p.FieldDistribution:
    if len(distributions) == 1:
        result = p.FieldDistribution()
        result.CopyFrom(distributions[0])
        return result
    # Coalesce overlapping ranges from independent ordinal blocks, preserving bounds.
    buckets = sorted(
        (value(b.lower), value(b.upper), b.documents)
        for distribution in distributions
        for b in distribution.buckets
    )
    merged: list[tuple[Scalar, Scalar, int]] = []
    for low, high, count in buckets:
        if merged and _less_equal(low, merged[-1][1]):
            before, end, previous = merged[-1]
            merged[-1] = before, max(end, high), previous + count
        else:
            merged.append((low, high, count))
    while len(merged) > 64:
        i = min(range(len(merged) - 1), key=lambda i: merged[i][2] + merged[i + 1][2])
        low, _, left = merged[i]
        _, high, right = merged[i + 1]
        merged[i : i + 2] = [(low, high, left + right)]
    result = p.FieldDistribution()
    # Retain the endpoint type; empty distributions have no endpoints.
    template = next((d for d in distributions if d.buckets), None)
    if template is None:
        return result
    kind = template.buckets[0].lower.WhichOneof("value")
    assert kind is not None
    for low, high, count in merged:
        bucket = result.buckets.add(documents=count)
        setattr(bucket.lower, kind, low)
        setattr(bucket.upper, kind, high)
    numeric = [d.numeric for d in distributions if d.HasField("numeric")]
    if numeric:
        count, mean, m2 = 0, 0.0, 0.0
        for summary in numeric:
            delta = summary.mean - mean
            total = count + summary.documents
            m2 += (
                summary.standard_deviation**2 * summary.documents
                + delta**2 * count * summary.documents / total
            )
            mean += delta * summary.documents / total
            count = total
        result.numeric.CopyFrom(
            p.NumericSummary(
                documents=count,
                mean=mean,
                total=sum(n.total for n in numeric),
                standard_deviation=(m2 / count) ** 0.5,
                minimum=result.buckets[0].lower,
                maximum=result.buckets[-1].upper,
            )
        )
    return result


def _less_equal(left: Scalar, right: Scalar) -> bool:
    if isinstance(left, str):
        assert isinstance(right, str)
        return left <= right
    assert isinstance(right, (int, float))
    return left <= right


def restore(catalog: Catalog, resource: q.Query) -> IndexedQuery:
    from premixdb.engine.indexed import LINEAGE_MAGIC, IndexedQuery

    selection = catalog._storage.load(
        "query", resource.id, a.IndexedSelection, suffix=SELECTION_SUFFIX
    )
    if catalog._storage.read_object("query", resource.lineage) != LINEAGE_MAGIC + wire(selection):
        raise ValueError("indexed selection differs from its lineage")
    return IndexedQuery(load_population(catalog, selection.population_id), selection, resource)
