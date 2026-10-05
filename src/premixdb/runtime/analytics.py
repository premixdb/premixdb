"""Execute ordinary predicates without expanding the prepared population into rows."""

from __future__ import annotations

from bisect import bisect_left
from typing import TYPE_CHECKING

from pyroaring import BitMap

from premixdb.engine.contracts import QuerySummary
from premixdb.engine.indexed import LINEAGE_MAGIC, IndexedQuery, counts, total
from premixdb.engine.queries import Query
from premixdb.internal import analytics_pb2 as a
from premixdb.runtime.analytical_preparation import prepare_population, prepared_field
from premixdb.schemas.protobuf import copy_message, wire
from premixdb.storage.analytics import SELECTION_SUFFIX, load_column, profile_index, select_index
from premixdb.v1 import query_pb2 as q
from premixdb.v1 import status_pb2 as status

if TYPE_CHECKING:
    from premixdb.runtime.coordinator import Coordinator


def execute(
    service: Coordinator, recipe: q.Query, *, publish: bool
) -> tuple[q.Query, Query] | None:
    """Use indexes for arbitrary thresholds and conjunctions; retain policy execution."""
    if (
        recipe.HasField("sampling")
        or recipe.HasField("decontaminate")
        or recipe.index_snapshot_ids
        or any(
            op.WhichOneof("kind") not in ("where", "field_where", "document_ids")
            for op in recipe.operations
        )
        or any(
            op.HasField("field_where")
            and op.field_where.projection == q.FieldComparison.VECTOR_COMPONENT
            for op in recipe.operations
        )
    ):
        return None
    from premixdb.runtime import environment
    from premixdb.runtime.catalog import resolve
    from premixdb.runtime.coordinator import query_profile
    from premixdb.runtime.enrichment import build, validate_selector
    from premixdb.storage.profiles import estimate_query

    for plan in resolve(recipe):
        service._once(plan, lambda plan=plan: build(service, plan))
    population = prepare_population(service, recipe.snapshot_ids)
    fields = {}
    for pin in recipe.field_snapshot_ids:
        item, indexed = prepared_field(service, population, pin)
        fields[pin] = (item.field, indexed)
    selected = BitMap(range(population.manifest.documents))
    initial = before = counts(population, selected)
    summary: QuerySummary = dict(input=initial, steps=[], output=initial)
    survivors: list[bytes] = []
    for operation in recipe.operations:
        if operation.HasField("where"):
            comparison = operation.where
            threshold = (
                comparison.text if comparison.WhichOneof("value") == "text" else comparison.count
            )
            selected = select_index(
                service,
                population.intrinsic(comparison.field),
                selected,
                comparison.operator,
                threshold,
            )
        elif operation.HasField("field_where"):
            selector = operation.field_where
            spec, indexed = fields[selector.field_snapshot_id]
            validate_selector(spec, selector)
            threshold = getattr(selector, selector.WhichOneof("value"))
            selected = select_index(
                service,
                indexed,
                selected,
                selector.operator,
                threshold,
                selector.projection,
                selector.class_name,
            )
        else:
            kept = BitMap()

            def document_id(ordinal: int) -> bytes:
                from premixdb.engine.analytics import BLOCK_ROWS

                raw = population.part(ordinal // BLOCK_ROWS)["id"][ordinal % BLOCK_ROWS].as_py()
                assert isinstance(raw, bytes)
                return raw

            for identity in operation.document_ids.ids:
                ordinal = bisect_left(
                    range(population.manifest.documents), identity, key=document_id
                )
                if ordinal in selected and document_id(ordinal) == identity:
                    kept.add(ordinal)
            selected = kept
        after = counts(population, selected)
        summary["steps"].append(dict(before=before, after=after))
        before = after
        survivors.append(selected.serialize())
    summary["output"] = before
    result = copy_message(recipe)
    result.profile.CopyFrom(query_profile(summary, len(recipe.snapshot_ids)))
    for indexed in [*population.manifest.intrinsic, *[entry[1] for entry in fields.values()]]:
        result.profile.fields.append(profile_index(service, indexed, selected))
    corpus = population.intrinsic(q.FIELD_SOURCE_CORPUS_ID)
    labels = {
        load_column(service, block).scalar(rank)
        for block in corpus.projections[0].blocks
        for rank in range(len(load_column(service, block).dictionary))
    }
    for label in labels:
        assert isinstance(label, str)
        members = select_index(service, corpus, selected, q.Comparison.OPERATOR_EQ, label)
        if members:
            result.profile.source_documents[label] = len(members)
            result.profile.source_content_bytes[label] = total(
                population, q.FIELD_TEXT_BYTES, members
            )
    code = environment.resolve_code(recipe.git_commit)
    selection = a.IndexedSelection(
        query_id=recipe.id,
        population_id=population.manifest.id,
        snapshot_ids=recipe.snapshot_ids,
        selected=selected.serialize(),
        steps=survivors,
        repository=code.repository,
        commit=code.commit,
        environment=bytes.fromhex(code.environment),
    )
    handle = IndexedQuery(population, selection, result)
    handle._encoding_provider = service._encodings
    if not publish:
        return copy_message(recipe), handle
    from premixdb.storage.preview import inline

    result.status = status.STATUS_COMPLETED
    result.estimate.CopyFrom(estimate_query(service, recipe))
    inline(service._storage, result.preview, handle)
    result.lineage.CopyFrom(service._storage.put("query", LINEAGE_MAGIC + wire(selection)))
    service._storage.save("query", result.id, selection, suffix=SELECTION_SUFFIX)
    service._storage.save("query", result.id, result)
    with service._lock:
        service._query_handles[result.id] = handle
        service._queries[result.id] = result
    return copy_message(result), handle
