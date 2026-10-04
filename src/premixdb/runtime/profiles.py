"""Profile materialized query occurrences using published field values."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from premixdb.storage.profiles import FieldProfiler, TextProfiler
from premixdb.v1 import profile_pb2 as p
from premixdb.v1 import query_pb2 as q

if TYPE_CHECKING:
    from premixdb.engine.queries import Query
    from premixdb.runtime.coordinator import Coordinator


def output_profiles(service: Coordinator, recipe: q.Query, handle: Query) -> list[p.FieldProfile]:
    """Profile the selected occurrences, retaining original external feature values."""
    from premixdb.runtime.enrichment import decode_value, load_build, read_rows

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
