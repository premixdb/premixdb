"""Intrinsic projections use metadata and reject unspecified field IDs."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest
from _type_support import coordinator

import premixdb as p
from premixdb import _runtime
from premixdb._ids import _decode_id
from premixdb.engine.curation import selector_key
from premixdb.engine.snapshots import StoredDocument
from premixdb.execution.coordinator import Coordinator
from premixdb.execution.enrichment import projections, query_inputs
from premixdb.execution.planner import field_definitions
from premixdb.v1 import query_pb2 as q


@pytest.fixture
def ready(tmp_path: Path) -> Iterator[tuple[Coordinator, p.Query]]:
    with p.PremixDB(storage=tmp_path) as db:
        query = db.corpus("intrinsic", [p.Source("https://example/a", "é🌍x")]).query().wait()
        yield coordinator(db), query


def test_intrinsic_projections_do_not_read_stored_text(ready: tuple[Coordinator, p.Query]) -> None:
    service, resource = ready
    query = service._query(_decode_id(resource.id))
    document = query.row(0).document
    assert isinstance(document, StoredDocument)
    selectors = [
        q.FieldComparison(field=field)
        for field in (
            q.FIELD_TEXT_BYTES,
            q.FIELD_TEXT_CHARACTERS,
            q.FIELD_OBJECT_URI,
            q.FIELD_SOURCE_CORPUS_ID,
        )
    ]
    with patch.object(
        StoredDocument, "text", new_callable=PropertyMock, side_effect=AssertionError("read text")
    ):
        values = projections(service, query, selectors)
    for selector, expected in zip(
        selectors, [7, 3, "https://example/a", document.corpus_id], strict=True
    ):
        assert values[selector_key(selector)] == {document.id: expected}


def test_intrinsic_sampling_plans_and_selects_without_reading_text(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.corpus("sampling", [p.Source("https://example/a", "é🌍x")])
        query = snapshot.query(
            sampling=p.sample(
                seed=3, documents=1, domains=(p.text.characters, p.object.uri, p.source.corpus_id)
            )
        )
        recipe = query._proto
        with patch.object(
            StoredDocument,
            "text",
            new_callable=PropertyMock,
            side_effect=AssertionError("read text"),
        ):
            with query_inputs(coordinator(db), recipe) as (index, steps):
                result = index.execute(
                    steps, _runtime.resolve_code(recipe.git_commit), field_definitions(recipe)
                )
        assert isinstance(result.row(0).document, StoredDocument)
        assert result.id == _decode_id(query.id).hex()
        assert result.row_count == 1
        assert result.summary()["output"] == {"documents": 1, "bytes": 7, "characters": 3}
