"""Document drilldowns read only the visible prefixes and preserve occurrences."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest
from _type_support import coordinator

import premixdb as p
from premixdb._ids import _decode_id
from premixdb.engine.queries import Query
from premixdb.engine.snapshots import FRAME_BYTES, StoredDocument
from premixdb.execution.inspection import rows


@pytest.mark.parametrize("population", ["snapshot", "input", "output", "retained"])
def test_document_inspection_reads_only_visible_prefix_frames(
    tmp_path: Path, population: str
) -> None:
    text = "DROP\n" + "é🌍" * (FRAME_BYTES // 6 + 100) + "\nDROP"
    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.corpus("inspection", [p.Source("large", text)])
        reference = db.corpus("reference", [p.Source("remove", "DROP")])
        query = (
            snapshot.query(
                decontaminate=p.decontaminate(reference, algorithm="line", granularity="span")
            )
            if population == "retained"
            else snapshot.query()
        ).wait()
        service = coordinator(db)
        original = next(iter(service._snapshot(_decode_id(snapshot.id)).documents.values()))
        assert isinstance(original, StoredDocument)
        assert len(original.record["frames"]) > 1
        parameters = {"population": ["input"]} if population == "input" else {}
        kind = "snapshot" if population == "snapshot" else "query"
        identity = snapshot.id if population == "snapshot" else query.id
        with (
            patch.object(
                StoredDocument,
                "text",
                new_callable=PropertyMock,
                side_effect=AssertionError("decoded entire document"),
            ),
            patch.object(Query, "rows", side_effect=AssertionError("copied entire query")),
            patch.object(service._storage, "_get", wraps=service._storage._get) as read,
        ):
            page = rows(service, kind, identity, parameters)
            empty = rows(service, kind, identity, {**parameters, "offset": ["1"]})
        expected = text.replace("DROP", "") if population == "retained" else text
        assert page["total"] == empty["total"] == 1
        assert page["rows"][0]["text"] == expected[:4096]
        assert page["rows"][0]["ordinal"] == 0
        assert empty["rows"] == []
        frame_reads = [
            call.args[0]
            for call in read.call_args_list
            if str(call.args[0]).startswith("snapshot/objects/")
        ]
        assert frame_reads == [
            "snapshot/objects/" + bytes(original.record["frames"][0]["digest"]).hex()
        ]


def test_document_pages_preserve_sampled_occurrences_and_filtered_totals(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.corpus("pages", [p.Source("keep", "é"), p.Source("other", "🌍")])
        query = snapshot.query(sampling=p.sample(seed=7, documents=125, replacement=True)).wait()
        service = coordinator(db)
        handle = service._query(_decode_id(query.id))
        with patch.object(Query, "rows", side_effect=AssertionError("copied entire query")):
            for offset in (0, 100, 125, 1000):
                page = rows(service, "query", query.id, {"offset": [str(offset)]})
                assert page["total"] == 125
                assert page["offset"] == offset
                assert [item["ordinal"] for item in page["rows"]] == list(
                    range(offset, min(offset + 100, 125))
                )
                assert [item["id"] for item in page["rows"]] == [
                    handle.row(i).id for i in range(offset, min(offset + 100, 125))
                ]
            page = rows(
                service,
                "query",
                query.id,
                {
                    "field": ["FIELD_TEXT_BYTES"],
                    "lower": ['{"count":"2"}'],
                    "upper": ['{"count":"2"}'],
                },
            )
        expected = [row.ordinal for row in handle if row.source_key == "keep"]
        assert page["total"] == len(expected)
        assert [item["ordinal"] for item in page["rows"]] == expected[:100]
        assert all(item["text"] == "é" for item in page["rows"])
