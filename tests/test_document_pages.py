"""Document drilldowns read only the visible prefixes and preserve occurrences."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest
from _type_support import coordinator

import premixdb as p
from premixdb._ids import _decode_id, _encode_id
from premixdb.engine.queries import Query
from premixdb.engine.snapshots import FRAME_BYTES, StoredDocument


@pytest.mark.parametrize("population", ["snapshot", "output", "retained"])
def test_document_preview_reads_only_visible_prefix_frames(tmp_path: Path, population: str) -> None:
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
        resource = snapshot if population == "snapshot" else query
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
            page = resource.preview(max_characters=4096)
            empty = resource.preview(offset=1, max_characters=4096)
        expected = text.replace("DROP", "") if population == "retained" else text
        assert (
            snapshot.profile().documents
            if population == "snapshot"
            else query.profile().output_documents
        ) == 1
        assert page[0]["text"] == expected[:4096]
        assert page[0]["ordinal"] == 0
        assert empty == []
        frame_reads = [
            call.args[0]
            for call in read.call_args_list
            if str(call.args[0])
            in {
                "snapshot/objects/" + bytes(frame["digest"]).hex()
                for frame in original.record["frames"]
            }
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
                page = query.preview(limit=100, offset=offset, max_characters=0)
                assert query.profile().output_documents == 125
                assert [item["ordinal"] for item in page] == list(
                    range(offset, min(offset + 100, 125))
                )
                assert [item["id"] for item in page] == [
                    _encode_id(bytes.fromhex(handle.row(i).id))
                    for i in range(offset, min(offset + 100, 125))
                ]
        selected = snapshot.query(steps=[p.where(p.text.bytes == 2)])
        assert selected.profile().output_documents == 1
        assert selected.preview()[0]["text"] == "é"
