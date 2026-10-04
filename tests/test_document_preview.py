"""Inline previews read a verified prefix instead of decoding whole documents."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest
from _type_support import coordinator

import premixdb as p
from premixdb._ids import _decode_id
from premixdb.engine.contracts import DocumentRecord, Frame
from premixdb.engine.snapshots import FRAME_BYTES, StoredDocument
from premixdb.execution.previewing import _text
from premixdb.execution.storage import ObjectStore


def test_retained_previews_cross_utf8_frames_and_skip_removed_frames(tmp_path: Path) -> None:
    parts = ["pré", "DROP", "🌍fin", "DROP", "tail"]
    with ObjectStore(tmp_path) as store:
        frames: list[Frame] = []
        for part in parts:
            data = part.encode()
            object = store.put("snapshot", data)
            frames.append(Frame(digest=list(object.blake3_digest), bytes=len(data)))
        source = DocumentRecord(key="fragmented", content=[], bytes=23, frames=frames)
        ranges = [(0, 4), (8, 15), (19, 23)]
        expected = "pré🌍fintail"
        with patch.object(store, "_get", wraps=store._get) as read:
            for width, count in ((0, 0), (3, 2), (5, 2), (11, 3), (100, 3)):
                read.reset_mock()
                assert _text(store, source, width, ranges) == (
                    expected[:width],
                    width < len(expected),
                )
                assert read.call_count == count
                assert all(
                    "snapshot/objects/" + bytes(frames[1]["digest"]).hex() != call.args[0]
                    for call in read.call_args_list
                )
            read.reset_mock()
            assert _text(store, source, 100, []) == ("", False)
            assert not read.called
        # Ranges can continue through several frames and end exactly at a boundary.
        assert _text(store, source, 100, [(2, 12), (15, 23)]) == ("éDROP🌍DROPtail", False)


@pytest.mark.parametrize("kind", ["snapshot", "query"])
def test_inline_preview_reads_only_the_prefix_frame(tmp_path: Path, kind: str) -> None:
    text = "é🌍" * (FRAME_BYTES // 6 + 50) + "tail" * (FRAME_BYTES // 4 + 16)
    reads: list[str] = []
    get = ObjectStore._get

    def frame(store: ObjectStore, relative: str | Path, limit: int = 64 * 1024 * 1024) -> bytes:
        if str(relative).startswith("snapshot/objects/"):
            reads.append(str(relative))
        return get(store, relative, limit)

    with (
        p.PremixDB(storage=tmp_path) as db,
        patch.object(ObjectStore, "_get", autospec=True, side_effect=frame),
        patch.object(
            StoredDocument,
            "text",
            new_callable=PropertyMock,
            side_effect=AssertionError("decoded whole document"),
        ),
    ):
        snapshot = db.Corpus("preview", [p.Source("large", text)])
        stored = coordinator(db)._snapshot(_decode_id(snapshot.id))
        document = next(iter(stored.documents.values()))
        assert isinstance(document, StoredDocument)
        assert len(document.record["frames"]) >= 2
        resource = snapshot if kind == "snapshot" else snapshot.query().wait()
        example = resource.preview()[0]
        assert example["text"] == text[:1024]
        assert example["truncated"]
        frames = {
            "snapshot/objects/" + bytes(frame["digest"]).hex()
            for frame in document.record["frames"]
        }
        text_reads = [relative for relative in reads if relative in frames]
        assert len(text_reads) == (1 if kind == "snapshot" else 2)
        assert set(text_reads) == {
            "snapshot/objects/" + bytes(document.record["frames"][0]["digest"]).hex()
        }


@pytest.mark.integration
@pytest.mark.parametrize("text, retained", [("pré\n秘密\nfin", "pré\n\nfin"), ("秘密", "")])
def test_saved_retained_previews_do_not_import_query_execution(
    tmp_path: Path, text: str, retained: str
) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        target = db.Corpus("target", [p.Source("a", text)])
        reference = db.Corpus("reference", [p.Source("b", "秘密")])
        query = target.query(
            decontaminate=p.decontaminate(reference, algorithm="line", granularity="span")
        ).wait()
        expected = query.preview(max_characters=2048)
        assert expected[0]["text"] == retained
        query_id = query.id
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
import sys

for module in (
    "premixdb.engine.queries",
    "premixdb.engine.curation",
    "premixdb.execution.selections",
    "premixdb.execution.coordinator",
):
    sys.modules[module] = None
import premixdb as p

with p.PremixDB(storage=sys.argv[1], read_only=True) as db:
    print(json.dumps(db._query(sys.argv[2]).preview(max_characters=2048)))
""",
            str(tmp_path),
            query_id,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(result.stdout) == expected
