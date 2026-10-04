"""Sequence previews read saved assets only when a page contains examples."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest
from _type_support import coordinator, wordpiece_tokenizer

import premixdb as p
from premixdb._sequences import Sequence
from premixdb.execution.coordinator import Coordinator


@pytest.fixture(params=["wordpiece", "bytes"])
def ready(
    tmp_path: Path, request: pytest.FixtureRequest
) -> Iterator[tuple[Coordinator, p.Dataset]]:
    with p.PremixDB(storage=tmp_path) as db:
        dataset = (
            db.Corpus("inspection", [p.Source("a", "hello world playing! café 中文\n" * 40)])
            .query()
            .mix(
                tokenizer=p.ByteTokenizer() if request.param == "bytes" else wordpiece_tokenizer(),
                packing=p.Concat(
                    separator=256 if request.param == "bytes" else 2,
                    drop_remainder=False,
                    pad_token=257 if request.param == "bytes" else 3,
                ),
                sequence_length=16,
            )[0]
            .wait()
        )
        yield coordinator(db), dataset


def test_preview_uses_saved_decoder(ready: tuple[Coordinator, p.Dataset]) -> None:
    service, dataset = ready
    expected = dataset.preview(limit=20, max_characters=1_000_000)
    with (
        patch.object(service, "_tokenizer", side_effect=AssertionError("execution tokenizer")),
        patch.object(
            Sequence,
            "tokens",
            new_callable=PropertyMock,
            side_effect=AssertionError("expanded tokens"),
        ),
    ):
        page = dataset.preview(limit=20, max_characters=1_000_000)
    assert len(dataset) > 0
    assert len(page) == len(expected) == 20
    for row, example in zip(page, expected, strict=True):
        assert row["ordinal"] == example["ordinal"]
        assert row["tokens"] == example["tokens"]
        assert row["text"] == example["text"]
        assert row["truncated"] == example["truncated"]


def test_empty_preview_pages_do_not_load_tokenizer(ready: tuple[Coordinator, p.Dataset]) -> None:
    service, dataset = ready
    with (
        patch.object(service, "_tokenizer", side_effect=AssertionError("execution tokenizer")),
        patch("premixdb._sequences.preview_decoder", side_effect=AssertionError("loaded asset")),
    ):
        assert dataset.preview(offset=999999) == []
        assert dataset.preview(limit=0) == []


@pytest.mark.parametrize("offset", [-1, "invalid", True])
def test_invalid_preview_offsets_fail_before_catalog_reads(tmp_path: Path, offset: object) -> None:
    from _type_support import invalid_call

    with p.PremixDB(storage=tmp_path) as db:
        dataset = db.Corpus("invalid", [p.Source("a", "text")]).query().mix()[0]
        with patch.object(dataset, "wait", side_effect=AssertionError("read")):
            with pytest.raises(ValueError):
                invalid_call(dataset.preview, offset=offset)
