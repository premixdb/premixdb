"""Sequence inspection reads saved assets only when a page contains examples."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest
from _type_support import coordinator, wordpiece_tokenizer

import premixdb as p
from premixdb._sequences import Sequence
from premixdb.execution.coordinator import Coordinator
from premixdb.execution.inspection import sequences


@pytest.fixture(params=["wordpiece", "bytes"])
def ready(
    tmp_path: Path, request: pytest.FixtureRequest
) -> Iterator[tuple[Coordinator, p.Dataset]]:
    with p.PremixDB(storage=tmp_path) as db:
        dataset = (
            db.corpus("inspection", [p.Source("a", "hello world playing! café 中文\n" * 40)])
            .query()
            .dataset(
                tokenizer=p.ByteTokenizer() if request.param == "bytes" else wordpiece_tokenizer(),
                packing=p.Concat(
                    separator=256 if request.param == "bytes" else 2,
                    drop_remainder=False,
                    pad_token=257 if request.param == "bytes" else 3,
                ),
                sequence_length=16,
            )
            .wait()
        )
        yield coordinator(db), dataset


def test_inspection_uses_saved_decoder(ready: tuple[Coordinator, p.Dataset]) -> None:
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
        patch.object(Sequence, "_preview", side_effect=AssertionError("read masks")),
    ):
        page = sequences(service, dataset.id, {})
    assert page["total"] == len(dataset)
    assert len(page["rows"]) == len(expected) == 20
    for row, example in zip(page["rows"], expected, strict=True):
        assert row["ordinal"] == example["ordinal"]
        assert row["tokens"] == example["tokens"]
        assert row["text"] == example["text"]
        assert row["truncated"] == example["truncated"]


@pytest.mark.parametrize("parameters", [{"offset": ["999999"]}, {"documents": ["999999"]}])
def test_empty_inspection_pages_do_not_load_tokenizer(
    ready: tuple[Coordinator, p.Dataset], parameters: dict[str, list[str]]
) -> None:
    service, dataset = ready
    with (
        patch.object(service, "_tokenizer", side_effect=AssertionError("execution tokenizer")),
        patch("premixdb._sequences.preview_decoder", side_effect=AssertionError("loaded asset")),
    ):
        page = sequences(service, dataset.id, parameters)
    assert page["rows"] == []
    assert page["total"] == (len(dataset) if "offset" in parameters else 0)


@pytest.mark.parametrize(
    "parameters",
    [
        {"offset": ["-1"]},
        {"offset": ["invalid"]},
        {"documents": ["-1"]},
        {"documents": ["invalid"]},
        {"crossing": ["false"]},
        {"padding": ["false"]},
        {"source": ["invalid"]},
    ],
)
def test_invalid_inspection_filters_fail_before_catalog_reads(
    tmp_path: Path, parameters: dict[str, list[str]]
) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        with patch("premixdb.execution.inspection.resource", side_effect=AssertionError("read")):
            with pytest.raises(ValueError):
                sequences(coordinator(db), "unused", parameters)
