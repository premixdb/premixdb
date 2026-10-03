"""Packed previews are bounded, paged, decoded, and usable after reopening."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from _type_support import (
    PreviewOptions,
)

import premixdb as p
from premixdb._policies import ByteTokenizer as BytePolicy
from premixdb.v1 import dataset_pb2 as dataset_pb


@pytest.mark.parametrize("tokenizer", [None, p.ByteTokenizer()])
def test_dataset_preview_materializes_and_reuses_inline_examples(
    tmp_path: Path, tokenizer: dataset_pb.Tokenizer | BytePolicy
) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        dataset = (
            db.corpus("preview", [p.Source("a", "Hello world.\n" * 6)])
            .query()
            .dataset(tokenizer=tokenizer, sequence_length=8)
        )
        assert dataset.status is p.ExecutionStatus.PENDING
        examples = dataset.preview()
        assert dataset.status is p.ExecutionStatus.COMPLETED
        assert len(examples) == 3
        for index, example in enumerate(examples):
            sequence = dataset[index]
            assert example["ordinal"] == index
            assert example["tokens"] == sequence.tokens
            assert example["mask"] == sequence.mask
            assert example["attention_mask"] == sequence.attention_mask
            assert example["document_ids"] == sequence.document_ids()
            assert example["text"] == dataset._proto.preview.sequences[index].text
            assert example["truncated"] is False
        with patch.object(db._object_reader, "read", side_effect=AssertionError("read data")):
            assert dataset.preview() == examples


@pytest.mark.parametrize("tokenizer", [None, p.ByteTokenizer()])
def test_dataset_preview_pages_across_sequence_index_pages_after_reopening(
    tmp_path: Path, tokenizer: dataset_pb.Tokenizer | BytePolicy
) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        dataset = (
            db.corpus(
                "preview",
                [p.Source("a", ("a" if tokenizer is not None else " hello") * (131 * 8 - 1))],
            )
            .query()
            .dataset(tokenizer=tokenizer, sequence_length=8)
            .wait()
        )
        assert len(dataset) == 131
        expected = [dataset[index] for index in range(127, 131)]
        examples = dataset.preview(limit=4, offset=127)
        for example, sequence in zip(examples, expected, strict=True):
            assert example["ordinal"] == sequence.ordinal
            assert example["tokens"] == sequence.tokens
            assert example["mask"] == sequence.mask
            assert example["attention_mask"] == sequence.attention_mask
            assert example["document_ids"] == sequence.document_ids()
            assert example["text"]
        dataset_id = dataset.id
    with p.PremixDB(storage=tmp_path, read_only=True) as db:
        with patch(
            "premixdb.execution.coordinator.Coordinator", side_effect=AssertionError("compute")
        ):
            assert db._dataset(dataset_id).preview(limit=4, offset=127) == examples


def test_preview_includes_unicode_separators_padding_and_truncation(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        dataset = (
            db.corpus("unicode", [p.Source("a", "é🌍ok")])
            .query()
            .dataset(tokenizer=p.ByteTokenizer(), sequence_length=8)
        )
        examples = dataset.preview()
        assert examples[0]["text"] == "é🌍ok"
        assert examples[0]["tokens"] == list("é🌍ok".encode())
        assert examples[1]["text"] == "<separator:256 ×1><padding:257 ×7>"
        assert examples[1]["mask"] == [True] + [False] * 7
        assert examples[1]["document_ids"] == []
        shortened = dataset.preview(limit=1, max_characters=1)[0]
        assert shortened["text"] == "é" and shortened["truncated"]
        assert dataset.preview(max_characters=0)[0]["text"] == ""
        assert dataset.preview(limit=1, offset=100) == []
        long = (
            db.corpus("long", [p.Source("a", "abc" * 200)])
            .query()
            .dataset(tokenizer=p.ByteTokenizer(), sequence_length=512)
        )
        example = long.preview(limit=1)[0]
        assert len(example["tokens"]) == len(example["mask"]) == 256
        assert example["truncated"]


@pytest.mark.parametrize(
    "options",
    [
        dict(limit=-1),
        dict(limit=True),
        dict(limit=1001),
        dict(offset=-1),
        dict(max_characters=-1),
        dict(max_characters=1_000_001),
    ],
)
def test_invalid_preview_options_fail_before_materializing(
    tmp_path: Path, options: PreviewOptions
) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        dataset = db.corpus("preview", [p.Source("a", "hello")]).query().dataset()
        with patch.object(p.Dataset, "wait", side_effect=AssertionError("started packing")):
            with pytest.raises(ValueError):
                dataset.preview(**options)
            assert dataset.preview(limit=0) == []


def test_empty_dataset_preview_is_empty(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        dataset = db.corpus("empty", []).query().dataset(tokenizer=p.ByteTokenizer())
        assert dataset.preview() == []
