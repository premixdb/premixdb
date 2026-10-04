"""Packed previews are bounded, paged, decoded, and usable after reopening."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest
from _type_support import (
    PreviewOptions,
    tokenizer_packing,
    wordpiece_tokenizer,
)

import premixdb as p
from premixdb.training.sequences import Sequence
from premixdb.v1 import data_mixture_pb2 as dataset_pb


@pytest.mark.parametrize(
    "tokenizer", [wordpiece_tokenizer(), p.ByteTokenizer()], ids=["wordpiece", "bytes"]
)
def test_dataset_preview_stays_lazy_and_matches_materialized_examples(
    tmp_path: Path, tokenizer: dataset_pb.Tokenizer
) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        dataset = (
            db.Corpus("preview", [p.Source("a", "Hello world.\n" * 6)])
            .query()
            .mix(tokenizer=tokenizer, packing=tokenizer_packing(tokenizer), sequence_length=8)[0]
        )
        assert dataset.status is p.ExecutionStatus.PENDING
        examples = dataset.preview()
        assert dataset.status is p.ExecutionStatus.PENDING
        assert not dataset._proto.HasField("profile")
        assert len(examples) == 3
        dataset.wait()
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


@pytest.mark.parametrize(
    "tokenizer", [wordpiece_tokenizer(), p.ByteTokenizer()], ids=["wordpiece", "bytes"]
)
def test_dataset_preview_pages_across_sequence_index_pages_after_reopening(
    tmp_path: Path, tokenizer: dataset_pb.Tokenizer
) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        dataset = (
            db.Corpus(
                "preview",
                [
                    p.Source(
                        "a",
                        ("hello " if tokenizer.HasField("hugging_face") else "a") * (131 * 8 - 1),
                    )
                ],
            )
            .query()
            .mix(tokenizer=tokenizer, packing=tokenizer_packing(tokenizer), sequence_length=8)[0]
            .wait()
        )
        assert len(dataset) == 131
        expected = [dataset[index] for index in range(127, 131)]
        with (
            patch.object(
                Sequence,
                "tokens",
                new_callable=PropertyMock,
                side_effect=AssertionError("expanded tokens"),
            ),
            patch.object(
                Sequence,
                "mask",
                new_callable=PropertyMock,
                side_effect=AssertionError("expanded mask"),
            ),
            patch.object(
                Sequence,
                "attention_mask",
                new_callable=PropertyMock,
                side_effect=AssertionError("expanded attention mask"),
            ),
        ):
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
            "premixdb.runtime.coordinator.Coordinator", side_effect=AssertionError("compute")
        ):
            assert db._dataset(dataset_id).preview(limit=4, offset=127) == examples


@pytest.mark.integration
@pytest.mark.parametrize(
    "tokenizer", [wordpiece_tokenizer(), p.ByteTokenizer()], ids=["wordpiece", "bytes"]
)
def test_read_only_paged_preview_does_not_import_packing_code(
    tmp_path: Path, tokenizer: dataset_pb.Tokenizer
) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        dataset = (
            db.Corpus("preview", [p.Source("a", " hello" * 96)])
            .query()
            .mix(tokenizer=tokenizer, packing=tokenizer_packing(tokenizer), sequence_length=8)[0]
        )
        dataset.wait()
        expected = dataset.preview(limit=1, offset=10)
        assert expected
        dataset_id = dataset.id
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
import sys

sys.modules["premixdb.storage.tokens"] = None
sys.modules["premixdb.engine.datasets"] = None
import premixdb as p

with p.PremixDB(storage=sys.argv[1], read_only=True) as db:
    print(json.dumps(db._dataset(sys.argv[2]).preview(limit=1, offset=10)))
""",
            str(tmp_path),
            dataset_id,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(result.stdout) == expected


def test_preview_includes_unicode_separators_padding_and_truncation(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        dataset = (
            db.Corpus("unicode", [p.Source("a", "é🌍ok")])
            .query()
            .mix(tokenizer=p.ByteTokenizer(), sequence_length=8)[0]
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
            db.Corpus("long", [p.Source("a", "abc" * 200)])
            .query()
            .mix(tokenizer=p.ByteTokenizer(), sequence_length=512)[0]
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
@pytest.mark.parametrize("kind", ["snapshot", "query", "dataset"])
def test_invalid_preview_options_fail_before_materializing(
    tmp_path: Path, options: PreviewOptions, kind: str
) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.Corpus("preview", [p.Source("a", "hello")])
        resource = (
            snapshot
            if kind == "snapshot"
            else snapshot.query()
            if kind == "query"
            else snapshot.query().mix()[0]
        )
        with patch.object(type(resource), "wait", side_effect=AssertionError("started execution")):
            with pytest.raises(ValueError):
                resource.preview(**options)
            assert resource.preview(limit=0) == []


def test_empty_dataset_preview_is_empty(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        dataset = db.Corpus("empty", []).query().mix(tokenizer=p.ByteTokenizer())[0]
        assert dataset.preview() == []
