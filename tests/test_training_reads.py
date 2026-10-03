"""Coalesced verified IO, batched tensors, and distributed streaming coverage."""

from __future__ import annotations

import pickle
import struct
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from _type_support import invalid_call
from blake3 import blake3

import premixdb as p
from premixdb import RangeReader
from premixdb._policies import ByteTokenizer as BytePolicy
from premixdb._resources import _read_page
from premixdb._torch import StreamingDataset, TorchDataset
from premixdb._typing import Scalar
from premixdb.execution.storage import ObjectStore
from premixdb.v1 import dataset_pb2 as d
from premixdb.v1 import dataset_pb2 as dataset_pb
from premixdb.v1.storage_pb2 import ObjectRef, SpanRef


def span(obj: ObjectRef, data: bytes, start: int = 0, end: int | None = None) -> SpanRef:
    end = len(data) if end is None else end
    return SpanRef(object=obj, start=start, end=end, blake3_digest=blake3(data[start:end]).digest())


def stored_dataset(store: ObjectStore, count: int = 132) -> dataset_pb.Dataset:
    tokens = struct.pack(f"<{count * 2}I", *[v for i in range(count) for v in (i, 2**32 - 1)])
    masks = bytes([1, 0]) * count
    token_object = store.put("dataset", tokens)
    mask_object = store.put("dataset", masks)
    resource = d.Dataset(sequence_length=2)
    resource.profile.sequences = count
    for first in range(0, count, 128):
        page = d.SequenceBatch()
        for i in range(first, min(first + 128, count)):
            page.sequences.add(
                ordinal=i,
                tokens=span(token_object, tokens, i * 8, (i + 1) * 8),
                attention_mask=span(mask_object, masks, i * 2, (i + 1) * 2),
                loss_mask=span(mask_object, masks, i * 2, (i + 1) * 2),
            )
        data = page.SerializeToString()
        resource.sequences.append(span(store.put("dataset", data), data))
    return resource


def test_coalescing_preserves_order_duplicates_and_checks_each_original_span(
    tmp_path: Path,
) -> None:
    with ObjectStore(tmp_path) as store:
        data = b"abcdefghijklmnop"
        obj = store.put("dataset", data)
        parts = [span(obj, data, 8, 12), span(obj, data, 0, 4), span(obj, data, 3, 9)]
        parts.append(parts[0])
        reader = RangeReader(local_root=tmp_path)
        reads = Mock(wraps=reader._read_bytes)
        reader._read_bytes = reads
        reads.reset_mock()
        assert reader.read_many(parts) == [b"ijkl", b"abcd", b"defghi", b"ijkl"]
        assert reads.call_count == 1
        assert (reads.call_args.args[0].start, reads.call_args.args[0].end) == (0, 12)
        parts[1].blake3_digest = b"x" * 32
        with pytest.raises(ValueError, match="integrity"):
            reader.read_many(parts)
        assert reader.read_many([]) == []
        with pytest.raises(ValueError):
            reader.read_many([], max_gap_bytes=-1)
        reader.close()


def test_batch_loading_matches_individual_reads_and_coalesces_file_reads(tmp_path: Path) -> None:
    with ObjectStore(tmp_path) as store:
        resource = stored_dataset(store)
        reader = RangeReader(local_root=tmp_path)
        reads = Mock(wraps=reader._read_bytes)
        reader._read_bytes = reads
        data = TorchDataset(resource, reader)
        indices = [130, 128, 129, 130]
        individual = [data[i] for i in indices]
        reads.reset_mock()
        batched = TorchDataset(resource, reader).__getitems__(indices)
        assert reads.call_count == 3  # one index page, one token range, one mask range
        for expected, actual in zip(individual, batched, strict=True):
            for name in expected:
                assert actual[name].equal(expected[name])
        assert batched[0]["input_ids"].tolist() == [130, 2**32 - 1]
        assert batched[0]["labels"].tolist() == [130, -100]
        assert data.__getitems__([]) == []
        with pytest.raises(IndexError):
            data.__getitems__([132])
        reader.close()


@pytest.mark.parametrize("batched", [False, True])
def test_whole_object_reads_verify_the_object_digest(tmp_path: Path, batched: bool) -> None:
    with ObjectStore(tmp_path) as store:
        data = b"unchanged content"
        reference = span(store.put("dataset", data), data)
        reference.object.blake3_digest = b"x" * 32
        reader = RangeReader(local_root=tmp_path)
        with pytest.raises(ValueError, match="object integrity"):
            if batched:
                reader.read_many([reference])
            else:
                reader.read(reference)
        reader.close()


def test_streaming_ranks_and_workers_cover_every_sequence_once(tmp_path: Path) -> None:
    with ObjectStore(tmp_path) as store:
        resource = stored_dataset(store, count=385)
    reader = RangeReader(local_root=tmp_path)

    def read(rank: int, worker: int = 0, workers: int = 1, epoch: int = 0) -> list[int]:
        data = StreamingDataset(resource, reader, rank=rank, world_size=2, seed=42, epoch=epoch)
        with patch(
            "premixdb._torch.get_worker_info",
            return_value=SimpleNamespace(id=worker, num_workers=workers),
        ):
            return [int(item["input_ids"][0].item()) for item in data]

    ranks = [read(rank) for rank in range(2)]
    assert sorted(value for rank in ranks for value in rank) == list(range(385))
    assert ranks[0] == read(0)
    assert ranks != [read(rank, epoch=1) for rank in range(2)]
    for rank in range(2):
        workers = [read(rank, worker, 2) for worker in range(2)]
        assert sorted(value for worker in workers for value in worker) == sorted(ranks[rank])
    reader.close()


@pytest.mark.integration
def test_streaming_dataset_survives_spawn_and_session_close(tmp_path: Path) -> None:
    from torch.utils.data import DataLoader

    with ObjectStore(tmp_path) as store:
        resource = stored_dataset(store, count=129)
    data = StreamingDataset(resource, RangeReader(local_root=tmp_path), seed=7)
    restored = pickle.loads(pickle.dumps(data))
    loader = DataLoader(restored, batch_size=16, num_workers=2, multiprocessing_context="spawn")
    ordinals = [value for batch in loader for value in batch["input_ids"][:, 0].tolist()]
    assert sorted(ordinals) == list(range(129))
    data.data.reader.close()


@pytest.mark.parametrize(
    "kwargs", [dict(rank=0), dict(rank=2, world_size=2), dict(seed=True), dict(epoch=-1)]
)
def test_streaming_rejects_invalid_topology_and_seeds(kwargs: dict[str, Scalar]) -> None:
    with pytest.raises(ValueError):
        invalid_call(StreamingDataset, d.Dataset(), None, **kwargs)


def test_local_reader_survives_fork(tmp_path: Path) -> None:
    with ObjectStore(tmp_path) as store:
        resource = stored_dataset(store, count=2)
    reader = RangeReader(local_root=tmp_path)
    data = TorchDataset(resource, reader)
    with patch("premixdb._storage.os.getpid", return_value=reader._pid + 1):
        assert data.__getitems__([0, 1])[1]["input_ids"][0].item() == 1
    reader.close()


def test_single_index_fetches_only_requested_tokens_and_coalesces_masks(tmp_path: Path) -> None:
    with ObjectStore(tmp_path) as store:
        resource = stored_dataset(store)
    reader = RangeReader(local_root=tmp_path)
    reads = Mock(wraps=reader._read_bytes)
    reader._read_bytes = reads
    try:
        data = TorchDataset(resource, reader)
        reads.reset_mock()
        assert data[130]["input_ids"].tolist() == [130, 2**32 - 1]
        assert reads.call_count == 3
        assert sorted(
            (call.args[0].start, call.args[0].end)
            for call in reads.call_args_list
            if call.args[0].object != resource.sequences[1].object
        ) == [(260, 262), (1040, 1048)]
        reads.reset_mock()
        assert data[131]["input_ids"][0].item() == 131
        assert reads.call_count == 2  # index page is retained, even with the byte cache disabled
    finally:
        reader.close()


@pytest.mark.parametrize("count", [0, 1, 127, 128, 129, 513])
def test_torch_page_boundaries_and_empty_datasets(tmp_path: Path, count: int) -> None:
    from torch.utils.data import DataLoader

    with p.PremixDB(storage=tmp_path) as db:
        sources = [p.Source("a", "a" * (count * 4 - 1))] if count else []
        dataset = (
            db.corpus("boundaries", sources)
            .query()
            .dataset(tokenizer=p.ByteTokenizer(), sequence_length=4)
        )
        assert dataset.status == p.ExecutionStatus.PENDING
        data = dataset.torch()
        assert dataset.status == p.ExecutionStatus.COMPLETED
        assert len(data) == count
        batches = list(DataLoader(data, batch_size=137))
        assert sum(len(batch["input_ids"]) for batch in batches) == count
        if count:
            assert data[-1]["input_ids"].tolist() == [97, 97, 97, 256]
            indices = [*range(0, count, 128), -1, 0]
            assert [row["input_ids"].tolist() for row in data.__getitems__(indices)] == [
                dataset[index].tokens for index in indices
            ]
            assert len(data._pages) <= 4
        else:
            assert list(dataset.torch(streaming=True)) == []
        for index in (-count - 1, count):
            with pytest.raises(IndexError):
                data[index]
        for index in (True, 0.0, "0"):
            with pytest.raises(TypeError):
                data[index]


@pytest.mark.parametrize("tokenizer", [None, p.ByteTokenizer()])
def test_public_torch_reopens_read_only_and_matches_all_sequence_fields(
    tmp_path: Path, tokenizer: dataset_pb.Tokenizer | BytePolicy
) -> None:
    from torch.utils.data import DataLoader

    with p.PremixDB(storage=tmp_path) as db:
        dataset = (
            db.corpus(
                "reopen",
                [p.Source("a", ("a" if tokenizer is not None else " hello") * (131 * 8 - 1))],
            )
            .query()
            .dataset(tokenizer=tokenizer, sequence_length=8)
        )
        expected = [
            dict(
                input_ids=sequence.tokens,
                attention_mask=[int(value) for value in sequence.attention_mask],
                labels=[
                    token if mask else -100 for token, mask in zip(sequence.tokens, sequence.mask)
                ],
            )
            for sequence in dataset
        ]
        dataset_id = dataset.id
        assert len(expected) == 131
    with p.PremixDB(storage=tmp_path, read_only=True) as db:
        reopened = db._dataset(dataset_id)
        data = pickle.loads(pickle.dumps(reopened.torch()))
        streams = [
            reopened.torch(streaming=True, seed=7, rank=rank, world_size=3) for rank in range(3)
        ]
    actual = [
        {name: batch[name][index].tolist() for name in batch}
        for batch in DataLoader(data, batch_size=137)
        for index in range(len(batch["input_ids"]))
    ]
    assert actual == expected
    assert Counter(
        tuple(item["input_ids"].tolist()) for stream in streams for item in stream
    ) == Counter(tuple(item["input_ids"]) for item in expected)
    data.reader.close()
    streams[0].data.reader.close()


@pytest.mark.parametrize("mode", ["plain", "batched", "streaming"])
@pytest.mark.parametrize("corruption", ["missing", "reordered", "duplicate"])
def test_all_readers_reject_corrupt_sequence_pages(
    tmp_path: Path, mode: str, corruption: str
) -> None:
    with ObjectStore(tmp_path) as store:
        resource = stored_dataset(store, count=2)
        reader = RangeReader(local_root=tmp_path)
        batch = d.SequenceBatch.FromString(reader.read(resource.sequences[0]))
        if corruption == "missing":
            del batch.sequences[-1]
        elif corruption == "reordered":
            batch.sequences.reverse()
        else:
            batch.sequences[1].ordinal = 0
        encoded = batch.SerializeToString()
        resource.sequences[0].CopyFrom(span(store.put("dataset", encoded), encoded))
    try:
        with pytest.raises(p.ExecutionError, match="incomplete or out of order"):
            if mode == "plain":
                _read_page(resource, reader, 0)
            elif mode == "batched":
                TorchDataset(resource, reader).__getitems__([0, 1])
            else:
                list(StreamingDataset(resource, reader))
    finally:
        reader.close()


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("tokens", struct.pack("<I", 1), "invalid length"),
        ("attention_mask", bytes([1]), "invalid length"),
        ("loss_mask", bytes([1]), "invalid length"),
        ("attention_mask", bytes([1, 2]), "invalid stored token mask"),
        ("loss_mask", bytes([1, 2]), "invalid stored token mask"),
    ],
)
def test_torch_rejects_invalid_token_lengths_and_masks(
    tmp_path: Path, field: str, value: bytes, message: str
) -> None:
    with ObjectStore(tmp_path) as store:
        resource = stored_dataset(store, count=1)
        reader = RangeReader(local_root=tmp_path)
        batch = d.SequenceBatch.FromString(reader.read(resource.sequences[0]))
        getattr(batch.sequences[0], field).CopyFrom(span(store.put("dataset", value), value))
        encoded = batch.SerializeToString()
        resource.sequences[0].CopyFrom(span(store.put("dataset", encoded), encoded))
    try:
        with pytest.raises(ValueError, match=message):
            TorchDataset(resource, reader)[0]
    finally:
        reader.close()
