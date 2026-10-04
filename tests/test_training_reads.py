"""Coalesced verified IO, batched tensors, and distributed streaming coverage."""

from __future__ import annotations

import json
import pickle
import struct
import tracemalloc
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from _type_support import invalid_call, tokenizer_packing, wordpiece_tokenizer
from blake3 import blake3

import premixdb as p
from premixdb import RangeReader
from premixdb._reader import permutation
from premixdb._sequences import read_page
from premixdb._typing import Scalar
from premixdb.execution.storage import ObjectStore
from premixdb.v1 import data_mixture_pb2 as d
from premixdb.v1 import status_pb2 as status
from premixdb.v1.storage_pb2 import ObjectRef, SpanRef


def span(obj: ObjectRef, data: bytes, start: int = 0, end: int | None = None) -> SpanRef:
    end = len(data) if end is None else end
    return SpanRef(object=obj, start=start, end=end, blake3_digest=blake3(data[start:end]).digest())


def stored_dataset(store: ObjectStore, count: int = 132) -> d.Dataset:
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


@pytest.mark.parametrize("count", [128, 129])
@pytest.mark.parametrize("topology", [p.Topology(), p.Topology(rank=1, world_size=3)])
def test_shuffled_reader_reuses_whole_pages_and_resumes_exactly(
    tmp_path: Path, count: int, topology: p.Topology
) -> None:
    with ObjectStore(tmp_path) as store:
        resource = stored_dataset(store, count)
        resource.id = b"d" * 32
        resource.status = status.STATUS_COMPLETED
    first, stride = topology._partition()
    expected = [permutation(i, count, 7) for i in range(first, count, stride)]

    def page_reads(ordinals: list[int]) -> int:
        pages = [ordinal // 128 for ordinal in ordinals]
        return sum(i == 0 or page != pages[i - 1] for i, page in enumerate(pages))

    with p.PremixDB(storage=tmp_path, read_only=True) as db:
        dataset = p.Dataset(db, resource)
        with patch.object(db._object_reader, "read", wraps=db._object_reader.read) as reads:
            reader = dataset._reader(seed=7, topology=topology)
            prefix = [next(reader).ordinal for _ in range(7)]
            checkpoint = json.loads(json.dumps(reader.checkpoint()))
            remaining = [sequence.ordinal for sequence in reader]
            assert prefix + remaining == expected
            assert reads.call_count == page_reads(expected)
            reads.reset_mock()
            resumed = dataset._reader(seed=7, topology=topology, checkpoint=checkpoint)
            assert [sequence.ordinal for sequence in resumed] == remaining
            assert reads.call_count == page_reads(remaining)
            assert (
                list(dataset._reader(seed=7, topology=topology, checkpoint=reader.checkpoint()))
                == []
            )


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


def test_duplicate_spans_share_bounded_bytes_and_keep_digest_checks(tmp_path: Path) -> None:
    with ObjectStore(tmp_path) as store, RangeReader(local_root=tmp_path) as reader:
        data = b"start" + b"x" * 1_000_000 + b"end"
        obj = store.put("dataset", data)
        reference = span(obj, data, 5, len(data) - 3)
        tracemalloc.start()
        try:
            values = reader.read_many([span(obj, data), *[reference] * 64])
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 4_000_000
        assert values == [data, *[data[5:-3]] * 64]
        conflicting = SpanRef()
        conflicting.CopyFrom(reference)
        conflicting.blake3_digest = b"x" * 32
        with pytest.raises(ValueError, match="integrity"):
            reader.read_many([reference, reference, conflicting])
        path = tmp_path / "dataset/objects" / obj.blake3_digest.hex()
        path.write_bytes(b"corrupt" * (len(data) // 7) + b"x" * (len(data) % 7))
        with pytest.raises(ValueError, match="integrity"):
            reader.read_many([reference] * 64)


def test_batch_loading_matches_individual_reads_and_coalesces_file_reads(tmp_path: Path) -> None:
    from premixdb._torch import TorchDataset

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


def test_tensor_mutations_do_not_change_duplicate_rows_labels_or_future_reads(
    tmp_path: Path,
) -> None:
    from premixdb._torch import TorchDataset

    with ObjectStore(tmp_path) as store, RangeReader(local_root=tmp_path) as reader:
        data = TorchDataset(stored_dataset(store, count=2), reader)
        first, other, duplicate = data.__getitems__([1, 0, 1])
        first["input_ids"].zero_()
        assert first["labels"].tolist() == [1, -100]
        first["labels"].fill_(99)
        first["attention_mask"].zero_()
        assert other["input_ids"].tolist() == [0, 2**32 - 1]
        assert duplicate["input_ids"].tolist() == [1, 2**32 - 1]
        assert duplicate["labels"].tolist() == [1, -100]
        assert duplicate["attention_mask"].tolist() == [1, 0]
        reread = data[1]
        for name, expected in duplicate.items():
            assert reread[name].equal(expected)


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
    from premixdb._torch import StreamingDataset

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

    from premixdb._torch import StreamingDataset

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
    from premixdb._torch import StreamingDataset

    with pytest.raises(ValueError):
        invalid_call(StreamingDataset, d.Dataset(), None, **kwargs)


def test_local_reader_survives_fork(tmp_path: Path) -> None:
    from premixdb._torch import TorchDataset

    with ObjectStore(tmp_path) as store:
        resource = stored_dataset(store, count=2)
    reader = RangeReader(local_root=tmp_path)
    data = TorchDataset(resource, reader)
    with patch("premixdb._storage.os.getpid", return_value=reader._pid + 1):
        assert data.__getitems__([0, 1])[1]["input_ids"][0].item() == 1
    reader.close()


def test_single_index_fetches_only_requested_tokens_and_coalesces_masks(tmp_path: Path) -> None:
    from premixdb._torch import TorchDataset

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
            db.Corpus("boundaries", sources)
            .query()
            .mix(tokenizer=p.ByteTokenizer(), sequence_length=4)[0]
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


@pytest.mark.parametrize(
    "tokenizer", [wordpiece_tokenizer(), p.ByteTokenizer()], ids=["wordpiece", "bytes"]
)
def test_public_torch_reopens_read_only_and_matches_all_sequence_fields(
    tmp_path: Path, tokenizer: d.Tokenizer
) -> None:
    from torch.utils.data import DataLoader

    with p.PremixDB(storage=tmp_path) as db:
        dataset = (
            db.Corpus(
                "reopen",
                [
                    p.Source(
                        "a",
                        ("hello " if tokenizer.HasField("hugging_face") else "a") * (131 * 8 - 1),
                    )
                ],
            )
            .query()
            .mix(tokenizer=tokenizer, packing=tokenizer_packing(tokenizer), sequence_length=8)[0]
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
    from premixdb._torch import StreamingDataset, TorchDataset

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
                read_page(resource, reader, 0)
            elif mode == "batched":
                TorchDataset(resource, reader).__getitems__([0, 1])
            else:
                list(StreamingDataset(resource, reader))
    finally:
        reader.close()


@pytest.mark.parametrize("mode", ["plain", "batched", "streaming"])
@pytest.mark.parametrize(
    "field,value,start,message",
    [
        ("tokens", struct.pack("<I", 1), 0, "invalid length"),
        ("attention_mask", bytes([1]), 0, "invalid length"),
        ("loss_mask", bytes([1]), 0, "invalid length"),
        ("attention_mask", bytes([1, 2]), 0, "invalid stored .*mask"),
        ("loss_mask", bytes([1, 2]), 0, "invalid stored .*mask"),
        ("tokens", b"\0" + struct.pack("<2I", 1, 2), 1, "not uint32 aligned"),
    ],
)
def test_all_readers_reject_invalid_token_lengths_alignment_and_masks(
    tmp_path: Path, mode: str, field: str, value: bytes, start: int, message: str
) -> None:
    from premixdb._torch import StreamingDataset, TorchDataset

    with ObjectStore(tmp_path) as store:
        resource = stored_dataset(store, count=1)
        reader = RangeReader(local_root=tmp_path)
        batch = d.SequenceBatch.FromString(reader.read(resource.sequences[0]))
        getattr(batch.sequences[0], field).CopyFrom(span(store.put("dataset", value), value, start))
        encoded = batch.SerializeToString()
        resource.sequences[0].CopyFrom(span(store.put("dataset", encoded), encoded))
    try:
        with pytest.raises(ValueError, match=message):
            if mode == "plain":
                sequence = read_page(resource, reader, 0)[0]
                getattr(sequence, "mask" if field == "loss_mask" else field)
            elif mode == "batched":
                TorchDataset(resource, reader)[0]
            else:
                list(StreamingDataset(resource, reader))
    finally:
        reader.close()


def test_sequence_values_are_detached_from_cached_storage(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        sequence = (
            db.Corpus("detached", [p.Source("a", "a")])
            .query()
            .mix(tokenizer=p.ByteTokenizer(), sequence_length=4)[0][0]
        )
        tokens, mask, attention, regions = (
            sequence.tokens,
            sequence.mask,
            sequence.attention_mask,
            sequence.spans,
        )
        tokens[0] = 0
        mask[0] = False
        attention[0] = False
        regions[0].start = 100
        assert sequence.tokens == [97, 256, 257, 257]
        assert sequence.mask == [True, True, False, False]
        assert sequence.attention_mask == [True, True, False, False]
        assert sequence.spans[0].start == 0


@pytest.mark.parametrize("read_only", [False, True])
def test_session_close_releases_owned_read_threads_and_detached_data_still_reads(
    tmp_path: Path, read_only: bool
) -> None:
    with p.PremixDB(storage=tmp_path) as writer:
        identity = (
            writer.Corpus("lifetime", [p.Source("a", "abcd")])
            .query()
            .mix(tokenizer=p.ByteTokenizer(), sequence_length=4)[0]
            .wait()
            .id
        )
    db = p.PremixDB(storage=tmp_path, read_only=read_only)
    data = db._dataset(identity).torch()
    try:
        expected = data[0]["input_ids"].tolist()
        pool = data.reader._pool
        assert pool is not None
        db.close()
        assert data.reader._pool is None
        with pytest.raises(RuntimeError, match="cannot schedule"):
            pool.submit(lambda: None)
        assert data[0]["input_ids"].tolist() == expected
        assert data.reader._pool is not pool
    finally:
        db.close()
        data.reader.close()


class FalsyReader(RangeReader):
    def __bool__(self) -> bool:
        return False


@pytest.mark.parametrize("reader_type", [RangeReader, FalsyReader])
def test_session_preserves_caller_supplied_reader(
    tmp_path: Path, reader_type: type[RangeReader]
) -> None:
    reader = reader_type(local_root=tmp_path)
    try:
        with patch.object(reader, "close", wraps=reader.close) as close:
            with p.PremixDB(storage=tmp_path, object_reader=reader) as db:
                data = (
                    db.Corpus("shared-reader", [p.Source("a", "abcd")])
                    .query()
                    .mix(tokenizer=p.ByteTokenizer(), sequence_length=4)[0]
                    .torch()
                )
                assert data.reader is reader
                assert data[0]["input_ids"].tolist() == list(b"abcd")
                pool = reader._pool
            close.assert_not_called()
            assert reader._pool is pool
            assert data[0]["input_ids"].tolist() == list(b"abcd")
    finally:
        reader.close()


def test_session_releases_reader_even_when_executor_close_fails(tmp_path: Path) -> None:
    db = p.PremixDB(storage=tmp_path)
    data = (
        db.Corpus("close-error", [p.Source("a", "abcd")])
        .query()
        .mix(tokenizer=p.ByteTokenizer(), sequence_length=4)[0]
        .torch()
    )
    try:
        assert data[0]["input_ids"].tolist() == list(b"abcd")
        assert data.reader._pool is not None
        with patch.object(db._executor, "close", side_effect=OSError("executor close failed")):
            with pytest.raises(OSError, match="executor close failed"):
                db.close()
        assert data.reader._pool is None
    finally:
        db._executor.close()
        data.reader.close()
