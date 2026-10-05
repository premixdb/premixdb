"""Content holdouts survive sampling, independent packing, reopening, and training reads."""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from _type_support import coordinator, invalid_call, tokenizer_packing, wordpiece_tokenizer
from test_training_reads import stored_dataset

import premixdb as p
from premixdb.schemas.ids import _decode_id
from premixdb.storage.objects import ObjectStore
from premixdb.training.torch import StreamingDataset, TorchDataset
from premixdb.v1 import data_mixture_pb2 as d


def sources() -> list[p.Source]:
    return [p.Source(str(i), f"hello world {i} " * (1 + i % 7)) for i in range(160)] + [
        p.Source("duplicate-a", "identical captured content" * 20),
        p.Source("duplicate-b", "identical captured content" * 20),
    ]


def test_split_views_do_not_expose_split_selection(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path, progress=False) as db:
        dataset = (
            db.Corpus("split-surface", [p.Source("a", "hello")])
            .query()
            .mix(splits=p.Splits(train=0.8, validation=0.1, test=0.1), tokenizer=p.ByteTokenizer())[
                0
            ]
        )
        with patch.object(
            coordinator(db), "_planned_dataset_profile", side_effect=AssertionError("profile")
        ):
            for view in (dataset.train, dataset.validation, dataset.test):
                assert view.status is p.ExecutionStatus.PENDING
                for name in ("train", "validation", "test"):
                    assert name not in dir(view)
                    assert not hasattr(view, name)
                    with pytest.raises(AttributeError):
                        getattr(view, name)
                with pytest.raises(ValueError, match="cannot be split again"):
                    invalid_call(p.DatasetSplit, view, "train")
        assert dataset.status is p.ExecutionStatus.PENDING


@pytest.mark.parametrize(
    "value,error",
    [
        ({"train": 0.8, "validation": 0.1, "test": 0.1}, TypeError),
        ((0.8, 0.1, 0.1), TypeError),
        (d.Splits(), ValueError),
        (d.Splits(train=1), ValueError),
        (d.Splits(train=0.8, validation=0.1, test=0.2), ValueError),
        (d.Splits(train=float("nan"), validation=0, test=0), ValueError),
        (d.Splits(train=float("inf"), validation=0, test=0), ValueError),
        (d.Splits(train=-0.1, validation=0.1, test=1), ValueError),
    ],
)
def test_split_validation_precedes_execution(value: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        invalid_call(p.mix, b"q" * 32, splits=value)


def test_split_defaults_typed_requests_and_detached_policies(tmp_path: Path) -> None:
    expected = p.Splits(train=0.8, validation=0.1, test=0.1, seed=0)
    assert p.mix(b"q" * 32).splits == expected
    policy = p.Splits(train=0.9, validation=0.05, test=0.05, seed=42)
    request = p.mix(b"q" * 32, splits=policy)
    policy.train = 0
    restored = d.CreateMixRequest.FromString(request.SerializeToString())
    assert restored.splits.train == 0.9
    assert restored.splits.seed == 42
    with p.PremixDB(storage=tmp_path, progress=False) as db:
        query = db.Corpus("lazy", sources()).query()
        service = coordinator(db)
        with patch.object(service, "_query", side_effect=AssertionError("executed")):
            mixture = query.mix()
            dataset = mixture[0]
            assert dataset._recipe.splits == expected
            assert isinstance(dataset.train, p.DatasetSplit)
            assert query.status == dataset.status == p.ExecutionStatus.PENDING
        raw = service.CreateMix(d.CreateMixRequest(query_id=_decode_id(query.id)))
        assert service.GetMix(d.GetMixRequest(id=raw.id)).mix.splits == expected
        raw_request = d.CreateMixRequest(query_id=_decode_id(query.id), splits=d.Splits(train=1))
        with pytest.raises(ValueError, match="requires"):
            service.CreateMix(raw_request)


@pytest.mark.parametrize("model", [False, True], ids=["byte", "model"])
@pytest.mark.parametrize("process_workers", [0, 2])
def test_splits_pack_independently_and_reopen(
    tmp_path: Path, model: bool, process_workers: int
) -> None:
    tokenizer = wordpiece_tokenizer() if model else p.ByteTokenizer()
    captured = sources()
    policy = p.Splits(train=0.9, validation=0.05, test=0.05, seed=42)
    with p.PremixDB(storage=tmp_path, process_workers=process_workers, progress=False) as db:
        query = db.Corpus("split", captured).query()
        dataset = query.mix(
            splits=policy,
            tokenizer=tokenizer,
            packing=tokenizer_packing(tokenizer),
            sequence_length=7,
        )[0]
        previews = [
            view.preview(limit=2) for view in (dataset.train, dataset.validation, dataset.test)
        ]
        assert dataset.status == p.ExecutionStatus.PENDING
        planned = dataset.profile()
        parts = [dataset.train, dataset.validation, dataset.test]
        profiles = [view.profile() for view in parts]
        assert sum(profile.sequences for profile in profiles) == planned.sequences
        assert sum(profile.source_documents for profile in profiles) == len(captured)
        lookup = {row["id"]: row["text"] for row in query.preview(limit=1000, max_characters=10000)}
        membership = []
        all_tokens = []
        for view, before, profile in zip(parts, previews, profiles, strict=True):
            assert len(view) == profile.sequences
            sequences = list(view)
            assert [seq.ordinal for seq in sequences] == list(range(len(view)))
            assert view.preview(limit=2) == before
            assert sequences
            assert view[-1].tokens == sequences[-1].tokens
            state = view._reader(seed=42)
            first = next(state)
            resumed = list(view._reader(seed=42, checkpoint=state.checkpoint()))
            assert sorted([first.ordinal, *(seq.ordinal for seq in resumed)]) == list(
                range(len(view))
            )
            with pytest.raises(ValueError, match="incompatible"):
                dataset._reader(seed=42, checkpoint=state.checkpoint())
            membership.append({lookup[id] for seq in sequences for id in seq.document_ids()})
            all_tokens.extend(seq.tokens for seq in sequences)
            torch_data = view.torch()
            assert len(torch_data) == len(view)
            assert torch_data[-1]["input_ids"].tolist() == sequences[-1].tokens
        assert all(not membership[a] & membership[b] for a, b in ((0, 1), (0, 2), (1, 2)))
        assert [seq.tokens for seq in dataset] == all_tokens
        assert planned == dataset.profile()
        identity = dataset.id
        expected = [view.profile() for view in parts]
    with p.PremixDB(storage=tmp_path, read_only=True, progress=False) as db:
        dataset = db._dataset(identity)
        assert dataset._proto.splits == policy
        assert [
            view.profile() for view in (dataset.train, dataset.validation, dataset.test)
        ] == expected
        assert dataset.validation.preview(limit=2) == previews[1]
        adapter = dataset.validation.torch()
        assert len(adapter) == expected[1].sequences
    assert adapter[0]["input_ids"].tolist() == all_tokens[expected[0].sequences]


def test_sampled_candidates_share_holdouts_and_exclude_heldout_content(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path, progress=False) as db:
        query = db.Corpus("candidates", sources()).query()
        mixture = query.mix(
            splits=p.Splits(train=0.8, validation=0.1, test=0.1, seed=42),
            weights=p.RegMix(),
            n_candidates=2,
            tokens=7000,
            replacement=True,
            tokenizer=p.ByteTokenizer(),
            sequence_length=32,
        )
        first, second = mixture[0], mixture[1]
        preview = first.validation.preview()
        first.wait()
        assert first.validation.preview() == preview
        first_train = {id for seq in first.train for id in seq.document_ids()}
        second_train = {id for seq in second.train for id in seq.document_ids()}
        heldout = {id for seq in first.validation for id in seq.document_ids()} | {
            id for seq in first.test for id in seq.document_ids()
        }
        assert not first_train & heldout
        assert not second_train & heldout
        assert first.train.profile().planned_content_tokens == 7000
        for name in ("validation", "test"):
            a, b = getattr(first, name), getattr(second, name)
            assert [seq.tokens for seq in a] == [seq.tokens for seq in b]
            assert a.profile() == b.profile()
        changed_sampling = query.mix(
            splits=p.Splits(train=0.8, validation=0.1, test=0.1, seed=42),
            tokens=128,
            seed=999,
            tokenizer=p.ByteTokenizer(),
            sequence_length=32,
        )[0]
        assert [seq.tokens for seq in changed_sampling.test] == [seq.tokens for seq in first.test]


def test_zero_splits_and_split_identity(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path, progress=False) as db:
        query = db.Corpus("zero", sources()).query()
        full = query.mix(
            splits=p.Splits(train=1, validation=0, test=0),
            tokenizer=p.ByteTokenizer(),
            sequence_length=8,
        )[0]
        default = query.mix(tokenizer=p.ByteTokenizer(), sequence_length=8)[0]
        other_seed = query.mix(
            splits=p.Splits(train=0.8, validation=0.1, test=0.1, seed=1),
            tokenizer=p.ByteTokenizer(),
            sequence_length=8,
        )[0]
        assert len({full.id, default.id, other_seed.id}) == 3
        assert full.validation.preview() == []
        assert full.validation.profile().sequences == 0
        assert len(full.validation) == 0
        assert list(full.test) == list(full.test.torch(streaming=True)) == []
        with pytest.raises(IndexError):
            full.validation[0]
        assert len(full.train) == len(full)
        assert full.train.id != full.id


def test_duplicate_content_across_corpora_and_dropped_split_tails(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path, progress=False) as db:
        original = sources()
        first = db.Corpus("original", original)
        second = db.Corpus(
            "copies", [p.Source(str(i), row.text) for i, row in enumerate(original[:8])]
        )
        query = first.union(second).query()
        dataset = query.mix(
            tokenizer=p.ByteTokenizer(),
            sequence_length=13,
            packing=p.Concat(separator=256),
        )[0]
        lookup = {row["id"]: row["text"] for row in query.preview(limit=1000, max_characters=10000)}
        content = []
        for view in (dataset.train, dataset.validation, dataset.test):
            profile = view.profile()
            assert profile.padding_tokens == 0
            assert profile.dropped_tokens < 13
            content.append({lookup[id] for seq in view for id in seq.document_ids()})
        assert all(not content[a] & content[b] for a, b in ((0, 1), (0, 2), (1, 2)))
        assert dataset.profile().dropped_tokens == sum(
            view.profile().dropped_tokens
            for view in (dataset.train, dataset.validation, dataset.test)
        )


def test_training_preview_reuses_pool_without_encoding_holdouts(tmp_path: Path) -> None:
    from premixdb.engine.datasets import HuggingFaceTokenizer

    with p.PremixDB(storage=tmp_path, progress=False) as db:
        dataset = (
            db.Corpus("bounded", sources())
            .query()
            .mix(
                tokenizer=wordpiece_tokenizer(),
                tokens=32,
                sequence_length=4,
                packing=tokenizer_packing(wordpiece_tokenizer()),
            )[0]
        )
        with patch.object(
            HuggingFaceTokenizer,
            "encode_with_offsets",
            side_effect=AssertionError("encoded holdout"),
        ):
            examples = dataset.train.preview(limit=1)
        assert len(examples) == 1
        assert dataset.status == p.ExecutionStatus.PENDING
        dataset.wait()
        assert dataset.train.preview(limit=1) == examples


def test_legacy_unsplit_dataset_train_view_remains_readable(tmp_path: Path) -> None:
    from premixdb.schemas import requests

    with p.PremixDB(storage=tmp_path, progress=False) as db:
        query = db.Corpus("legacy", [p.Source("a", "hello world")]).query()
        identity = (
            coordinator(db)
            .CreateDataset(
                requests.dataset(query.id, tokenizer=p.ByteTokenizer(), sequence_length=4)
            )
            .id
        )
        dataset = db._dataset(identity)
        assert not dataset._proto.HasField("splits")
        assert [seq.tokens for seq in dataset.train] == [seq.tokens for seq in dataset]
        assert dataset.train.profile() == dataset.profile()
        with pytest.raises(ValueError, match="no validation split"):
            _ = dataset.validation


@pytest.mark.parametrize("window", [(3, 129), (127, 258), (128, 128), (256, 385)])
def test_range_adapters_filter_boundary_pages_for_all_workers(
    tmp_path: Path, window: tuple[int, int]
) -> None:
    with ObjectStore(tmp_path) as store:
        resource = stored_dataset(store, count=385)
    with p.RangeReader(local_root=tmp_path) as reader:
        data = TorchDataset(resource, reader, window=window)
        assert len(data) == window[1] - window[0]
        expected = list(range(*window))
        assert [int(item["input_ids"][0]) for item in data] == expected
        restored = pickle.loads(pickle.dumps(data))
        assert len(restored) == len(data)
        if expected:
            assert int(restored[-1]["input_ids"][0]) == expected[-1]
        values = []
        for rank in range(2):
            for worker in range(3):
                stream = StreamingDataset(
                    resource, reader, window=window, seed=42, rank=rank, world_size=2
                )
                with patch(
                    "premixdb.training.torch.get_worker_info",
                    return_value=SimpleNamespace(id=worker, num_workers=3),
                ):
                    values.extend(int(item["input_ids"][0]) for item in stream)
        assert sorted(values) == expected


@pytest.mark.integration
@pytest.mark.parametrize("streaming", [False, True])
def test_split_adapters_survive_spawned_loader_workers(tmp_path: Path, streaming: bool) -> None:
    from torch.utils.data import DataLoader

    with ObjectStore(tmp_path) as store:
        resource = stored_dataset(store, count=385)
    with p.RangeReader(local_root=tmp_path) as reader:
        data = (
            StreamingDataset(resource, reader, window=(127, 258), seed=42)
            if streaming
            else TorchDataset(resource, reader, window=(127, 258))
        )
        restored = pickle.loads(pickle.dumps(data))
        loader = DataLoader(restored, batch_size=16, num_workers=2, multiprocessing_context="spawn")
        values = [value for batch in loader for value in batch["input_ids"][:, 0].tolist()]
        assert sorted(values) == list(range(127, 258))
