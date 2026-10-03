"""Invalid reader options fail before a lazy candidate launches computation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from _type_support import invalid_call

import premixdb as p
from premixdb._typing import JSON, Scalar


@pytest.fixture
def candidate(tmp_path: Path) -> Iterator[p.Dataset]:
    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.corpus("options", [p.Source("a", "hello world")])
        yield snapshot.query().dataset(tokenizer=p.ByteTokenizer(), sequence_length=4)


@pytest.mark.parametrize(
    "options,error",
    [
        (dict(streaming=1), TypeError),
        (dict(seed=42), ValueError),
        (dict(epoch=1), ValueError),
        (dict(rank=0, world_size=1), ValueError),
        (dict(streaming=True, seed=True), ValueError),
        (dict(streaming=True, epoch=-1), ValueError),
        (dict(streaming=True, rank=0), ValueError),
        (dict(streaming=True, rank=1, world_size=1), ValueError),
    ],
)
def test_invalid_torch_options_do_not_materialize_candidate(
    candidate: p.Dataset, options: dict[str, Scalar], error: type[BaseException]
) -> None:
    assert candidate.status == p.ExecutionStatus.PENDING
    with patch.object(p.Dataset, "wait", side_effect=AssertionError("started packing")):
        with pytest.raises(error):
            invalid_call(
                candidate.torch,
                streaming=options.get("streaming", False),
                seed=options.get("seed", 0),
                epoch=options.get("epoch", 0),
                rank=options.get("rank"),
                world_size=options.get("world_size"),
            )
    assert candidate.status == p.ExecutionStatus.PENDING


def test_invalid_reader_topology_does_not_materialize_candidate(candidate: p.Dataset) -> None:
    with patch.object(p.Dataset, "wait", side_effect=AssertionError("started packing")):
        with pytest.raises(ValueError, match="topology"):
            candidate.reader(topology=p.Topology(rank=2, world_size=1))


def test_invalid_streaming_seed_does_not_discover_distributed_rank() -> None:
    from premixdb._torch import streaming_topology

    with patch(
        "premixdb._torch.torch.distributed.is_available",
        side_effect=AssertionError("discovered rank"),
    ):
        with pytest.raises(ValueError, match="uint64"):
            streaming_topology(seed=-1, epoch=0, rank=None, world_size=None)


@pytest.mark.parametrize(
    "changes",
    [
        {"version": 2},
        {"version": True},
        {"version": 1.0},
        {"dataset": "other"},
        {"topology": [1, 1, 0, 1]},
        {"topology": [False, 1, 0, 1]},
        {"topology": [0, True, 0, 1]},
        {"topology": [0.0, 1, 0, 1]},
        {"shuffle_seed": 7},
        {"next_ordinal": "missing"},
        {"next_ordinal": False},
        {"next_ordinal": 0.0},
        {"next_ordinal": -1},
        {"next_ordinal": "0"},
    ],
)
def test_incompatible_checkpoint_does_not_materialize_candidate(
    candidate: p.Dataset, changes: dict[str, JSON]
) -> None:
    checkpoint: dict[str, JSON] = dict(
        version=1,
        dataset=candidate.id,
        topology=list(p.Topology()._checkpoint_values()),
        next_ordinal=0,
    )
    checkpoint.update(changes)
    if changes == {"next_ordinal": "missing"}:
        del checkpoint["next_ordinal"]
    with patch.object(p.Dataset, "wait", side_effect=AssertionError("started packing")):
        with pytest.raises(ValueError, match="checkpoint"):
            invalid_call(candidate.reader, checkpoint=checkpoint)


@pytest.mark.parametrize("seed", [False, 0.0])
def test_checkpoint_seed_requires_an_integer_before_materialization(
    candidate: p.Dataset, seed: JSON
) -> None:
    checkpoint: dict[str, JSON] = dict(
        version=1,
        dataset=candidate.id,
        topology=list(p.Topology()._checkpoint_values()),
        shuffle_seed=seed,
        next_ordinal=0,
    )
    with patch.object(p.Dataset, "wait", side_effect=AssertionError("started packing")):
        with pytest.raises(ValueError, match="checkpoint"):
            invalid_call(candidate.reader, seed=0, checkpoint=checkpoint)


@pytest.mark.parametrize("checkpoint", [[], False, "checkpoint"])
def test_checkpoint_requires_a_dictionary_before_materialization(
    candidate: p.Dataset, checkpoint: JSON
) -> None:
    with patch.object(p.Dataset, "wait", side_effect=AssertionError("started packing")):
        with pytest.raises(ValueError, match="checkpoint"):
            invalid_call(candidate.reader, checkpoint=checkpoint)


@pytest.mark.parametrize("ordinal", [0, 2])
def test_checkpoint_from_another_partition_does_not_materialize_candidate(
    candidate: p.Dataset, ordinal: int
) -> None:
    topology = p.Topology(rank=1, world_size=2)
    checkpoint: p.Checkpoint = dict(
        version=1,
        dataset=candidate.id,
        topology=topology._checkpoint_values(),
        next_ordinal=ordinal,
    )
    with patch.object(p.Dataset, "wait", side_effect=AssertionError("started packing")):
        with pytest.raises(ValueError, match="checkpoint"):
            candidate.reader(topology=topology, checkpoint=checkpoint)


@pytest.mark.parametrize("seed", [None, 0])
def test_exhausted_checkpoint_does_not_materialize_candidate(
    candidate: p.Dataset, seed: int | None
) -> None:
    checkpoint: p.Checkpoint = dict(
        version=1,
        dataset=candidate.id,
        topology=p.Topology()._checkpoint_values(),
        next_ordinal=None,
    )
    if seed is not None:
        checkpoint["shuffle_seed"] = seed
    with patch.object(p.Dataset, "wait", side_effect=AssertionError("started packing")):
        reader = candidate.reader(checkpoint=checkpoint, seed=seed)
        assert list(reader) == []
        assert reader.checkpoint() == checkpoint
    assert candidate.status == p.ExecutionStatus.PENDING
