"""Invalid reader options fail before a lazy candidate launches computation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from _type_support import invalid_call

import premixdb as p
import premixdb as sdk
from premixdb._typing import JSON, Scalar


@pytest.fixture
def candidate(tmp_path: Path) -> Iterator[sdk.Dataset]:
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
    candidate: sdk.Dataset, options: dict[str, Scalar], error: type[BaseException]
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


def test_invalid_reader_topology_does_not_materialize_candidate(candidate: sdk.Dataset) -> None:
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
        {"dataset": "other"},
        {"topology": [1, 1, 0, 1]},
        {"shuffle_seed": 7},
        {"next_ordinal": "missing"},
    ],
)
def test_incompatible_checkpoint_does_not_materialize_candidate(
    candidate: sdk.Dataset, changes: dict[str, JSON]
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
