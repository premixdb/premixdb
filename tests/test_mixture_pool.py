"""Mixture exposure uses the pool's pinned encoding without retaining its tokenizer."""

import gc
import weakref
from pathlib import Path

import pytest
from blake3 import blake3

from premixdb.engine.datasets import HuggingFaceTokenizer
from premixdb.engine.identity import CodeVersion
from premixdb.engine.mixtures import MixturePool
from premixdb.engine.queries import CorpusIndex, Query
from premixdb.engine.snapshots import Snapshot


@pytest.fixture
def query() -> Query:
    code = CodeVersion("local://mixture-pool", "a" * 40, "09" * 32)
    snapshot = Snapshot("01" * 16, [("a", "hello world"), ("b", "playing"), ("empty", "")], code)
    return CorpusIndex([snapshot]).execute([], code)


def tokenizer() -> HuggingFaceTokenizer:
    data = (Path(__file__).parent / "fixtures" / "wordpiece.json").read_bytes()
    return HuggingFaceTokenizer.from_bytes(data, blake3(data).hexdigest(), 1024)


@pytest.mark.parametrize("length", [1, 3, 8])
@pytest.mark.parametrize("padding", [None, 3])
def test_model_token_exposure_matches_planning_across_packing_tails(
    query: Query, length: int, padding: int | None
) -> None:
    pool = MixturePool(query, "object.uri", {}, tokenizer())
    assert pool.inventory() == {"a": 2, "b": 2, "empty": 0}
    allocations = {"a": 3, "b": 1, "empty": 0}
    dataset = pool.dataset(allocations, "04" * 32, 42, True, 2, length, 2, padding)
    profile = pool.profile(allocations, "04" * 32, 42, True, 2, length, 2, padding)
    exposure = pool.exposure(dataset)
    assert exposure == profile["stratum_tokens"]
    assert sum(exposure.values()) == profile["content_tokens"]
    assert exposure["empty"] == 0
    assert len(dataset) == profile["sequences"]


def test_pool_releases_tokenizer_after_encoding_and_still_builds_datasets(query: Query) -> None:
    model = tokenizer()
    definition = model.definition
    reference = weakref.ref(model)
    pool = MixturePool(query, "object.uri", {}, model)
    del model
    gc.collect()
    assert reference() is None
    dataset = pool.dataset({"a": 2, "b": 2, "empty": 0}, "04" * 32, 0, False, None, 8, 2, 3)
    assert dataset.tokenizer_definition == definition
    assert pool.exposure(dataset) == {"a": 2, "b": 2, "empty": 0}


def test_byte_and_model_pools_reject_each_others_datasets(query: Query) -> None:
    byte_pool = MixturePool(query, "object.uri", {})
    model_pool = MixturePool(query, "object.uri", {}, tokenizer())
    byte_dataset = query.dataset(4, None, None)
    model_dataset = model_pool.dataset(
        {"a": 2, "b": 2, "empty": 0}, "04" * 32, 0, False, None, 8, 2, 3
    )
    with pytest.raises(ValueError, match="does not belong"):
        model_pool.exposure(byte_dataset)
    with pytest.raises(ValueError, match="does not belong"):
        byte_pool.exposure(model_dataset)
