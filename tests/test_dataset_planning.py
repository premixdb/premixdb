"""Packing and domain errors fail before tokenization or mixture draw construction."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pytest
from blake3 import blake3
from tokenizers import Tokenizer, models

from premixdb.engine.datasets import HuggingFaceTokenizer
from premixdb.engine.identity import CodeVersion
from premixdb.engine.mixtures import MixturePool
from premixdb.engine.queries import CorpusIndex, Query
from premixdb.engine.snapshots import Snapshot


@pytest.fixture
def ready(tmp_path: Path) -> tuple[Query, HuggingFaceTokenizer]:
    asset = tmp_path / "tokenizer.json"
    Tokenizer(models.WordLevel({"[UNK]": 0, "text": 1}, unk_token="[UNK]")).save(str(asset))
    tokenizer = HuggingFaceTokenizer(asset, blake3(asset.read_bytes()).hexdigest(), 1024)
    code = CodeVersion("local://test", "a" * 40, "09" * 32)
    snapshot = Snapshot("01" * 16, [("a", "text")], code)
    return CorpusIndex([snapshot]).execute([], code), tokenizer


INVALID_PACKING = [(0, None, None), (4, -1, None), (4, None, 2**32)]


@pytest.mark.parametrize("packing", INVALID_PACKING)
def test_invalid_packing_does_not_encode_query_documents(
    ready: tuple[Query, HuggingFaceTokenizer], packing: tuple[int, int | None, int | None]
) -> None:
    query, tokenizer = ready
    with (
        patch("premixdb.engine.token_cache.token_pool", side_effect=AssertionError("encoded")),
        pytest.raises(ValueError),
    ):
        query.dataset(*packing, tokenizer=tokenizer)


@pytest.mark.parametrize("case", ["missing", "conflicting", "empty"])
def test_invalid_domains_do_not_build_a_token_pool(
    ready: tuple[Query, HuggingFaceTokenizer], case: Literal["missing", "conflicting", "empty"]
) -> None:
    query, tokenizer = ready
    assignments = {} if case == "missing" else {query.row(0).id: "" if case == "empty" else "a"}
    field = "object.uri" if case == "conflicting" else ""
    with (
        patch("premixdb.engine.token_cache.token_pool", side_effect=AssertionError("encoded")),
        pytest.raises(ValueError),
    ):
        MixturePool(query, field, assignments, tokenizer)


@pytest.mark.parametrize("packing", INVALID_PACKING)
@pytest.mark.parametrize("kind", ["dataset", "profile", "geometry"])
def test_invalid_packing_does_not_construct_mixture_draws(
    ready: tuple[Query, HuggingFaceTokenizer],
    packing: tuple[int, int | None, int | None],
    kind: Literal["dataset", "profile", "geometry"],
) -> None:
    query, _ = ready
    pool = MixturePool(query, "object.uri", {})
    compute = {"dataset": pool.dataset, "profile": pool.profile, "geometry": pool.geometry}[kind]
    with (
        patch.object(pool, "_draw", side_effect=AssertionError("drew samples")),
        pytest.raises(ValueError),
    ):
        compute({"a": 1}, "04" * 32, 0, True, None, *packing)


def test_summary_mutation_cannot_change_dataset_counts(
    ready: tuple[Query, HuggingFaceTokenizer],
) -> None:
    query, _ = ready
    dataset = query.dataset(4, 256, 257)
    expected = dataset.summary()
    modified = dataset.summary()
    modified["input"]["bytes"] = 999
    modified["sequences"] = 999
    assert dataset.summary() == expected
    assert len(dataset) == 2
    assert expected["input"] == dict(documents=1, bytes=4, characters=4)
    assert expected["content_tokens"] == 4
    assert expected["separator_tokens"] == 1
    assert expected["padding_tokens"] == 3
    assert expected["output_tokens"] == 8
