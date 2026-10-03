"""Generated cases for reproducibility and distributed reader invariants."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from premixdb._reader import Topology, permutation
from premixdb.engine.identity import CodeVersion
from premixdb.engine.plans import filter_documents, query_identity

DIGESTS = st.binary(min_size=32, max_size=32).map(bytes.hex)
CODE = CodeVersion("local://property-tests", "a" * 40, "09" * 32)


@given(st.lists(DIGESTS, min_size=1, max_size=20))
def test_snapshot_identity_ignores_order_and_duplicate_inputs(snapshots: list[str]) -> None:
    expected = query_identity(snapshots, [], CODE)
    assert query_identity(list(reversed(snapshots)) + snapshots, [], CODE) == expected


@given(st.lists(DIGESTS, max_size=30))
def test_document_selection_is_a_set(documents: list[str]) -> None:
    step = filter_documents(documents)
    assert filter_documents(list(reversed(documents)) + documents) == step
    assert step.members == frozenset(documents)


@given(count=st.integers(1, 300), seed=st.integers(0, 2**64 - 1))
def test_shuffle_is_a_bijection_for_arbitrary_sizes_and_seeds(count: int, seed: int) -> None:
    shuffled = [permutation(index, count, seed) for index in range(count)]
    assert sorted(shuffled) == list(range(count))


@given(count=st.integers(0, 300), ranks=st.integers(1, 8), workers=st.integers(1, 8))
def test_distributed_partitions_cover_every_ordinal_exactly_once(
    count: int, ranks: int, workers: int
) -> None:
    ordinals = []
    for rank in range(ranks):
        for worker in range(workers):
            first, stride = Topology(rank, ranks, worker, workers)._partition()
            ordinals.extend(range(first, count, stride))
    assert sorted(ordinals) == list(range(count))
