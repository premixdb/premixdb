"""Generated cases for reproducibility and distributed reader invariants."""

from __future__ import annotations

from fractions import Fraction
from itertools import accumulate

from hypothesis import given
from hypothesis import strategies as st

from premixdb._mixing import allocations
from premixdb._reader import Topology, permutation
from premixdb.engine.identity import CodeVersion
from premixdb.engine.plans import filter_documents, query_identity
from premixdb.execution.pipeline import _packing_blocks

DIGESTS = st.binary(min_size=32, max_size=32).map(bytes.hex)
CODE = CodeVersion("local://property-tests", "a" * 40, "09" * 32)


@given(
    weights=st.lists(
        st.one_of(st.integers(0, 2**64 - 1), st.floats(min_value=0, max_value=1_000_000)),
        min_size=1,
        max_size=12,
    ).filter(any),
    budget=st.integers(0, 2**64 - 1),
)
def test_allocations_preserve_exact_budget_quotas_and_input_order_independence(
    weights: list[int | float], budget: int
) -> None:
    values = {str(i): weight for i, weight in enumerate(weights)}
    counts = allocations(values, budget)
    assert sum(counts.values()) == budget
    assert counts == allocations(dict(reversed(values.items())), budget)
    total = sum(Fraction(weight) for weight in weights)
    for key, count in counts.items():
        quota = Fraction(values[key]) * budget / total
        assert int(quota) <= count <= int(quota) + bool(quota % 1)
        assert values[key] or count == 0


def test_allocation_ties_use_labels_and_leave_zero_weights_empty() -> None:
    assert allocations({"c": 1, "b": 1, "a": 1, "zero": 0}, 2) == {
        "a": 1,
        "b": 1,
        "c": 0,
        "zero": 0,
    }


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


@given(
    lengths=st.lists(st.integers(0, 20), max_size=128),
    bounds=st.tuples(st.integers(0, 10_000), st.integers(0, 10_000)),
)
def test_packing_block_windows_preserve_repeated_boundaries_and_empty_blocks(
    lengths: list[int], bounds: tuple[int, int]
) -> None:
    boundaries = list(accumulate(lengths, initial=0))
    prefixes, total = boundaries[:-1], boundaries[-1]
    low, high = sorted(bound % (total + 2) for bound in bounds)
    expected = [
        index
        for index, (start, end) in enumerate(zip(prefixes, boundaries[1:], strict=True))
        if start < high and end > low
    ]
    assert list(_packing_blocks(prefixes, total, low, high)) == expected
