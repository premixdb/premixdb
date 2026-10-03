"""Keep native computation bounded when pytest and loader workers run together."""

import os
from collections import Counter

import pytest

# Tiny tensor/model fixtures do not benefit from a native pool per pytest worker.
# Defaults also reach spawned DataLoader processes; explicit user settings win.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(variable, "1")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Start expensive process work early, preserving order within each file."""
    counts = Counter(item.path for item in items)
    process_files = {
        "test_training_reads.py",
        "test_torch.py",
        "test_packaging.py",
        "test_tutorials.py",
        "test_extended_resources.py",
        "test_cli.py",
    }
    integration = any(item.get_closest_marker("integration") is not None for item in items)
    # xdist's count-based ordering otherwise leaves the small spawn/build files
    # until the end. Fast-only runs retain largest-file-first ordering.
    items.sort(
        key=lambda item: (
            not (integration and item.path.name in process_files),
            -counts[item.path],
        )
    )
