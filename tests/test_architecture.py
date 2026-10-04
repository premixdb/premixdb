"""Keep shared algorithms and readers independent of API and runtime loading."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src/premixdb"


@pytest.mark.parametrize(
    "package,forbidden",
    [
        ("engine", {"api", "runtime", "cli"}),
        ("schemas", {"api", "runtime", "cli"}),
        ("fields", {"api", "runtime", "cli"}),
        ("storage", {"api", "runtime", "cli"}),
        ("training", {"api", "runtime", "cli"}),
        ("enrichment", {"api", "runtime", "cli"}),
        ("runtime", {"api", "cli"}),
    ],
)
def test_package_import_boundaries(package: str, forbidden: set[str]) -> None:
    violations = []
    for path in sorted((PACKAGE / package).rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parent = ["premixdb", *path.relative_to(PACKAGE).parts[:-1]]
                    base = ".".join(parent[: len(parent) - node.level + 1])
                    module = base + ("." + node.module if node.module else "")
                else:
                    module = node.module or ""
                imports = [module + "." + alias.name for alias in node.names]
            else:
                continue
            for name in imports:
                parts = name.split(".")
                if len(parts) >= 2 and parts[0] == "premixdb" and parts[1] in forbidden:
                    violations.append(f"{path.relative_to(PACKAGE)}:{node.lineno}: {name}")
    assert not violations, "Forbidden layer imports:\n" + "\n".join(violations)


def test_public_import_and_read_only_catalog_do_not_load_runtime(tmp_path: Path) -> None:
    # Use a fresh interpreter so test collection cannot mask eager imports.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "premixdb.runtime" or fullname.startswith("premixdb.runtime."):
            raise AssertionError("unexpected runtime import: " + fullname)

sys.meta_path.insert(0, BlockRuntime())
import premixdb
from premixdb.storage.objects import ObjectStore

with ObjectStore(sys.argv[1]):
    pass
with premixdb.PremixDB(storage=sys.argv[1], read_only=True) as db:
    assert db.Corpus.list() == []
""",
            str(tmp_path),
        ],
        check=True,
    )
