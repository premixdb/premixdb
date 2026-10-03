"""Execution fingerprints change with code and transitive runtime dependencies."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from premixdb import _runtime
from premixdb.execution.catalog import plan


def test_source_fingerprint_includes_uncommitted_code_but_ignores_bytecode(tmp_path: Path) -> None:
    source = tmp_path / "kernel.py"
    source.write_text("value = 1\n")
    first = _runtime._source_digest(tmp_path)
    (tmp_path / "kernel.pyc").write_bytes(b"compiled cache")
    assert _runtime._source_digest(tmp_path) == first
    source.write_text("value = 2\n")
    assert _runtime._source_digest(tmp_path) != first


def test_runtime_fingerprint_tracks_dependency_extras_and_transitive_versions() -> None:
    requirements = {
        "premixdb": ["model-runtime[inference]>=1"],
        "model-runtime": ['numeric-runtime>=1; extra == "inference"'],
        "numeric-runtime": [],
    }
    versions = {"premixdb": "0.1.0", "model-runtime": "1", "numeric-runtime": "1"}
    with (
        patch.object(
            _runtime,
            "distribution",
            side_effect=lambda n: SimpleNamespace(requires=requirements[n]),
        ),
        patch.object(_runtime, "version", side_effect=lambda n: versions[n]),
    ):
        first = _runtime._capture_environment()
        versions["numeric-runtime"] = "2"
        assert _runtime._capture_environment() != first
        versions["numeric-runtime"] = "1"
        assert _runtime._capture_environment() == first
        with patch.object(_runtime.platform, "python_version", return_value="3.99.0"):
            assert _runtime._capture_environment() != first


def test_field_derivation_pins_include_runtime_and_are_frozen_with_the_process() -> None:
    code = _runtime.current_code()
    first = plan([b"s" * 32], "language.en", bytes.fromhex(code.commit))
    assert code.environment != "00" * 32
    with patch.object(_runtime, "_capture_environment", side_effect=AssertionError("recaptured")):
        assert _runtime.current_code() == code
    changed = replace(code, environment="ab" * 32)
    with patch.object(_runtime, "current_code", return_value=changed):
        second = plan([b"s" * 32], "language.en", bytes.fromhex(code.commit))
        assert first.id != second.id
