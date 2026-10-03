"""Capture the repository revision once when the execution runtime is loaded."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
from importlib.metadata import PackageNotFoundError, distribution, version
from os import PathLike
from pathlib import Path

from .engine.identity import CodeVersion


def _source_digest(package: str | PathLike[str] | None = None) -> str:
    package = Path(__file__).resolve().parent if package is None else Path(package)
    digest = hashlib.sha256(b"premixdb-installed-source/v1\0")
    for path in sorted(package.rglob("*.py")):
        name = path.relative_to(package).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big") + name)
        digest.update(len(content).to_bytes(8, "big") + content)
    return digest.hexdigest()


def _capture_environment() -> str:
    # Metadata only: importing torch or loading a model must never be needed to
    # identify an execution. Include transitive runtime dependencies, not tools.
    from packaging.requirements import Requirement

    packages, requirements_by_name, expanded = {}, {}, set()
    pending = [("premixdb", "")]
    while pending:
        name, extra = pending.pop()
        name = re.sub(r"[-_.]+", "-", name).lower()
        if (name, extra) in expanded:
            continue
        expanded.add((name, extra))
        try:
            if name not in packages:
                packages[name] = version(name)
                requirements_by_name[name] = distribution(name).requires or ()
        except PackageNotFoundError:
            packages[name] = "unavailable"
            requirements_by_name[name] = ()
            continue
        for text in requirements_by_name[name]:
            requirement = Requirement(text)
            if requirement.marker is None or requirement.marker.evaluate({"extra": extra}):
                pending.extend((requirement.name, value) for value in ("", *requirement.extras))
    data = json.dumps(
        {
            "source": _source_digest(),
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "machine": platform.machine(),
            "packages": packages,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(b"premixdb-environment/v1\0" + data).hexdigest()


def _capture_commit() -> str:
    commit = os.environ.get("PREMIXDB_GIT_COMMIT")
    if commit is None:
        root = Path(__file__).resolve().parents[2]
        if not (root / ".git").exists():
            # Wheels and source archives need no Git installation or environment
            # setup. Their execution revision is a deterministic source fingerprint.
            commit = _source_digest()[:40]
        else:
            commit = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=10
            ).strip()
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None:
        raise ValueError("PREMIXDB_GIT_COMMIT must be a full lowercase Git commit")
    return commit


_COMMIT = _capture_commit()
_ENVIRONMENT = _capture_environment()


def current_code() -> CodeVersion:
    return CodeVersion("premixdb://repository", _COMMIT, _ENVIRONMENT)


def resolve_code(commit: bytes = b"") -> CodeVersion:
    from ._requests import _commit

    requested = _commit(commit)
    current = current_code()
    if requested and requested.hex() != current.commit:
        raise NotImplementedError("this runtime cannot execute a different Git revision")
    return current
