"""Read the package's version from its installation or source project metadata."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def current_version() -> str:
    try:
        return version("premixdb")
    except PackageNotFoundError:
        import tomllib

        project = Path(__file__).resolve().parents[2] / "pyproject.toml"
        if not project.is_file():
            raise RuntimeError("premixdb package metadata is missing") from None
        with project.open("rb") as stream:
            value = tomllib.load(stream)["project"]["version"]
        if not isinstance(value, str) or not value:
            raise RuntimeError("premixdb project version is missing")
        return value


__version__ = current_version()
