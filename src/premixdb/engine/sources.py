"""Captured text values and adapters for source requests."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from itertools import islice
from os import PathLike, fspath
from pathlib import Path
from typing import Iterable, Iterator, Self
from urllib.parse import urlsplit

from premixdb.schemas.protobuf import copy_message
from premixdb.v1 import storage_pb2 as storage


@dataclass(frozen=True)
class HuggingFaceSource:
    """Stream a Hub dataset, resolving a branch/tag to an immutable commit at capture.

    Use either HuggingFaceSource("allenai", "c4") or a full repository name.
    C4 defaults to its English subset; other datasets use their default subset.
    Supply a full revision to avoid a Hub metadata request.
    """

    repository: str
    dataset: str | None = None
    configuration: str | None = None
    split: str = "train"
    revision: str = "main"
    text_column: str = "text"
    key_column: str | None = None

    def __post_init__(self) -> None:
        for name in ("repository", "split", "revision", "text_column"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("dataset", "configuration", "key_column"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string or None")

    def _to_proto(self) -> storage.Source:
        repository = (
            self.repository if self.dataset is None else f"{self.repository}/{self.dataset}"
        )
        revision = self.revision
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            from huggingface_hub import HfApi

            revision = HfApi().dataset_info(repository, revision=revision).sha
        configuration = self.configuration
        if configuration is None and repository == "allenai/c4":
            configuration = "en"
        return storage.Source(
            hugging_face=storage.HuggingFaceDataset(
                repository=repository,
                configuration=configuration or "",
                split=self.split,
                revision=revision,
                text_column=self.text_column,
                key_column=self.key_column or "",
            )
        )


@dataclass(frozen=True)
class Source:
    key: str
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not isinstance(self.text, str):
            raise TypeError("source key and text must be strings")
        if not self.key:
            raise ValueError("source key must not be empty")
        self.key.encode("utf-8")
        self.text.encode("utf-8")

    @classmethod
    def read(cls, key: str, path: str | PathLike[str]) -> Self:
        """Read a UTF-8 file into a source with the given stable key."""
        try:
            return cls(key, Path(path).read_bytes().decode("utf-8"))
        except (OSError, UnicodeError) as exc:
            raise ValueError(str(exc)) from exc

    @classmethod
    def read_jsonl(
        cls,
        path: str | PathLike[str],
        *,
        text_column: str = "text",
        key_column: str | None = None,
        key_prefix: str | None = None,
        limit: int | None = None,
    ) -> Iterator[Self]:
        """Yield one source per UTF-8 JSONL row, including gzip-compressed files.

        By default, keys combine the filename (without `.gz`) and zero-based row
        number. They survive relocation and preserve repeated URLs/text as separate
        occurrences. Set key_prefix to namespace shards or key_column to use an
        existing unique string ID. limit bounds rows read; None reads the whole file.
        The returned iterator is single-use; call again to recapture the input.
        Snapshot capture still materializes the selected documents in memory.
        """
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("JSONL limit must be a nonnegative integer or None")
        path = Path(path)
        prefix = path.name.removesuffix(".gz") if key_prefix is None else key_prefix
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("JSONL key prefix must be a nonempty string")
        opener = gzip.open if path.suffix == ".gz" else open
        seen = set()
        with opener(path, "rt", encoding="utf-8") as stream:
            for ordinal, line in enumerate(islice(stream, limit)):
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("record must be an object")
                    text = row[text_column]
                    key = row[key_column] if key_column is not None else f"{prefix}/{ordinal:08d}"
                    item = cls(key, text)
                    if key in seen:
                        raise ValueError("duplicate document key")
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}: row {ordinal + 1}: {exc}") from exc
                seen.add(key)
                yield item


type SourceInput = (
    str
    | PathLike[str]
    | storage.Source
    | storage.HuggingFaceDataset
    | HuggingFaceSource
    | Iterable[Source]
)


def source_files(path: Path) -> Iterator[tuple[str, Path]]:
    """Enumerate local documents in path order, with relative POSIX keys."""
    directory = path.is_dir()
    for file in sorted(path.rglob("*")) if directory else (path,):
        if not directory or file.is_file():
            key = file.relative_to(path).as_posix() if directory else file.name
            yield key, file


def source_proto(value: SourceInput, *, limit: int | None = None) -> storage.Source:
    """Detach a source request, consuming at most limit items from an iterable."""
    if isinstance(value, HuggingFaceSource):
        value = value._to_proto()
    if isinstance(value, storage.HuggingFaceDataset):
        value = storage.Source(hugging_face=value)
    if isinstance(value, storage.Source):
        result = copy_message(value)
        if limit is not None:
            result.limit = limit
        return result
    if isinstance(value, (str, PathLike)):
        path = fspath(value)
        if urlsplit(path).scheme:
            raise ValueError("source paths must be local files")
        return storage.Source(
            limit=limit,
            files=storage.FileSources(
                documents=[storage.FileSource(path=str(Path(path).resolve()))]
            ),
        )
    return storage.Source(
        limit=limit,
        memory=storage.MemorySources(
            documents=(
                storage.MemorySource(uri=item.key, text=item.text) for item in islice(value, limit)
            )
        ),
    )
