"""Verified local partition files and immutable completion receipts.

Local workers publish immutable partition results.
Processors own kernel semantics; this module owns engine and artifact integrity.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import BinaryIO, Callable
from urllib.parse import unquote, urlsplit

from blake3 import blake3

from .._typing import JSON, json_object, json_string, load_json

CONTROL_LIMIT = 1024 * 1024
CHUNK_SIZE = 1024 * 1024


class IntegrityError(ValueError):
    pass


class Kernel(str, Enum):
    FEATURES = "features"
    DEDUPE_INDEX = "dedupe_index"
    DECONTAMINATION_INDEX = "decontamination_index"
    TOKENIZE = "tokenize"
    PACK = "pack"


def _path(uri: str) -> Path:
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "file"
        or parsed.netloc not in ("", "localhost")
        or not parsed.path.startswith("/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("partition objects require absolute file:// URIs")
    return Path(unquote(parsed.path))


@dataclass(frozen=True)
class Artifact:
    uri: str
    digest: bytes

    def __post_init__(self) -> None:
        _path(self.uri)
        if not isinstance(self.digest, bytes) or len(self.digest) != 32:
            raise ValueError("artifact requires a 32-byte content digest")


@dataclass(frozen=True)
class PartitionTask:
    key: bytes
    engine_digest: bytes
    kernel: Kernel
    inputs: tuple[Artifact, ...]
    output_uri: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(pin, bytes) or len(pin) != 32 for pin in (self.key, self.engine_digest)
        ):
            raise ValueError("task and engine identities must contain 32 bytes")
        if not isinstance(self.kernel, Kernel):
            raise TypeError("kernel requires a typed enum value")
        if not isinstance(self.inputs, tuple) or any(
            not isinstance(item, Artifact) for item in self.inputs
        ):
            raise TypeError("inputs must be immutable artifact references")
        _path(self.output_uri)


@dataclass(frozen=True)
class Receipt:
    task_key: bytes
    engine_digest: bytes
    output: Artifact


def checked_receipt(task: PartitionTask, receipt: Receipt) -> Receipt:
    if (
        not isinstance(receipt, Receipt)
        or receipt.task_key != task.key
        or receipt.engine_digest != task.engine_digest
        or not isinstance(receipt.output, Artifact)
        or receipt.output.uri != task.output_uri
    ):
        raise IntegrityError("worker returned a receipt for another task, engine, or output")
    return receipt


def _artifact(value: Artifact) -> dict[str, JSON]:
    return {"uri": value.uri, "digest": value.digest.hex()}


def _task_data(task: PartitionTask) -> dict[str, JSON]:
    return {
        "key": task.key.hex(),
        "engine_digest": task.engine_digest.hex(),
        "kernel": task.kernel.value,
        "inputs": [_artifact(value) for value in task.inputs],
        "output_uri": task.output_uri,
    }


class PartitionStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve().as_uri().rstrip("/")

    def _copy_verified(self, artifact: Artifact, target: BinaryIO | None = None) -> None:
        digest = blake3()
        with _path(artifact.uri).open("rb") as source:
            while data := source.read(CHUNK_SIZE):
                digest.update(data)
                if target is not None:
                    target.write(data)
        if digest.digest() != artifact.digest:
            raise IntegrityError("partition object checksum mismatch")

    def verify(self, artifact: Artifact) -> None:
        self._copy_verified(artifact)

    def download(self, artifact: Artifact, path: str | Path | None = None) -> None:
        if path is None:
            self._copy_verified(artifact)
            return
        with Path(path).open("wb") as target:
            self._copy_verified(artifact, target)

    def publish_file(self, uri: str, path: str | Path) -> Artifact:
        destination = _path(uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Hash the staged bytes and publish with a link so readers never see a partial file.
        with NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
            staging = Path(temporary.name)
            try:
                digest = blake3()
                with Path(path).open("rb") as source:
                    while data := source.read(CHUNK_SIZE):
                        digest.update(data)
                        temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
                artifact = Artifact(uri, digest.digest())
                try:
                    os.link(staging, destination)
                except FileExistsError:
                    # Identical retries/races are safe; never overwrite conflicting bytes.
                    self.verify(artifact)
                descriptor = os.open(destination.parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                return artifact
            finally:
                staging.unlink(missing_ok=True)

    def receipt_uri(self, task: PartitionTask) -> str:
        return self.root + "/receipts/" + task.key.hex() + ".json"

    def read_control(self, uri: str) -> bytes:
        with _path(uri).open("rb") as source:
            data = source.read(CONTROL_LIMIT + 1)
        if len(data) > CONTROL_LIMIT:
            raise IntegrityError("partition control record exceeds 1 MiB")
        return data

    def find(self, task: PartitionTask) -> Receipt | None:
        # Only an absent receipt is a cache miss; missing committed output is an error.
        try:
            data = self.read_control(self.receipt_uri(task))
        except (FileNotFoundError, KeyError):
            return None
        except IntegrityError as exc:
            raise IntegrityError("invalid partition completion record") from exc
        try:
            value = json_object(load_json(data))
            if (
                set(value) != {"version", "task", "output"}
                or value["version"] != 1
                or value["task"] != _task_data(task)
            ):
                raise ValueError("completion record belongs to another task")
            output = json_object(value["output"])
            receipt = Receipt(
                task.key,
                task.engine_digest,
                Artifact(json_string(output["uri"]), bytes.fromhex(json_string(output["digest"]))),
            )
            checked_receipt(task, receipt)
        except (KeyError, TypeError, ValueError) as exc:
            raise IntegrityError("invalid partition completion record") from exc
        self.verify(receipt.output)
        return receipt

    def publish_receipt(self, task: PartitionTask, output: Artifact) -> Receipt:
        receipt = checked_receipt(task, Receipt(task.key, task.engine_digest, output))
        self.verify(output)
        data = json.dumps(
            {"version": 1, "task": _task_data(task), "output": _artifact(output)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if len(data) > CONTROL_LIMIT:
            raise IntegrityError("partition completion record exceeds 1 MiB")
        with NamedTemporaryFile() as temporary:
            temporary.write(data)
            temporary.flush()
            self.publish_file(self.receipt_uri(task), temporary.name)
        return receipt


class PartitionWorker:
    """Run a processor with verified local inputs; publish output before its receipt."""

    def __init__(
        self,
        storage: PartitionStore,
        engine_digest: bytes,
        processor: Callable[[PartitionTask, tuple[Path, ...], Path], None],
    ) -> None:
        if not isinstance(engine_digest, bytes) or len(engine_digest) != 32:
            raise ValueError("worker engine digest must contain 32 bytes")
        self.storage, self.engine_digest, self.processor = storage, engine_digest, processor

    def __call__(self, task: PartitionTask) -> Receipt:
        if task.engine_digest != self.engine_digest:
            raise IntegrityError("worker engine does not match partition pin")
        receipt = self.storage.find(task)
        if receipt is not None:
            return receipt
        with TemporaryDirectory(prefix="premixdb-partition-") as directory:
            root = Path(directory)
            inputs = tuple(root / f"input-{i}" for i in range(len(task.inputs)))
            for artifact, path in zip(task.inputs, inputs, strict=True):
                self.storage.download(artifact, path)
            output = root / "output"
            self.processor(task, inputs, output)
            artifact = self.storage.publish_file(task.output_uri, output)
            return self.storage.publish_receipt(task, artifact)
