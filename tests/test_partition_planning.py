"""Partition planning reads captured text once and retains only admitted artifacts."""

import base64
import json
import tracemalloc
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from unittest.mock import patch

import pytest
from blake3 import blake3

from premixdb._typing import JSON
from premixdb.engine.identity import CodeVersion
from premixdb.engine.snapshots import Snapshot, Store, StoredDocument
from premixdb.enrichment.types import Document
from premixdb.execution.partitions import Artifact, Kernel, PartitionStore, PartitionTask
from premixdb.execution.pipeline import PartitionPipeline
from premixdb.internal import derivation_pb2 as e


@pytest.mark.parametrize(
    "kernel", [Kernel.TOKENIZE, Kernel.DEDUPE_INDEX, Kernel.DECONTAMINATION_INDEX]
)
def test_map_reads_each_captured_frame_once(tmp_path: Path, kernel: Kernel) -> None:
    store = Store(tmp_path / "snapshot")
    code = CodeVersion("local://test", "a" * 40, "09" * 32)
    sources = [("unicode", "pré 🌍\n" * 3), ("empty", "")]
    original = Snapshot("01" * 16, sources, code)
    store.save(original, frame_bytes=8)
    snapshot = store.load(original.id, lazy=True)
    documents = list(snapshot.documents.values())
    assert all(isinstance(doc, StoredDocument) for doc in documents)
    frames = sum(len(doc.record["frames"]) for doc in documents if isinstance(doc, StoredDocument))
    pipeline = PartitionPipeline(PartitionStore(tmp_path / "partitions"))
    tasks: list[PartitionTask] = []

    def admit(values: Iterable[PartitionTask]) -> tuple[Artifact, ...]:
        tasks.extend(values)
        return ()

    options: dict[str, JSON] = {"tokenizer": ""} if kernel == Kernel.TOKENIZE else {"algorithm": 1}
    try:
        with (
            patch.object(pipeline, "_execute", side_effect=admit),
            patch.object(store, "_frame", wraps=store._frame) as read,
        ):
            pipeline._map(kernel, documents, options)
        assert read.call_count == frames
        assert len(tasks) == 1
        request = json.loads(pipeline.storage.read(tasks[0].inputs[0]))
        assert [(row["text"], row["ranges"]) for row in request["rows"]] == [
            (text, [[0, len(text.encode())]]) for _, text in sorted(sources)
        ]
    finally:
        pipeline.close()


def test_feature_planning_retains_artifacts_instead_of_all_cohort_text(tmp_path: Path) -> None:
    pipeline = PartitionPipeline(PartitionStore(tmp_path / "partitions"))
    policy = e.EnrichmentProducer()
    policy.language.SetInParent()
    definition = b'{"provider":"test"}'
    count = 32
    tasks: list[PartitionTask] = []

    def cohorts() -> Iterator[Sequence[Document]]:
        for i in range(count):
            yield [Document(f"{i}-{j}", f"{i}-{j}:" + "x" * 100_000) for j in range(i % 3)]

    def admit(values: Iterable[PartitionTask]) -> tuple[Artifact, ...]:
        tasks.extend(values)
        return ()

    try:
        tracemalloc.start()
        try:
            with patch.object(pipeline, "_execute", side_effect=admit):
                assert list(pipeline.features(policy, definition, cohorts())) == []
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 2_000_000
        assert len(tasks) == count
        producer = base64.b64encode(policy.SerializeToString(deterministic=True)).decode()
        for i, task in enumerate(tasks):
            request = dict(
                version=1,
                producer=producer,
                definition=definition.decode(),
                rows=[
                    dict(id=f"{i}-{j}", text=f"{i}-{j}:" + "x" * 100_000, url=None)
                    for j in range(i % 3)
                ],
            )
            data = json.dumps(request, separators=(",", ":")).encode()
            assert pipeline.storage.read(task.inputs[0]) == data
            expected = blake3(
                b"premixdb-features/v1\0"
                + pipeline.engine
                + blake3(data).digest()
                + i.to_bytes(8, "big")
                + count.to_bytes(8, "big")
            ).digest()
            assert task.key == expected
            assert task.output_uri.endswith("/outputs/" + expected.hex())
    finally:
        pipeline.close()
