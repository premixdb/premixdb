"""Regression coverage for the local partition publication and retry contract."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
import time
import unittest
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import unquote, urlsplit

import pytest
from _type_support import invalid_call
from blake3 import blake3

from premixdb.runtime.partitions import (
    CHUNK_SIZE,
    CONTROL_LIMIT,
    Artifact,
    IntegrityError,
    Kernel,
    PartitionStore,
    PartitionTask,
    PartitionWorker,
    Receipt,
    checked_receipt,
)
from premixdb.runtime.pipeline import PartitionPipeline
from premixdb.storage import publication as _files


class PartitionTests(unittest.TestCase):
    def test_submission_window_is_bounded_ordered_and_cancels_pending_on_failure(self) -> None:
        from premixdb.runtime.pipeline import execute_tasks

        tasks = [replace(self.task, key=i.to_bytes(32, "big")) for i in range(5)]
        submitted = []

        def submit(run: Callable[[PartitionTask], Receipt], task: PartitionTask) -> Mock:
            future = Mock()
            future.result.return_value = Receipt(
                task.key, task.engine_digest, Artifact(task.output_uri, b"o" * 32)
            )
            submitted.append(future)
            return future

        pool = Mock(submit=submit)
        results = execute_tasks(pool, Mock(), tasks, 4)
        first = next(results)
        self.assertEqual(len(submitted), 4)
        submitted[1].result.side_effect = RuntimeError("worker failed")
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            next(results)
        self.assertEqual(first.task_key, tasks[0].key)
        self.assertEqual(len(submitted), 5)
        for future in submitted[2:]:
            future.cancel.assert_called_once()
        submitted.clear()
        self.assertEqual(
            [result.task_key for result in execute_tasks(pool, Mock(), tasks, 4)],
            [task.key for task in tasks],
        )

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="premixdb partitions ")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.store = PartitionStore(self.root / "partitions")
        self.source = self.root / "source"
        self.source.write_bytes(b"document")
        artifact = self.store.publish_file(self.store.root + "/inputs/document", self.source)
        self.task = PartitionTask(
            b"k" * 32,
            b"e" * 32,
            Kernel.FEATURES,
            (artifact,),
            self.store.root + "/outputs/result",
        )
        self.processor = Mock(side_effect=self.process)
        self.worker = PartitionWorker(self.store, self.task.engine_digest, self.processor)

    def process(self, task: PartitionTask, inputs: tuple[Path, ...], output: Path) -> None:
        self.assertEqual(task, self.task)
        self.assertFalse(self.receipt_path.exists())
        output.write_bytes(inputs[0].read_bytes().upper())

    @property
    def receipt_path(self) -> Path:
        return self.root / "partitions/receipts" / (self.task.key.hex() + ".json")

    @property
    def output_path(self) -> Path:
        return self.root / "partitions/outputs/result"

    def test_spawn_serialization_and_restart_reuse_verified_receipt(self) -> None:
        task = pickle.loads(pickle.dumps(self.task))
        self.assertEqual(task, self.task)
        receipt = self.worker(task)
        self.assertEqual(receipt.output.digest, blake3(b"DOCUMENT").digest())
        self.assertEqual(self.output_path.read_bytes(), b"DOCUMENT")
        self.assertEqual(checked_receipt(task, receipt), receipt)
        restarted = PartitionWorker(
            PartitionStore(self.root / "partitions"), task.engine_digest, self.processor
        )
        self.assertEqual(restarted(task), receipt)
        self.processor.assert_called_once()

    def test_wrong_engine_and_corrupt_inputs_fail_before_processing(self) -> None:
        with self.assertRaisesRegex(ValueError, "engine"):
            self.worker(replace(self.task, engine_digest=b"x" * 32))
        input_path = Path(unquote(urlsplit(self.task.inputs[0].uri).path))
        input_path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.worker(self.task)
        self.processor.assert_not_called()
        self.assertFalse(self.output_path.exists())
        self.assertIsNone(self.store.find(self.task))

    def test_checked_receipt_rejects_wrong_task_engine_or_output(self) -> None:
        receipt = self.worker(self.task)
        for bad in (
            None,
            replace(receipt, task_key=b"x" * 32),
            replace(receipt, engine_digest=b"x" * 32),
            replace(receipt, output=self.task.inputs[0]),
            invalid_call(Receipt, self.task.key, self.task.engine_digest, None),
        ):
            with self.subTest(receipt=bad), self.assertRaisesRegex(ValueError, "receipt"):
                invalid_call(checked_receipt, self.task, bad)

    def test_receipt_binds_full_descriptor_even_when_key_is_reused(self) -> None:
        self.worker(self.task)
        for changed in (
            replace(self.task, kernel=Kernel.TOKENIZE),
            replace(self.task, inputs=()),
            replace(self.task, inputs=(replace(self.task.inputs[0], digest=b"x" * 32),)),
            replace(self.task, engine_digest=b"x" * 32),
            replace(self.task, output_uri=self.store.root + "/outputs/other"),
        ):
            with self.subTest(task=changed), self.assertRaisesRegex(ValueError, "completion"):
                self.store.find(changed)

    def test_corrupt_receipts_are_errors_not_cache_misses(self) -> None:
        self.worker(self.task)
        valid = json.loads(self.receipt_path.read_bytes())
        for data in (
            b"not json",
            b"null",
            b"[]",
            json.dumps({**valid, "version": 2}).encode(),
            json.dumps({**valid, "output": {"uri": self.task.output_uri, "digest": "00"}}).encode(),
            b"x" * (CONTROL_LIMIT + 1),
        ):
            with self.subTest(data=data[:80]):
                self.receipt_path.write_bytes(data)
                with self.assertRaisesRegex(ValueError, "completion"):
                    self.worker(self.task)
        self.processor.assert_called_once()

    def test_missing_or_corrupt_committed_output_is_not_recomputed(self) -> None:
        self.worker(self.task)
        self.output_path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.worker(self.task)
        self.output_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.worker(self.task)
        self.processor.assert_called_once()

    def test_failure_before_output_publication_leaves_no_receipt(self) -> None:
        self.processor.side_effect = RuntimeError("processor failed")
        with self.assertRaisesRegex(RuntimeError, "processor failed"):
            self.worker(self.task)
        self.assertFalse(self.output_path.exists())
        self.assertIsNone(self.store.find(self.task))
        self.processor.side_effect = self.process
        self.assertEqual(self.worker(self.task), self.store.find(self.task))

    def test_retry_after_output_publication_before_receipt(self) -> None:
        with patch.object(self.store, "publish_receipt", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.worker(self.task)
        self.assertEqual(self.output_path.read_bytes(), b"DOCUMENT")
        self.assertIsNone(self.store.find(self.task))
        self.assertEqual(self.worker(self.task), self.store.find(self.task))
        self.assertEqual(self.processor.call_count, 2)

    def test_identical_concurrent_publication_and_conflicting_output(self) -> None:
        with ThreadPoolExecutor(max_workers=4) as pool:
            artifacts = list(
                pool.map(
                    lambda i: (
                        self.store.publish_file(self.task.output_uri, self.source)
                        if i % 2
                        else self.store.publish_bytes(self.task.output_uri, b"document")
                    ),
                    range(8),
                )
            )
        self.assertTrue(all(artifact == artifacts[0] for artifact in artifacts))
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.store.publish_bytes(self.task.output_uri, b"conflict")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.worker(self.task)
        self.assertEqual(self.output_path.read_bytes(), b"document")
        self.assertIsNone(self.store.find(self.task))
        self.assertEqual(list(self.output_path.parent.iterdir()), [self.output_path])

    def test_memory_reads_verify_empty_and_multichunk_objects(self) -> None:
        for data in (b"", b"first" * CHUNK_SIZE + b"last"):
            with self.subTest(length=len(data)):
                uri = self.store.root + "/objects/" + blake3(data).hexdigest()
                artifact = self.store.publish_bytes(uri, data)
                self.assertEqual(artifact.digest, blake3(data).digest())
                self.assertEqual(self.store.read(artifact), data)
                destination = self.root / "download"
                self.store.download(artifact, destination)
                self.assertEqual(destination.read_bytes(), data)
                self.assertEqual(self.store.publish_file(uri, destination), artifact)
                Path(unquote(urlsplit(uri).path)).write_bytes(b"corrupt")
                with self.assertRaisesRegex(IntegrityError, "checksum"):
                    self.store.read(artifact)

    def test_partition_rows_are_verified_before_being_exposed(self) -> None:
        pipeline = PartitionPipeline(self.store)
        self.addCleanup(pipeline.close)
        data = json.dumps({"version": 1, "rows": [{"id": "original"}]}).encode()
        artifact = self.store.publish_bytes(self.task.output_uri, data)
        self.assertEqual(list(pipeline.rows((artifact,))), [{"id": "original"}])
        self.output_path.write_bytes(
            json.dumps({"version": 1, "rows": [{"id": "changed"}]}).encode()
        )
        with self.assertRaisesRegex(IntegrityError, "checksum"):
            next(pipeline.rows((artifact,)))

    def test_reconciliation_failure_cancels_pending_tasks_with_a_live_traceback(self) -> None:
        pipeline = PartitionPipeline(self.store)
        self.addCleanup(pipeline.close)
        tasks = [
            replace(self.task, key=i.to_bytes(32, "big"), engine_digest=pipeline.engine)
            for i in range(3)
        ]
        output = self.store.publish_bytes(self.task.output_uri, b"result")
        receipts = [Receipt(task.key, task.engine_digest, output) for task in tasks]
        for phase in ("receipt", "checksum"):
            with self.subTest(phase=phase):
                bad = (
                    replace(receipts[0], task_key=b"x" * 32)
                    if phase == "receipt"
                    else replace(receipts[0], output=replace(output, digest=b"x" * 32))
                )
                futures = [Mock() for _ in tasks]
                for future, receipt in zip(futures, [bad, *receipts[1:]], strict=True):
                    future.result.return_value = receipt
                with patch.object(pipeline._pool, "submit", side_effect=futures) as submit:
                    with pytest.raises(IntegrityError, match=phase) as failure:
                        pipeline._execute(iter(tasks))
                self.assertIsNotNone(failure.value.__traceback__)
                self.assertEqual(submit.call_count, pipeline._window)
                futures[1].cancel.assert_called_once()
                futures[2].result.assert_not_called()

        futures = [Mock() for _ in tasks]
        for future, receipt in zip(futures, receipts, strict=True):
            future.result.return_value = receipt
        with patch.object(pipeline._pool, "submit", side_effect=futures) as submit:
            self.assertEqual(pipeline._execute(iter(tasks)), (output,) * len(tasks))
            self.assertEqual(pipeline._execute(iter(())), ())
        self.assertEqual(submit.call_count, len(tasks))

    def test_failed_publication_closes_input_and_cleans_staging_with_live_tracebacks(self) -> None:
        for phase in ("read", "link"):
            with self.subTest(phase=phase):
                source = self.source.open("rb")
                self.addCleanup(source.close)
                fault = (
                    patch.object(
                        source, "read", side_effect=[b"document", OSError("publish failed")]
                    )
                    if phase == "read"
                    else patch.object(_files.os, "link", side_effect=OSError("publish failed"))
                )
                with patch.object(Path, "open", return_value=source), fault:
                    with pytest.raises(OSError, match="publish failed") as failure:
                        self.store.publish_file(self.task.output_uri, self.source)
                self.assertIsNotNone(failure.value.__traceback__)
                self.assertTrue(source.closed)
                self.assertFalse(self.output_path.exists())
                self.assertEqual(list(self.output_path.parent.iterdir()), [])

    def test_receipt_requires_verified_output(self) -> None:
        output = Artifact(self.task.output_uri, blake3(b"DOCUMENT").digest())
        with self.assertRaises(FileNotFoundError):
            self.store.publish_receipt(self.task, output)
        self.output_path.parent.mkdir(parents=True)
        self.output_path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.store.publish_receipt(self.task, output)
        self.assertIsNone(self.store.find(self.task))

    def test_task_admission_rejects_invalid_pins_and_unsupported_uris(self) -> None:
        for change in (
            {"key": b"short"},
            {"engine_digest": b"short"},
            {"kernel": "features"},
            {"inputs": list(self.task.inputs)},
            {"output_uri": "https://bucket/output"},
            {"output_uri": "file://remote/output"},
            {"output_uri": "relative/output"},
            {"output_uri": self.task.output_uri + "?query=1"},
        ):
            with self.subTest(change=change), self.assertRaises((ValueError, TypeError)):
                replace(self.task, **change)
        with self.assertRaisesRegex(ValueError, "digest"):
            Artifact(self.task.output_uri, b"short")


def concurrent_pid(directory: str | Path) -> int:
    path = Path(directory) / str(os.getpid())
    path.touch()
    deadline = time.monotonic() + 10
    while len(list(Path(directory).iterdir())) < 2:
        if time.monotonic() > deadline:
            raise RuntimeError("process pool did not run two tasks concurrently")
        time.sleep(0.02)
    return os.getpid()


class LocalProcessPartitionTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = PartitionStore(self.root / "partitions")
        source = self.root / "source"
        source.write_bytes(b"verified input")
        self.input = self.store.publish_file((self.root / "input").as_uri(), source)
        self.task = PartitionTask(
            b"t" * 32, b"e" * 32, Kernel.TOKENIZE, (self.input,), (self.root / "output").as_uri()
        )

    def test_completed_partitions_are_reused_and_corruption_is_not_a_cache_miss(self) -> None:
        calls = []

        def processor(task: PartitionTask, inputs: tuple[Path, ...], output: Path) -> None:
            calls.append(task.key)
            output.write_bytes(inputs[0].read_bytes().upper())

        worker = PartitionWorker(self.store, self.task.engine_digest, processor)
        first = worker(self.task)
        self.assertEqual(worker(self.task), first)
        self.assertEqual(len(calls), 1)
        with self.assertRaisesRegex(IntegrityError, "engine"):
            worker(replace(self.task, engine_digest=b"x" * 32))
        (self.root / "output").write_bytes(b"corrupt")
        with self.assertRaisesRegex(IntegrityError, "checksum"):
            worker(self.task)
        (self.root / "output").unlink()
        with self.assertRaises(FileNotFoundError):
            worker(self.task)
        self.assertEqual(len(calls), 1)

    def test_input_integrity_failure_never_publishes_completion(self) -> None:
        (self.root / "input").write_bytes(b"changed")
        worker = PartitionWorker(self.store, self.task.engine_digest, lambda *args: None)
        with self.assertRaisesRegex(IntegrityError, "checksum"):
            worker(self.task)
        self.assertIsNone(self.store.find(self.task))
        self.assertFalse((self.root / "output").exists())

    def test_conflicting_outputs_and_wrong_task_receipts_fail(self) -> None:
        receipt = self.store.publish_receipt(
            self.task, self.store.publish_file(self.task.output_uri, self.root / "source")
        )
        self.assertEqual(self.store.find(self.task), receipt)
        with self.assertRaisesRegex(IntegrityError, "completion"):
            self.store.find(replace(self.task, kernel=Kernel.PACK))
        (self.root / "source").write_bytes(b"different")
        with self.assertRaisesRegex(IntegrityError, "checksum"):
            self.store.publish_file(self.task.output_uri, self.root / "source")

    @pytest.mark.integration
    def test_pool_computes_concurrently_and_reuses_worker_processes(self) -> None:
        pipeline = PartitionPipeline(self.store, workers=2)
        self.addCleanup(pipeline.close)
        directory = self.root / "pids"
        directory.mkdir()
        first = [pipeline._pool.submit(concurrent_pid, str(directory)) for _ in range(2)]
        pids = {future.result(timeout=20) for future in first}
        self.assertEqual(len(pids), 2)
        self.assertNotIn(os.getpid(), pids)
        later = {pipeline._pool.submit(os.getpid).result(timeout=5) for _ in range(4)}
        self.assertTrue(later <= pids)
        pipeline.close()
        with self.assertRaisesRegex(RuntimeError, "shutdown"):
            pipeline._pool.submit(os.getpid)


if __name__ == "__main__":
    unittest.main()
