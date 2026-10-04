"""Resource reads preserve durable state while jobs and cache entries change."""

from __future__ import annotations

import tracemalloc
from concurrent.futures import Future
from pathlib import Path
from threading import Event
from typing import Literal
from unittest.mock import patch

import pytest
from _type_support import coordinator

import premixdb as p
from premixdb.runtime.coordinator import wait
from premixdb.runtime.materialization import Materializer
from premixdb.schemas.ids import _decode_id
from premixdb.schemas.protobuf import copy_message
from premixdb.storage.catalog import Catalog
from premixdb.v1 import data_mixture_pb2 as d
from premixdb.v1 import query_pb2 as q
from premixdb.v1 import status_pb2 as status


@pytest.mark.parametrize("fail", [False, True])
def test_resource_wait_wakes_on_local_completion_without_polling(
    tmp_path: Path, fail: bool
) -> None:
    release = Event()
    with p.PremixDB(storage=tmp_path, poll_interval=60, progress=False) as db:
        service = coordinator(db)
        query = db.Corpus("wake", [p.Source("a", "hello")]).query()
        recipe = query._proto

        def execute() -> q.Query:
            assert release.wait(timeout=5)
            if fail:
                raise ValueError("worker failed")
            completed = copy_message(recipe)
            completed.status = status.STATUS_COMPLETED
            completed.profile.output_documents = 1
            return completed

        future = service._materialize("query", recipe, execute)
        running = copy_message(recipe)
        running.status = status.STATUS_RUNNING
        handle = p.Query(db, running)

        def finish(futures: tuple[Future[q.Query], ...], *, timeout: float) -> object:
            release.set()
            return wait(futures, timeout=timeout)

        try:
            with (
                patch("premixdb.runtime.coordinator.wait", side_effect=finish) as wake,
                patch("premixdb.api.base.time.sleep", side_effect=AssertionError("polled")),
            ):
                if fail:
                    with pytest.raises(p.ExecutionError, match="worker failed"):
                        handle.wait(timeout=5)
                else:
                    assert handle.wait(timeout=5) is handle
                    assert handle.profile().output_documents == 1
                wake.assert_called_once()
        finally:
            release.set()
            if fail:
                with pytest.raises(ValueError, match="worker failed"):
                    future.result(timeout=5)
            else:
                future.result(timeout=5)


def test_resource_wait_deadline_does_not_cancel_the_local_job(tmp_path: Path) -> None:
    release = Event()
    with p.PremixDB(storage=tmp_path, poll_interval=60, progress=False) as db:
        service = coordinator(db)
        query = db.Corpus("deadline", [p.Source("a", "hello")]).query()
        recipe = query._proto

        def execute() -> q.Query:
            assert release.wait(timeout=5)
            completed = copy_message(recipe)
            completed.status = status.STATUS_COMPLETED
            completed.profile.output_documents = 1
            return completed

        future = service._materialize("query", recipe, execute)
        running = copy_message(recipe)
        running.status = status.STATUS_RUNNING
        handle = p.Query(db, running)
        try:
            with patch("premixdb.api.base.time.sleep", side_effect=AssertionError("polled")):
                with pytest.raises(TimeoutError, match="timed out waiting for query"):
                    handle.wait(timeout=0.001)
            assert not future.cancelled() and not future.done()
        finally:
            release.set()
            future.result(timeout=5)
        with patch(
            "premixdb.api.base.time.sleep", side_effect=AssertionError("stale running state")
        ):
            assert handle.wait().profile().output_documents == 1


@pytest.mark.parametrize("fail", [False, True])
def test_repeated_admission_has_bounded_memory_and_allows_retries(fail: bool) -> None:
    materializer = Materializer[int]()
    entered, release, finished = Event(), Event(), Event()
    identity = b"q" * 32

    def work() -> int:
        entered.set()
        if not release.wait(timeout=30):
            raise TimeoutError("work was not released")
        if fail:
            raise ValueError("attempt failed")
        return 42

    try:
        future = materializer.ensure("query", identity, work)
        future.add_done_callback(lambda _: finished.set())
        assert entered.wait(timeout=5)
        tracemalloc.start()
        try:
            for _ in range(10_000):
                assert materializer.ensure("query", identity, work) is future
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 256 * 1024
        release.set()
        if fail:
            with pytest.raises(ValueError, match="attempt failed"):
                future.result(timeout=5)
        else:
            assert future.result(timeout=5) == 42
        assert finished.wait(timeout=5)
        assert materializer.active("query", identity) is None
        retry = materializer.ensure("query", identity, lambda: 43)
        assert retry is not future
        assert retry.result(timeout=5) == 43
    finally:
        release.set()
        materializer.close()
    assert materializer.active("query", identity) is None


@pytest.mark.parametrize("kind", ["query", "dataset"])
@pytest.mark.parametrize("cache_bytes", [0, 128 * 1024 * 1024])
def test_reads_prioritize_completion_and_active_retries(
    tmp_path: Path, kind: Literal["query", "dataset"], cache_bytes: int
) -> None:
    with p.PremixDB(storage=tmp_path, cache_bytes=cache_bytes) as db:
        query = db.Corpus("states", [p.Source("a", "abcd")]).query()
        service = coordinator(db)
        catalog = Catalog(service._storage)
        identity = _decode_id(query.id)
        if kind == "dataset":
            dataset = query.mix(tokenizer=p.ByteTokenizer(), sequence_length=2)[0]
            identity = _decode_id(dataset.id)

        def read(executor: Catalog = service) -> q.Query | d.Dataset:
            if kind == "query":
                return executor.GetQuery(q.GetQueryRequest(id=identity)).query
            return executor.GetDataset(d.GetDatasetRequest(id=identity)).dataset

        recipe = read()
        assert recipe.status == status.STATUS_PENDING
        failed = copy_message(recipe)
        failed.status, failed.error = status.STATUS_ERROR, "previous attempt failed"
        service._storage.save(kind, recipe.id, failed, suffix=".failed", failure=True)
        assert read().error == failed.error

        entered, release = Event(), Event()

        def retry() -> q.Query | d.Dataset:
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("retry was not released")
            return recipe

        future = service._jobs.ensure(kind, recipe.id, retry)
        try:
            assert entered.wait(timeout=5)
            running = read()
            assert running.status == status.STATUS_RUNNING
            assert not running.error
            assert read(catalog).status == status.STATUS_ERROR
            # Publication can finish before the active job leaves the registry.
            completed = copy_message(recipe)
            completed.status = status.STATUS_COMPLETED
            service._storage.save(kind, recipe.id, completed)
            assert read().status == status.STATUS_COMPLETED
            assert read(catalog).status == status.STATUS_COMPLETED
        finally:
            release.set()
            future.result(timeout=5)

        returned = read()
        returned.Clear()
        assert read().status == status.STATUS_COMPLETED
