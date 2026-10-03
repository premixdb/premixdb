"""Shared admission and execution for immutable recipes, with in-flight deduplication."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import cast


class SingleFlight:
    def __init__(self) -> None:
        self._lock = Lock()
        self._active: dict[Hashable, Future[object]] = {}

    def run[R](self, key: Hashable, work: Callable[[], R]) -> R:
        # A key identifies one work/result contract. The registry erases that
        # result type only while storing jobs with different contracts together.
        with self._lock:
            existing = self._active.get(key)
            owner = existing is None
            future = cast(Future[R], existing) if existing is not None else Future[R]()
            if owner:
                self._active[key] = cast(Future[object], future)
        if owner:
            try:
                future.set_result(work())
            except BaseException as exc:
                future.set_exception(exc)
            finally:
                with self._lock:
                    del self._active[key]
        return future.result()

    def __len__(self) -> int:
        with self._lock:
            return len(self._active)


class Materializer[R]:
    """One pool for queries and datasets; completed artifacts live in storage.

    Registry entries exist only while work is active. Recipes and logical IDs
    survive retries and restarts; scheduling and attempts do not enter identity.
    """

    def __init__(self, workers: int = 1) -> None:
        if type(workers) is not int or workers <= 0:
            raise ValueError("workers must be a positive integer")
        self._pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="premixdb-materialize"
        )
        self._lock = Lock()
        self._active: dict[tuple[str, bytes], Future[R]] = {}

    def ensure(self, kind: str, id: bytes, work: Callable[[], R]) -> Future[R]:
        key = kind, id
        with self._lock:
            future = self._active.get(key)
            if future is None:
                future = self._pool.submit(work)
                self._active[key] = future
        # Register outside the lock: already finished futures invoke callbacks inline.
        future.add_done_callback(lambda done: self._forget(key, done))
        return future

    def _forget(self, key: tuple[str, bytes], future: Future[R]) -> None:
        with self._lock:
            if self._active.get(key) is future:
                del self._active[key]

    def active(self, kind: str, id: bytes) -> Future[R] | None:
        with self._lock:
            return self._active.get((kind, id))

    def close(self) -> None:
        self._pool.shutdown(wait=True)
