"""Delayed terminal status for blocking SDK operations, without touching stdout."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from threading import Event, Lock, Thread
from time import monotonic
from typing import Callable, Iterator, TextIO

_DELAY = 1.0
_INTERVAL = 5.0
_ACTIVE = ContextVar("premixdb_progress_active", default=False)
_OUTPUT_LOCK = Lock()


def _write(stream: TextIO, label: str, state: str, started: float) -> None:
    try:
        with _OUTPUT_LOCK:
            print(
                f"[premixdb] {label}: {state} ({monotonic() - started:.1f}s elapsed)",
                file=stream,
                flush=True,
            )
    except (OSError, ValueError):
        # A closed or redirected terminal must not fail a data operation.
        pass


@contextmanager
def operation(label: str, *, enabled: bool = True) -> Iterator[None]:
    """Report while synchronous work blocks, suppressing nested SDK reporters."""
    if not enabled or _ACTIVE.get():
        yield
        return
    token = _ACTIVE.set(True)
    stopped = Event()
    started = monotonic()
    stream = sys.stderr
    reported = False

    def report() -> None:
        nonlocal reported
        if stopped.wait(_DELAY):
            return
        while not stopped.is_set():
            reported = True
            _write(stream, label, "running", started)
            if stopped.wait(_INTERVAL):
                return

    reporter = Thread(target=report, name="premixdb-progress", daemon=True)
    state = "completed"
    try:
        reporter.start()
        try:
            yield
        except BaseException:
            state = "failed"
            raise
    finally:
        stopped.set()
        if reporter.ident is not None:
            reporter.join()
        _ACTIVE.reset(token)
        if reported:
            _write(stream, label, state, started)


def report_progress[**P, R](label: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Preserve a method's complete signature while reporting its progress."""

    def decorate(method: Callable[P, R]) -> Callable[P, R]:
        @wraps(method)
        def call(*args: P.args, **kwargs: P.kwargs) -> R:
            owner = args[0]
            client = getattr(owner, "_db", owner)
            enabled = getattr(client, "_progress_enabled", True)
            if not isinstance(enabled, bool):
                raise TypeError("progress must be boolean")
            id = getattr(owner, "id", "")
            description = label.format(id=id, kind=type(owner).__name__.lower())
            with operation(description, enabled=enabled):
                return method(*args, **kwargs)

        return call

    return decorate
