"""Network readers cannot trap cancellation or publish an incomplete capture."""

from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from threading import Timer
from threading import enumerate as threads
from time import monotonic
from unittest.mock import patch

import pytest

import premixdb as p
from premixdb.runtime import hub_capture
from premixdb.v1.storage_pb2 import Source


@pytest.fixture
def readers() -> Iterator[list[subprocess.Popen[bytes]]]:
    opened: list[subprocess.Popen[bytes]] = []
    popen = subprocess.Popen

    def tracked(command: list[str], *, stdout: int) -> subprocess.Popen[bytes]:
        process = popen(command, stdout=stdout)
        opened.append(process)
        return process

    with patch.object(hub_capture, "Popen", side_effect=tracked):
        yield opened
    assert opened
    assert all(process.poll() is not None for process in opened)
    assert not any(thread.name == "premixdb-hub-pipe" for thread in threads())


def source() -> Source:
    return p.SourceSpec(
        hugging_face=p.HuggingFaceDataset(repository="test/repo", revision="a" * 40, split="train")
    )


def command(program: str) -> list[str]:
    return [sys.executable, "-c", program]


def frame(message: list[str]) -> str:
    return f"print({json.dumps(message)!r}, flush=True)\n"


def test_stream_preserves_text_and_order(readers: list[subprocess.Popen[bytes]]) -> None:
    rows = [("first", "hello\n世界"), ("second", "")]
    program = "".join(frame(["row", key, text]) for key, text in rows) + frame(["done"])
    with patch.object(hub_capture, "_command", return_value=command(program)):
        assert list(hub_capture.capture(source())) == rows


def test_worker_limits_rows_and_keeps_library_output_off_the_pipe(
    readers: list[subprocess.Popen[bytes]], capfd: pytest.CaptureFixture[str]
) -> None:
    request = source()
    request.limit = 2
    encoded = base64.b64encode(request.SerializeToString()).decode("ascii")
    program = f"""
import sys
from types import ModuleType, SimpleNamespace
datasets = ModuleType('datasets')
datasets.config = SimpleNamespace(STREAMING_READ_MAX_RETRIES=20, STREAMING_OPEN_MAX_RETRIES=20)
def rows(*args, **kwargs):
    print('dataset notice')
    assert datasets.config.STREAMING_READ_MAX_RETRIES == 2
    assert datasets.config.STREAMING_OPEN_MAX_RETRIES == 2
    assert kwargs['revision'] == 'a' * 40
    yield {{'text': 'first'}}
    yield {{'text': 'second'}}
    raise AssertionError('read beyond the capture limit')
datasets.load_dataset = rows
sys.modules['datasets'] = datasets
from premixdb.runtime.hub_capture import main
sys.argv = ['reader', {encoded!r}]
main()
"""
    with patch.object(hub_capture, "_command", return_value=command(program)):
        assert list(hub_capture.capture(request)) == [
            ("hf://test/repo//train/0", "first"),
            ("hf://test/repo//train/1", "second"),
        ]
    output = capfd.readouterr()
    assert "dataset notice" in output.err
    assert not output.out


@pytest.mark.parametrize(
    ("program", "error", "message"),
    [
        (frame(["error", "ValueError", "duplicate dataset row key"]), ValueError, "duplicate"),
        (frame(["error", "ConnectionError", "Server Disconnected"]), RuntimeError, "Disconnected"),
        (frame(["unexpected"]), ValueError, "invalid Hub"),
        ("pass", RuntimeError, "without completing"),
    ],
)
def test_reader_failures_propagate(
    readers: list[subprocess.Popen[bytes]], program: str, error: type[Exception], message: str
) -> None:
    with (
        patch.object(hub_capture, "_command", return_value=command(program)),
        pytest.raises(error, match=message),
    ):
        list(hub_capture.capture(source()))


def test_stalled_reader_has_a_deadline(readers: list[subprocess.Popen[bytes]]) -> None:
    started = monotonic()
    with (
        patch.object(hub_capture, "_command", return_value=command("import time; time.sleep(60)")),
        patch.object(hub_capture, "_READ_TIMEOUT", 0.2),
        pytest.raises(TimeoutError, match="test/repo.*stalled"),
    ):
        list(hub_capture.capture(source()))
    assert monotonic() - started < 3


def test_early_close_kills_a_reader_that_ignores_termination(
    readers: list[subprocess.Popen[bytes]],
) -> None:
    program = (
        "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        + frame(["row", "a", "one"])
        + "time.sleep(60)"
    )
    with patch.object(hub_capture, "_command", return_value=command(program)):
        rows = hub_capture.capture(source())
        assert next(rows) == ("a", "one")
        started = monotonic()
        rows.close()
        assert monotonic() - started < 3


def test_sigint_cancels_capture_and_preserves_last_snapshot(
    tmp_path: Path, readers: list[subprocess.Popen[bytes]]
) -> None:
    with p.PremixDB(storage=tmp_path, progress=False) as db:
        previous = db.Corpus("hub", [p.Source("saved", "original")])
        program = frame(["row", "new", "partial"]) + "import time; time.sleep(60)"
        timer = Timer(0.3, lambda: os.kill(os.getpid(), signal.SIGINT))
        handler = signal.getsignal(signal.SIGINT)
        try:
            signal.signal(signal.SIGINT, signal.default_int_handler)
            with patch.object(hub_capture, "_command", return_value=command(program)):
                timer.start()
                started = monotonic()
                with pytest.raises(KeyboardInterrupt):
                    db.Corpus("hub", source=source())
                assert monotonic() - started < 3
        finally:
            timer.cancel()
            timer.join()
            signal.signal(signal.SIGINT, handler)
        assert db.Corpus("hub").id == previous.id
        assert db.Corpus("hub").preview()[0]["text"] == "original"
        assert any(event.error == "KeyboardInterrupt" for event in db._executions())
        with patch.object(
            hub_capture,
            "_command",
            return_value=command(frame(["row", "new", "complete"]) + frame(["done"])),
        ):
            retried = db.Corpus("hub", source=source())
        assert retried.preview()[0]["text"] == "complete"


def test_capture_limit_closes_reader(
    readers: list[subprocess.Popen[bytes]], tmp_path: Path
) -> None:
    program = frame(["row", "a", "one"]) + "import time; time.sleep(60)"
    with (
        patch.object(hub_capture, "_command", return_value=command(program)),
        p.PremixDB(storage=tmp_path, progress=False) as db,
    ):
        assert db.Corpus("hub", source=source(), limit=1).profile().documents == 1
