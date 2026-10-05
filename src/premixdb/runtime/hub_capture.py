"""Stream Hub rows with an interruptible parent and a disposable network reader."""

from __future__ import annotations

import base64
import json
import sys
from collections.abc import Generator
from contextlib import redirect_stdout
from itertools import islice
from queue import Empty, Full, Queue
from subprocess import PIPE, Popen, TimeoutExpired
from threading import Event, RLock, Thread
from time import monotonic
from typing import BinaryIO

from premixdb.v1.storage_pb2 import Source

_READ_TIMEOUT = 60.0


def _rows(source: Source) -> Generator[tuple[str, str], None, None]:
    from datasets import load_dataset

    policy = source.hugging_face
    dataset = load_dataset(
        policy.repository,
        policy.configuration or None,
        split=policy.split,
        revision=policy.revision,
        streaming=True,
    )
    seen = set()
    for ordinal, row in enumerate(
        islice(dataset, source.limit if source.HasField("limit") else None)
    ):
        text = row.get(policy.text_column or "text")
        if not isinstance(text, str):
            raise ValueError("dataset text column must contain strings")
        key = str(row[policy.key_column]) if policy.key_column else str(ordinal)
        if key in seen:
            raise ValueError("duplicate dataset row key")
        seen.add(key)
        yield f"hf://{policy.repository}/{policy.configuration}/{policy.split}/{key}", text


def _command(source: Source) -> list[str]:
    return [
        sys.executable,
        "-m",
        "premixdb.runtime.hub_capture",
        base64.b64encode(source.SerializeToString()).decode("ascii"),
    ]


def _read(stream: BinaryIO, messages: Queue[bytes | None], stopped: Event) -> None:
    while not stopped.is_set():
        line = stream.readline()
        while not stopped.is_set():
            try:
                messages.put(line or None, timeout=0.1)
                break
            except Full:
                continue
        if not line:
            return


def capture(source: Source) -> Generator[tuple[str, str], None, None]:
    """Stop the reader on Ctrl-C, a stalled read, failure, or an early row limit."""
    process = Popen(_command(source), stdout=PIPE)
    assert process.stdout is not None
    messages: Queue[bytes | None] = Queue(maxsize=1)
    stopped = Event()
    reader = Thread(
        target=_read,
        args=(process.stdout, messages, stopped),
        name="premixdb-hub-pipe",
        daemon=True,
    )
    try:
        reader.start()
        while True:
            deadline = monotonic() + _READ_TIMEOUT
            while True:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Hub capture for {source.hugging_face.repository!r} stalled for "
                        f"{_READ_TIMEOUT:g}s while waiting for data; retry the capture"
                    )
                try:
                    line = messages.get(timeout=min(0.1, remaining))
                    break
                except Empty:
                    continue
            if line is None:
                raise RuntimeError("Hub capture reader exited without completing the stream")
            message: object = json.loads(line)
            if message == ["done"]:
                return
            if isinstance(message, list) and len(message) == 3:
                kind, key, text = message
                if (
                    not isinstance(kind, str)
                    or not isinstance(key, str)
                    or not isinstance(text, str)
                ):
                    raise ValueError("invalid Hub capture reader message")
                if kind == "row":
                    yield key, text
                    continue
                if kind == "error":
                    error = {"ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError}
                    raise error.get(key, RuntimeError)(text)
            raise ValueError("invalid Hub capture reader message")
    finally:
        stopped.set()
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=1)
        except TimeoutExpired:
            process.kill()
            process.wait()
        if reader.ident is not None:
            reader.join(timeout=1)
        process.stdout.close()


def main() -> None:
    source = Source.FromString(base64.b64decode(sys.argv[1]))
    output = sys.stdout
    # Dataset libraries may print status to stdout; reserve this pipe for rows.
    with redirect_stdout(sys.stderr):
        try:
            from tqdm import tqdm

            # This reader has no download subprocesses. A thread lock avoids
            # allocating multiprocessing semaphores that termination would orphan.
            tqdm.set_lock(RLock())
            from datasets import config

            # Avoid multiplying the Hub's own request retries by twenty more
            # dataset retries. These settings affect only this disposable reader.
            setattr(config, "STREAMING_READ_MAX_RETRIES", 2)
            setattr(config, "STREAMING_OPEN_MAX_RETRIES", 2)
            for key, text in _rows(source):
                print(json.dumps(["row", key, text]), file=output, flush=True)
            print(json.dumps(["done"]), file=output, flush=True)
        except Exception as exc:
            print(json.dumps(["error", type(exc).__name__, str(exc)]), file=output, flush=True)


if __name__ == "__main__":
    main()
