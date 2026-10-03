"""Terminal status stays live during blocking work and preserves SDK results."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path
from threading import Event
from threading import enumerate as threads
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from _type_support import coordinator, invalid_call

import premixdb as p
from premixdb._progress import operation
from premixdb.v1 import dataset_pb2 as dataset_pb
from premixdb.v1 import query_pb2 as query_pb


class StatusOutput(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.first = Event()
        self.second = Event()
        self.updates = 0

    def write(self, value: str) -> int:
        size = super().write(value)
        if ": running (" in value:
            self.updates += 1
            self.first.set()
            if self.updates >= 2:
                self.second.set()
        return size

    def reset(self) -> None:
        self.seek(0)
        self.truncate()
        self.first.clear()
        self.second.clear()
        self.updates = 0


@pytest.fixture
def output() -> Iterator[StatusOutput]:
    stream = StatusOutput()
    with (
        patch("premixdb._progress._DELAY", 0.01),
        patch("premixdb._progress._INTERVAL", 0.01),
        patch("premixdb._progress.sys", SimpleNamespace(stderr=stream)),
    ):
        yield stream
    assert not any(thread.name == "premixdb-progress" for thread in threads())


def test_reports_repeatedly_before_the_operation_finishes(output: StatusOutput) -> None:
    with operation("Query abc"):
        assert output.first.wait(timeout=3)
        assert "completed" not in output.getvalue()
        assert output.second.wait(timeout=3)
    lines = output.getvalue().splitlines()
    assert len(lines) >= 3
    assert all(line.startswith("[premixdb] Query abc:") for line in lines)
    assert all("s elapsed)" in line for line in lines)
    assert ": completed (" in lines[-1]


def test_fast_and_disabled_operations_are_quiet(output: StatusOutput) -> None:
    with patch("premixdb._progress._DELAY", 60):
        with operation("Fast operation"):
            pass
    with operation("Disabled operation", enabled=False):
        assert not output.first.wait(timeout=0.03)
    assert output.getvalue() == ""


def test_nested_operations_share_one_reporter(output: StatusOutput) -> None:
    with operation("Outer query"):
        with operation("Inner submission"):
            assert output.second.wait(timeout=3)
    assert "Inner submission" not in output.getvalue()
    assert output.getvalue().count(": completed (") == 1


@pytest.mark.parametrize("error", [RuntimeError("broken"), TimeoutError(), KeyboardInterrupt()])
def test_failures_and_interruptions_stop_reporting(
    output: StatusOutput, error: BaseException
) -> None:
    with pytest.raises(type(error)) as caught:
        with operation("Failing query"):
            assert output.first.wait(timeout=3)
            raise error
    assert caught.value is error
    assert ": failed (" in output.getvalue().splitlines()[-1]
    assert "completed" not in output.getvalue()
    # A failed operation must not suppress the next operation's reporter.
    output.first.clear()
    with operation("Next query"):
        assert output.first.wait(timeout=3)
    assert "Next query: completed" in output.getvalue()


def test_closed_output_does_not_fail_work() -> None:
    stream = io.StringIO()
    stream.close()
    with (
        patch("premixdb._progress.sys.stderr", stream),
        patch("premixdb._progress._DELAY", 0),
    ):
        with operation("No terminal"):
            pass


def test_query_profile_reports_while_submission_blocks(
    tmp_path: Path, output: StatusOutput, capsys: pytest.CaptureFixture[str]
) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.corpus("progress", [p.Source("a", "é🌍")])
        query = snapshot.query()
        output.reset()
        execute = coordinator(db).run_query

        def slow_query(recipe: query_pb.Query) -> query_pb.Query:
            assert output.second.wait(timeout=3)
            assert "completed" not in output.getvalue()
            return execute(recipe)

        with patch.object(coordinator(db), "run_query", side_effect=slow_query):
            profile = query.profile()
        assert profile.output_documents == 1
        assert profile.output_characters == 2
        assert f"Waiting for query {query.id}: running" in output.getvalue()
        assert ": completed (" in output.getvalue().splitlines()[-1]
        assert "Executing query:" not in output.getvalue()
        assert capsys.readouterr().out == ""
        before = output.getvalue()
        assert query.profile() == profile
        assert output.getvalue() == before


def test_capture_and_dataset_profiles_report_blocking_work(
    tmp_path: Path, output: StatusOutput
) -> None:
    with p.PremixDB(storage=tmp_path) as db:

        def sources() -> Iterator[p.Source]:
            output.reset()
            assert output.first.wait(timeout=3)
            yield p.Source("a", "hello")

        snapshot = db.corpus("progress", sources())
        assert "Capturing snapshot: running" in output.getvalue()
        assert snapshot.profile().documents == 1
        dataset = snapshot.query().dataset(tokenizer=p.ByteTokenizer(), sequence_length=4)
        output.first.clear()
        compute = coordinator(db)._planned_dataset_profile

        def slow_profile(resource: dataset_pb.Dataset) -> dataset_pb.DatasetProfile:
            assert output.first.wait(timeout=3)
            return compute(resource)

        with patch.object(coordinator(db), "_planned_dataset_profile", side_effect=slow_profile):
            assert dataset.profile().content_tokens == 5
        assert f"Profiling dataset {dataset.id}: running" in output.getvalue()


def test_progress_can_be_disabled_for_the_session(tmp_path: Path, output: StatusOutput) -> None:
    with p.PremixDB(storage=tmp_path, progress=False) as db:
        snapshot = db.corpus("quiet", [p.Source("a", "hello")])
        query = snapshot.query()
        execute = coordinator(db).run_query

        def slow_query(recipe: query_pb.Query) -> query_pb.Query:
            assert not output.first.wait(timeout=0.03)
            return execute(recipe)

        with patch.object(coordinator(db), "run_query", side_effect=slow_query):
            assert query.profile().output_documents == 1
    assert output.getvalue() == ""
    with pytest.raises(TypeError, match="progress must be boolean"):
        invalid_call(p.PremixDB, storage=tmp_path, progress="yes")
