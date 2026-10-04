"""The terminal interface reuses published resources and closes its shells."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from _type_support import SHELL_SOURCES

import premixdb as p
from premixdb.cli.main import main


@pytest.fixture
def shell_store(tmp_path: Path) -> Path:
    storage = tmp_path / "store"
    with p.PremixDB(storage=storage, progress=False) as db:
        db.Corpus("demo", SHELL_SOURCES).wait()
    return storage


@pytest.mark.parametrize("ending", ["exit", "exit()", "quit", "quit()", ""])
@pytest.mark.integration
def test_shell_closes_on_exit_quit_and_eof(shell_store: Path, ending: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "premixdb", "--storage", str(shell_store), "shell"],
        input=(
            "original_close = db.close\n"
            "db.close = lambda close=original_close, db=db: (close(), print('DATABASE_CLOSED', db._closed))\n"
            + (ending + "\n" if ending else "")
        ),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DATABASE_CLOSED True\n" in result.stdout


@pytest.mark.parametrize("name", ["SIGTERM", "SIGHUP"])
@pytest.mark.integration
def test_shell_closes_on_termination_signals(shell_store: Path, name: str) -> None:
    if not hasattr(signal, name):
        pytest.skip(f"{name} is unavailable")
    result = subprocess.run(
        [sys.executable, "-m", "premixdb", "--storage", str(shell_store), "shell"],
        input=(
            "original_close = db.close\n"
            "db.close = lambda close=original_close, db=db: (close(), print('DATABASE_CLOSED', db._closed))\n"
            f"import os, signal; os.kill(os.getpid(), signal.{name})\n"
        ),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 128 + getattr(signal, name), result.stdout + result.stderr
    assert "DATABASE_CLOSED True\n" in result.stdout


@pytest.mark.parametrize("error", [SystemExit(0), EOFError(), KeyboardInterrupt(), RuntimeError()])
def test_shell_closes_when_the_repl_raises(shell_store: Path, error: BaseException) -> None:
    seen = []
    previous = {
        name: signal.getsignal(getattr(signal, name))
        for name in ("SIGTERM", "SIGHUP")
        if hasattr(signal, name)
    }

    def interact(namespace: dict[str, object], *, banner: str, history: Path | None) -> None:
        db = namespace["db"]
        assert isinstance(db, p.PremixDB)
        db.close = Mock(wraps=db.close)
        seen.append(db)
        raise error

    with patch("premixdb.cli.shell._interact", side_effect=interact), pytest.raises(type(error)):
        main(["--storage", str(shell_store), "shell"])
    assert seen[0]._closed
    seen[0].close.assert_called_once_with()
    for name, handler in previous.items():
        assert signal.getsignal(getattr(signal, name)) == handler


def test_ipython_completes_only_public_names_and_saves_history(
    tmp_path: Path, shell_store: Path
) -> None:
    import sqlite3

    from IPython.terminal.ptutils import IPythonPTCompleter
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    from premixdb.cli.shell import _interact, _Shell

    with p.PremixDB(storage=shell_store) as db:
        query = db.Corpus("completion", [p.Source("a", "hello")]).query()
        history = tmp_path / "history"

        def interact(shell: _Shell) -> None:
            assert shell.colors == "linux" and shell.true_color
            assert shell.Completer.use_jedi is False
            completer = IPythonPTCompleter(shell.Completer, shell=shell)

            def matches(line: str) -> set[str]:
                return {
                    completion.text.rsplit(".", 1)[-1].removesuffix("(")
                    for completion in completer.get_completions(
                        Document(line), CompleteEvent(completion_requested=True)
                    )
                }

            assert matches("db.") == {"version", "close", "Corpus"}
            assert matches("q.") == {"id", "status", "preview", "profile", "mix", "wait"}
            assert matches("db._") == matches("q._") == set()
            assert matches("%") == matches("%%") == matches("%ti") == set()
            assert "print" in matches("pri")
            assert not any(value.startswith("%") for value in matches(""))
            shell.run_cell("answer = 42", store_history=True)

        with patch.object(_Shell, "mainloop", new=interact):
            _interact(dict(db=db, p=p, q=query), banner="test shell", history=history)
        with closing(sqlite3.connect(str(history) + ".sqlite3")) as connection:
            assert connection.execute("SELECT source FROM history").fetchall() == [("answer = 42",)]


@pytest.mark.integration
def test_shell_starts_in_a_fresh_process(shell_store: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "premixdb", "--storage", str(shell_store), "shell"],
        input="assert db.Corpus('demo').profile().documents == 2; print('DEMO_OK')\nexit\n",
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DEMO_OK" in result.stdout
    assert "IPython" in result.stdout
    assert "Traceback" not in result.stdout + result.stderr


def test_cli_reads_metadata_profiles_and_bounded_previews(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.Corpus("cli", [p.Source("first", "hello world"), p.Source("second", "bye")])
        query = snapshot.query()
        dataset = query.mix(tokenizer=p.ByteTokenizer(), sequence_length=4)[0].wait()
    prefix = ["--storage", str(tmp_path)]
    with patch(
        "premixdb.runtime.coordinator.Coordinator._execute_query", side_effect=AssertionError
    ):
        assert main([*prefix, "corpora", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)[0]["snapshot_id"] == snapshot.id
        assert main([*prefix, "profile", "query", "--json", "--", query.id]) == 0
        assert json.loads(capsys.readouterr().out)["output_documents"] == "2"
        assert (
            main(
                [
                    *prefix,
                    "profile",
                    "snapshot",
                    "--field",
                    "text.characters",
                    "--json",
                    "--",
                    snapshot.id,
                ]
            )
            == 0
        )
        assert json.loads(capsys.readouterr().out)["mean"] == 7.0
        assert (
            main(
                [
                    *prefix,
                    "preview",
                    "query",
                    "--limit",
                    "1",
                    "--max-characters",
                    "2",
                    "--json",
                    "--",
                    query.id,
                ]
            )
            == 0
        )
        rows = json.loads(capsys.readouterr().out)
        assert len(rows) == 1 and len(rows[0]["text"]) == 2 and rows[0]["truncated"]
        assert (
            main([*prefix, "preview", "dataset", "--limit", "1", "--json", "--", dataset.id]) == 0
        )
        rows = json.loads(capsys.readouterr().out)
        assert len(rows) == 1 and rows[0]["ordinal"] == 0 and rows[0]["tokens"]
        assert main([*prefix, "profile", "dataset", "--", dataset.id]) == 0
        summary = capsys.readouterr().out
        assert "Sequences:" in summary
        assert len(summary.splitlines()) <= 15
        assert main([*prefix, "executions", "--resource-id=" + query.id, "--json"]) == 0
        assert json.loads(capsys.readouterr().out)


def test_shells_supply_the_existing_python_api_and_close_storage(tmp_path: Path) -> None:
    seen = []

    def interact(namespace: dict[str, object], *, banner: str, history: Path | None) -> None:
        assert namespace["p"] is p
        db = namespace["db"]
        assert isinstance(db, p.PremixDB)
        demo = db.Corpus("demo")
        assert len(demo.preview()) == 2
        assert any("Citizen:" in row["text"] for row in demo.preview(limit=100))
        retained = demo.query(steps=[p.where(p.text.characters > 0), p.dedupe()]).profile()
        assert retained.output_documents == 2
        assert db.Corpus("shell", [p.Source("a", "hello")]).preview()[0]["text"] == "hello"
        seen.append(db)

    with (
        patch("premixdb.cli.main._demo_sources", return_value=SHELL_SOURCES) as demo_sources,
        patch("premixdb.cli.shell._interact", side_effect=interact),
    ):
        assert main(["--storage", str(tmp_path), "shell"]) == 0
    demo_sources.assert_called_once_with()
    assert len(seen) == 1 and all(db._closed for db in seen)
    with p.PremixDB(storage=tmp_path) as db:
        assert db.Corpus("demo").profile().documents == len(SHELL_SOURCES)


def test_shell_upgrades_previous_builtin_demo(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        previous = db.Corpus(
            "demo",
            [
                p.Source("science", "Science explains how stars form and planets move."),
                p.Source("science-copy", "Science explains how stars form and planets move."),
                p.Source("code", "Code turns ideas into programs you can run."),
                p.Source("empty", ""),
            ],
        )

    def interact(namespace: dict[str, object], *, banner: str, history: Path | None) -> None:
        db = namespace["db"]
        assert isinstance(db, p.PremixDB)
        demo = db.Corpus("demo")
        assert demo.id != previous.id
        assert demo.profile().documents == len(SHELL_SOURCES)
        assert any("Citizen:" in row["text"] for row in demo.preview(limit=100))
        assert db._snapshot(previous.id).profile().documents == 4

    with (
        patch("premixdb.cli.main._demo_sources", return_value=SHELL_SOURCES) as demo_sources,
        patch("premixdb.cli.shell._interact", side_effect=interact),
    ):
        assert main(["--storage", str(tmp_path), "shell"]) == 0
    demo_sources.assert_called_once_with()


@pytest.mark.parametrize(("read_only", "existing"), [(False, True), (True, True), (True, False)])
def test_shell_preserves_existing_demo_and_read_only_storage(
    tmp_path: Path, read_only: bool, existing: bool
) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        if existing:
            saved = db.Corpus("demo", [p.Source("custom", "My own demo.")])

    def interact(namespace: dict[str, object], *, banner: str, history: Path | None) -> None:
        db = namespace["db"]
        assert isinstance(db, p.PremixDB)
        if existing:
            assert db.Corpus("demo").id == saved.id
            assert db.Corpus("demo").preview()[0]["text"] == "My own demo."
            assert "db.Corpus('demo').preview()" in banner
        else:
            assert db.Corpus.list() == []
            assert "db.Corpus.list()" in banner

    arguments = ["--storage", str(tmp_path)]
    if read_only:
        arguments.append("--read-only")
    with patch("premixdb.cli.shell._interact", side_effect=interact):
        assert main([*arguments, "shell"]) == 0


def test_cli_errors_are_short_and_do_not_start_a_shell(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prefix = ["--storage", str(tmp_path)]
    with pytest.raises(SystemExit) as invalid:
        main([*prefix, "profile", "query", "not-an-id"])
    assert invalid.value.code == 1
    assert "premixdb:" in capsys.readouterr().err
    assert main([]) == 0
    assert "profile" in capsys.readouterr().out


@pytest.mark.parametrize("missing", ["index", "tokens"])
def test_missing_sequence_files_have_a_short_cli_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], missing: str
) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        dataset = (
            db.Corpus("missing", [p.Source("a", "abcd" * 11)])
            .query()
            .mix(tokenizer=p.ByteTokenizer(), sequence_length=4)[0]
            .wait()
        )
        reference = dataset._proto.sequences[0] if missing == "index" else dataset._proto.tokens[0]
        path = tmp_path / "dataset/objects" / reference.object.blake3_digest.hex()
        path.unlink()
    with pytest.raises(SystemExit) as failed:
        main(
            [
                "--storage",
                str(tmp_path),
                "preview",
                "dataset",
                dataset.id,
                "--offset",
                "10",
                "--limit",
                "1",
            ]
        )
    assert failed.value.code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.startswith("premixdb: ")
    assert str(path) in output.err
    assert len(output.err.splitlines()) == 1
