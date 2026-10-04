"""A small terminal interface over the existing Python resource API."""

from __future__ import annotations

import argparse
import json
import signal
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from importlib.resources import files
from pprint import pprint
from threading import current_thread, main_thread
from types import FrameType
from typing import Iterator

from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message

import premixdb as p
from premixdb.api.collections import _pages
from premixdb.contracts import JSON, is_json
from premixdb.schemas.ids import _encode_id
from premixdb.v1 import corpus_pb2 as c

_LEGACY_DEMO_SOURCES = (
    p.Source("science", "Science explains how stars form and planets move."),
    p.Source("science-copy", "Science explains how stars form and planets move."),
    p.Source("code", "Code turns ideas into programs you can run."),
    p.Source("empty", ""),
)


def _demo_sources() -> list[p.Source]:
    text = files("premixdb").joinpath("data/tiny_shakespeare.txt").read_text(encoding="utf-8")
    blocks = text.split("\n\n")
    return [
        p.Source(f"speech/{i:04d}", block + ("\n\n" if i < len(blocks) - 1 else ""))
        for i, block in enumerate(blocks)
        if block or i < len(blocks) - 1
    ]


def _builtin_demo(demo: p.Snapshot) -> bool:
    count = demo.profile().documents
    if count == len(_LEGACY_DEMO_SOURCES):
        expected = {source.key: source.text for source in _LEGACY_DEMO_SOURCES}
    elif count == 9:
        excerpt = (
            files("premixdb")
            .joinpath("data/tiny_shakespeare_excerpt.txt")
            .read_text(encoding="utf-8")
        )
        expected = {
            f"speech/{i:04d}": block + "\n\n"
            for i, block in enumerate(excerpt.strip().split("\n\n"))
        }
    else:
        return False
    return {
        row["source_key"]: row["text"] for row in demo.preview(limit=count, max_characters=1024)
    } == expected


def _shell_banner(db: p.PremixDB) -> str:
    writable_local = not db._read_only
    try:
        demo = db.Corpus("demo")
    except ValueError:
        if not writable_local:
            return "premixdb: db is open; p is premixdb. Try db.Corpus.list()."
        demo = None
    if writable_local and (demo is None or _builtin_demo(demo)):
        db.Corpus("demo", _demo_sources())
    return "premixdb: db is open; p is premixdb. Try db.Corpus('demo').preview()."


@contextmanager
def _shell_exit_signals() -> Iterator[None]:
    """Let termination unwind the console and close the database."""
    if current_thread() is not main_thread():
        yield
        return
    previous, stopping = {}, None

    def stop(signum: int, frame: FrameType | None) -> None:
        nonlocal stopping
        stopping = signum
        raise SystemExit(128 + signum)

    try:
        for name in ("SIGTERM", "SIGHUP"):
            if (signum := getattr(signal, name, None)) is not None:
                previous[signum] = signal.signal(signum, stop)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if stopping is not None:
        raise SystemExit(128 + stopping)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="premixdb",
        description="Explore local data from Python or the terminal.",
    )
    parser.add_argument("--storage", help="local database directory")
    parser.add_argument("--read-only", action="store_true", help="disable compute and writes")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("shell", help="open a shell with db and p already available")
    corpora = commands.add_parser("corpora", help="list published corpus names and snapshot IDs")
    profile = commands.add_parser("profile", help="read a published resource's statistics")
    profile.add_argument("kind", choices=("snapshot", "query", "dataset"))
    profile.add_argument("id", help="published resource ID")
    profile.add_argument("--field", help="summarize one snapshot/query field, e.g. text.characters")
    preview = commands.add_parser("preview", help="browse published documents or packed sequences")
    preview.add_argument("kind", choices=("snapshot", "query", "dataset"))
    preview.add_argument("id", help="published resource ID")
    preview.add_argument("--limit", type=int, default=3)
    preview.add_argument("--offset", type=int, default=0)
    preview.add_argument("--max-characters", type=int, default=1024)
    executions = commands.add_parser("executions", help="show execution history")
    executions.add_argument("--resource-id")
    for command in (corpora, profile, preview, executions):
        command.add_argument("--json", action="store_true", help="emit JSON for scripts")
    return parser


def _json_value(value: object) -> JSON:
    if isinstance(value, Message):
        # Protobuf JSON keeps uint64 counts as strings, without losing precision.
        result: object = MessageToDict(value, preserving_proto_field_name=True)
        if is_json(result):
            return result
        raise TypeError("protobuf serializer returned an invalid JSON value")
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        result: dict[str, JSON] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int)):
                raise TypeError("JSON mapping keys must be strings or integers")
            result[str(key)] = _json_value(item)
        return result
    if is_json(value):
        return value
    raise TypeError(f"cannot serialize {type(value).__name__} as JSON")


def _display(value: object, json_output: bool) -> None:
    if json_output:
        print(json.dumps(_json_value(value), ensure_ascii=False, indent=2))
    elif isinstance(value, Message):
        print(value)
    else:
        pprint(value, sort_dicts=False)


def _corpora(db: p.PremixDB) -> list[dict[str, str]]:
    return [
        dict(
            name=corpus.name,
            id=_encode_id(corpus.id),
            snapshot_id=_encode_id(db._get("Corpus", corpus.id).latest_snapshot_id),
        )
        for corpus in _pages(
            db,
            db._executor.ListCorpus,
            c.ListCorpusRequest(),
            lambda response: response.corpora,
        )
    ]


def main(argv: list[str] | None = None) -> int:
    """Run a terminal command, closing the database before returning."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "profile" and args.field and args.kind == "dataset":
        parser.error("--field summarizes snapshots or queries; datasets have packing profiles")
    try:
        with p.PremixDB(
            storage=args.storage,
            read_only=True if args.read_only or args.command != "shell" else None,
        ) as db:
            if args.command == "shell":
                with _shell_exit_signals():
                    from pathlib import Path

                    from premixdb.cli.shell import _interact

                    _interact(
                        dict(db=db, p=p),
                        banner=_shell_banner(db)
                        + "\nTab: completion; Up/Down: history; Ctrl-R: history search.",
                        history=Path(db._storage) / ".shell_history",
                    )
            elif args.command == "corpora":
                _display(_corpora(db), args.json)
            elif args.command == "executions":
                _display(db._executions(args.resource_id), args.json)
            else:
                resource = getattr(db, "_" + args.kind)(args.id)
                if args.command == "profile":
                    if args.field:
                        from premixdb.api.profiles import _describe_field

                        value = _describe_field(resource.profile().fields, args.field)
                    else:
                        value = resource.profile()
                else:
                    value = resource.preview(
                        limit=args.limit, offset=args.offset, max_characters=args.max_characters
                    )
                _display(value, args.json)
    except (
        ValueError,
        KeyError,
        OSError,
        sqlite3.Error,
        p.ExecutionError,
        NotImplementedError,
    ) as exc:
        parser.exit(1, f"premixdb: {exc}\n")
    return 0
