"""Run the numbered lessons against bounded inputs without model downloads."""

from __future__ import annotations

import ast
import gzip
import importlib.util
import json
import runpy
import subprocess
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import FunctionType
from typing import Protocol, TypedDict, Unpack, cast
from unittest.mock import patch

import pytest

import premixdb as p
from premixdb.enrichment.types import ComputedRow, field
from premixdb.enrichment.types import Document as FeatureDocument
from premixdb.execution import enrichment

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path]:
    web = tmp_path / "web.json.gz"
    with gzip.open(web, "wt", encoding="utf-8") as stream:
        for text in ("Science explains how stars form. " * 20, "A brief page.", ""):
            stream.write(json.dumps({"text": text}) + "\n")
    papers = tmp_path / "papers.jsonl"
    papers.write_text(
        "".join(
            json.dumps({"id": f"paper/{i}", "text": "Scientific paper text. " * (i + 1) * 30})
            + "\n"
            for i in range(3)
        ),
        encoding="utf-8",
    )
    return web, papers


class LessonSettings(TypedDict, total=False):
    INPUT: Path
    WEB: Path
    PAPERS: Path
    LITERATURE: Path
    LIMIT: int


class QualityLesson(Protocol):
    INPUT: Path
    STORAGE: Path

    def main(self) -> None: ...


def lesson_result(
    name: str, storage: Path, **settings: Unpack[LessonSettings]
) -> subprocess.CompletedProcess[str]:
    values = {
        key: str(value) if isinstance(value, Path) else value for key, value in settings.items()
    }
    values["STORAGE"] = str(storage)
    # Configure the lesson's editable Python values without passing it CLI flags.
    program = f"""
import json
import runpy
import sys
from pathlib import Path
from types import FunctionType
sys.path.insert(0, {str(EXAMPLES)!r})
namespace = runpy.run_path({str(EXAMPLES / name)!r}, run_name="tutorial")
main = namespace["main"]
assert isinstance(main, FunctionType)
for key, value in json.loads({json.dumps(values)!r}).items():
    main.__globals__[key] = Path(value) if isinstance(value, str) else value
main()
"""
    return subprocess.run(
        [sys.executable, "-c", program], text=True, capture_output=True, timeout=60
    )


def run_lesson(name: str, storage: Path, **settings: Unpack[LessonSettings]) -> str:
    output = StringIO()
    with patch.object(sys, "path", [str(EXAMPLES), *sys.path]), redirect_stdout(output):
        namespace = runpy.run_path(str(EXAMPLES / name), run_name="tutorial")
        main = namespace["main"]
        assert isinstance(main, FunctionType)
        main.__globals__.update(STORAGE=storage, **settings)
        main()
    return output.getvalue()


@pytest.mark.integration
def test_lesson_entry_point_runs_in_a_fresh_process(tmp_path: Path) -> None:
    result = lesson_result("01_tiny_shakespeare_snapshots.py", tmp_path / "store")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Snapshot unchanged: True" in result.stdout


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("01_tiny_shakespeare_snapshots.py", "Snapshot unchanged: True"),
        ("05_tiny_shakespeare_packing.py", "'input_ids': (2, 64)"),
        ("06_tiny_shakespeare_decontamination.py", "After excluding held-out text: 8"),
        ("10_tiny_shakespeare_resume.py", "next sequence 1 with identical tokens"),
    ],
)
@pytest.mark.integration
def test_bundled_shakespeare_lessons_run_without_preparation(
    tmp_path: Path, name: str, expected: str
) -> None:
    output = run_lesson(name, tmp_path / "store")
    assert expected in output
    if name.startswith("01"):
        assert "One captured speech:" in output


@pytest.mark.integration
def test_web_and_science_lessons_show_filter_dedupe_and_distribution(
    tmp_path: Path, inputs: tuple[Path, Path]
) -> None:
    web, papers = inputs
    storage = tmp_path / "store"
    output = run_lesson("02_c4_filtering.py", storage, INPUT=web)
    assert "Captured documents: 3" in output
    assert "Selected documents: 1" in output
    output = run_lesson("03_c4_deduplication.py", storage, INPUT=web)
    assert "4 → 3" in output
    assert "Removed copies: 1" in output
    assert "Retained sample:" in output
    output = run_lesson("04_s2orc_distributions.py", storage, INPUT=papers)
    assert "Papers inspected: 3" in output
    assert "Mean words per paper: 240.0" in output
    assert "Word-count range: 120 → 360" in output


@pytest.mark.integration
def test_mixture_lessons_use_named_domains_and_reproducible_candidates(
    tmp_path: Path, inputs: tuple[Path, Path]
) -> None:
    web, papers = inputs
    storage = tmp_path / "store"
    output = run_lesson("08_pile_source_mixture.py", storage, WEB=web, PAPERS=papers)
    assert "Planned content tokens: 1024" in output
    for label in ("web/C4", "science/peS2o-train", "literature/Tiny-Shakespeare"):
        assert label in output
    first = run_lesson("09_regmix_candidates.py", storage, WEB=web, PAPERS=papers)
    second = run_lesson("09_regmix_candidates.py", storage, WEB=web, PAPERS=papers)
    # Reopening the same inputs preserves both proposal weights and packed tokens.
    assert first == second
    assert first.count("Candidate ") == 4
    assert "First chosen sequence: 0" in first
    assert "No winning mixture yet" in first


def test_quality_lesson_reuses_real_cache_with_controlled_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    web = tmp_path / "quality.jsonl"
    web.write_text(
        '{"text": "Science explains stars."}\n{"text": "A brief page."}\n',
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(EXAMPLES))

    class Scores:
        definition = {"provider": "tutorial-quality-test", "version": 1}
        fields = (field("quality.educational_value"),)
        calls = 0

        def compute(self, documents: Sequence[FeatureDocument]) -> list[ComputedRow]:
            Scores.calls += 1
            return [
                {
                    "id": doc.id,
                    "quality.educational_value": 2.0 if doc.text.startswith("Science") else 0.5,
                }
                for doc in documents
            ]

    spec = importlib.util.spec_from_file_location(
        "quality_lesson", EXAMPLES / "07_c4_quality_scores.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lesson = cast(QualityLesson, module)
    lesson.INPUT = web
    lesson.STORAGE = tmp_path / "store"
    with patch.object(enrichment, "producer", return_value=Scores()):
        lesson.main()
    assert "Captured → selected documents: 2 → 1" in capsys.readouterr().out
    assert Scores.calls == 1


def test_snapshot_and_query_inline_previews_preserve_fields_and_truncation(tmp_path: Path) -> None:
    # Lesson 01 exercises snapshot previews, whose message type differs from query previews.
    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.corpus("preview", [p.Source("speech", "héllo world")])
        for resource in (snapshot, snapshot.query()):
            row = resource.preview(limit=1, max_characters=3)[0]
            assert row["text"] == "hél"
            assert row["truncated"] is True
            assert row["source_key"] == "speech"
            assert row["corpus_id"] == snapshot.corpus_id
            assert row["ordinal"] == 0


def test_numbered_lessons_use_editable_values_without_argument_parsers() -> None:
    scripts = sorted(EXAMPLES.glob("[0-9][0-9]_*.py"))
    assert [name.name[:2] for name in scripts] == [f"{i:02d}" for i in range(1, 11)]
    for script in scripts:
        tree = ast.parse(script.read_text())
        description = ast.get_docstring(tree)
        assert description is not None
        assert description.strip()
        assert "argparse" not in script.read_text()
        assignments = {
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        assert {"STORAGE", "LIMIT"} <= assignments
        assert "INPUT" in assignments or {"WEB", "PAPERS", "LITERATURE"} <= assignments


@pytest.mark.integration
def test_lesson_invalid_limit_fails_before_creating_storage(tmp_path: Path) -> None:
    result = lesson_result("01_tiny_shakespeare_snapshots.py", tmp_path / "store", LIMIT=0)
    assert result.returncode == 1
    assert "LIMIT must be positive" in result.stderr
    assert not (tmp_path / "store").exists()


@pytest.mark.integration
def test_lesson_missing_input_explains_preparation(tmp_path: Path) -> None:
    result = lesson_result(
        "02_c4_filtering.py", tmp_path / "store", INPUT=tmp_path / "missing.jsonl"
    )
    assert result.returncode == 1
    assert "Missing" in result.stderr and "examples/README.md" in result.stderr
    assert not (tmp_path / "store").exists()
