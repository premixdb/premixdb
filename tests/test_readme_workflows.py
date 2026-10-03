"""The README's public workflows, using bounded Hub and model fixtures offline."""

from __future__ import annotations

import ast
import re
import tempfile
import unittest
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from _type_support import SHELL_SOURCES, invalid_call

import premixdb as p
from premixdb._typing import JSON, FieldValue
from premixdb.enrichment.types import ComputedRow, field
from premixdb.enrichment.types import Document as FeatureDocument
from premixdb.execution import enrichment
from premixdb.internal import derivation_pb2 as d
from premixdb.v1 import field_pb2 as f
from premixdb.v1 import query_pb2 as q


class TutorialModels:
    """Keep inference controlled while exercising real planning and persistence."""

    definition = {"provider": "readme-test", "version": 1}
    cache_scope = "document"

    def __init__(self, policy: d.EnrichmentProducer) -> None:
        if policy.HasField("language"):
            self.fields = (
                field("language.en"),
                field("language.fr"),
                field("language.label", element_type=f.VALUE_STRING),
            )
        elif policy.model.kind == d.ModelProducer.TOPIC:
            self.fields = (field("weborganizer.topic", width=24, classes=tuple(p.Topic)),)
        elif policy.model.kind == d.ModelProducer.CONTENT_TYPE:
            self.fields = (
                field("weborganizer.content_type", width=24, classes=tuple(p.ContentType)),
            )
        else:
            self.fields = (field("quality.educational_value"),)

    def compute(self, documents: Sequence[FeatureDocument]) -> list[ComputedRow]:
        result: list[ComputedRow] = []
        for doc in documents:
            scientific = doc.text.startswith("Science")
            values: dict[str, FieldValue] = {
                "language.en": 0.95 if doc.text else None,
                "language.fr": 0.05 if doc.text else None,
                "language.label": "en" if doc.text else None,
                "quality.educational_value": 2.0 if scientific else 0.5,
                "weborganizer.topic": [
                    10.0
                    if label
                    is (p.Topic.SCIENCE_AND_TECH if scientific else p.Topic.EDUCATION_AND_JOBS)
                    else 0.0
                    for label in p.Topic
                ],
                "weborganizer.content_type": [
                    10.0
                    if label
                    is (p.ContentType.TUTORIAL if scientific else p.ContentType.NEWS_ARTICLE)
                    else 0.0
                    for label in p.ContentType
                ],
            }
            result.append({"id": doc.id, **{spec.name: values[spec.name] for spec in self.fields}})
        return result


class ReadmeWorkflows(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.db = p.PremixDB(storage=self.root)
        self.addCleanup(self.db.close)

    def test_package_reads_its_current_version(self) -> None:
        from importlib.metadata import PackageNotFoundError, version

        from premixdb._version import current_version

        self.assertEqual(p.__version__, version("premixdb"))
        self.assertEqual(self.db.version, p.__version__)
        with patch("premixdb._version.version", side_effect=PackageNotFoundError):
            import tomllib

            project = Path(__file__).resolve().parents[1] / "pyproject.toml"
            expected = tomllib.loads(project.read_text())["project"]["version"]
            self.assertEqual(current_version(), expected)

    def test_hub_defaults_pin_revision_and_capture_only_the_requested_rows(self) -> None:
        consumed = []

        def rows() -> Iterator[dict[str, str]]:
            for i in range(3):
                consumed.append(i)
                yield {"text": f"document {i}"}

        api = Mock()
        api.dataset_info.return_value = SimpleNamespace(sha="a" * 40)
        with (
            patch("huggingface_hub.HfApi", return_value=api),
            patch("datasets.load_dataset", side_effect=lambda *args, **kwargs: rows()) as load,
        ):
            snapshot = self.db.corpus("c4", source=p.HuggingFaceSource("allenai", "c4"), limit=2)
        self.assertEqual(consumed, [0, 1])
        self.assertEqual(snapshot.profile().documents, 2)
        self.assertEqual(load.call_args.args, ("allenai/c4", "en"))
        self.assertEqual(
            load.call_args.kwargs, dict(split="train", revision="a" * 40, streaming=True)
        )
        api.dataset_info.assert_called_once_with("allenai/c4", revision="main")
        self.assertEqual(self.db.corpus("c4").id, snapshot.id)
        source = (
            p.HuggingFaceSource(
                "org/data",
                configuration="small",
                split="validation",
                revision="b" * 40,
                text_column="body",
                key_column="key",
            )
            ._to_proto()
            .hugging_face
        )
        self.assertEqual(
            (source.repository, source.configuration, source.split),
            ("org/data", "small", "validation"),
        )

    def test_named_snapshots_survive_restart_and_do_not_follow_later_captures(self) -> None:
        with self.assertRaisesRegex(ValueError, "capture with"):
            self.db.corpus("absent")
        first = self.db.corpus("history", [p.Source("a", "first")])
        opened = self.db.corpus("history")
        second = self.db.corpus("history", [p.Source("a", "second")], base=first)
        self.assertIsInstance(opened, p.Snapshot)
        self.assertEqual(opened.preview()[0]["text"], "first")
        self.assertNotEqual(first.id, second.id)
        with self.assertRaises(ValueError):
            self.db.corpus("history", [p.Source("a", "bad"), p.Source("a", "duplicate")])
        self.db.close()
        with p.PremixDB(storage=self.root, cache_bytes=0) as reopened:
            self.assertEqual(reopened.corpus("history").id, second.id)
            self.assertEqual(reopened._snapshot(first.id).preview()[0]["text"], "first")

    def test_wrong_base_is_rejected_before_consuming_or_resolving_sources(self) -> None:
        base = self.db.corpus("base", [])
        target = self.db._create_corpus("target")
        consumed = []

        def sources() -> Iterator[p.Source]:
            consumed.append("read")
            yield p.Source("a", "unused")

        with self.assertRaisesRegex(ValueError, "base snapshot must belong to the same corpus"):
            target.snapshot(source=sources(), base=base)
        self.assertEqual(consumed, [])
        with (
            patch("huggingface_hub.HfApi", side_effect=AssertionError("resolved metadata")),
            patch.object(base, "wait", side_effect=AssertionError("waited")),
            self.assertRaisesRegex(ValueError, "base snapshot must belong to the same corpus"),
        ):
            target.snapshot(source=p.HuggingFaceSource("org/data"), base=base)

    def test_query_mix_inspect_torch_and_lineage_work_together(self) -> None:
        from torch.utils.data import DataLoader

        sources = [
            p.Source(
                "science", "Science explains how stars form and how planets move around them."
            ),
            p.Source(
                "education",
                "Learning works best when students discuss ideas and practise new skills daily.",
            ),
            p.Source("short", "too short"),
        ]
        real_producer = enrichment.producer
        with patch.object(
            enrichment,
            "producer",
            side_effect=lambda policy: (
                real_producer(policy) if policy.HasField("datatrove") else TutorialModels(policy)
            ),
        ):
            snapshot = self.db.corpus("c4", sources)
            steps = [p.where(p.language.en > 0.8), p.where(p.datatrove.n_words >= 10)]
            query = self.db.corpus("c4").query(steps=steps)
            self.assertEqual(query.id, snapshot.query(steps=steps).id)
            self.assertEqual(query.profile().output_documents, 2)
            mixtures = query.mix(
                domains=p.Topic,
                sampler=p.RegMixSampler(),
                n_candidates=3,
                tokenizer=p.ByteTokenizer(),
                tokens=32,
                sequence_length=64,
            )
            self.assertEqual(len(mixtures), 3)
            self.assertEqual(
                set(mixtures.weights[0]),
                {p.Topic.SCIENCE_AND_TECH.value, p.Topic.EDUCATION_AND_JOBS.value},
            )
            self.assertEqual(len(mixtures.profile()), 3)
            self.assertIs(mixtures[0].status, p.ExecutionStatus.PENDING)
            batch = next(iter(DataLoader(mixtures[0].torch(), batch_size=2)))
            self.assertEqual(batch["input_ids"].shape[1], 64)
            self.assertTrue((batch["labels"] == -100).any())
            tutorial = snapshot.query(
                steps=[p.where(p.content_type.label == p.ContentType.TUTORIAL)]
            )
            self.assertEqual(tutorial.profile().output_documents, 1)
            educational = snapshot.query(
                steps=[
                    p.where(p.topic.label == p.Topic.SCIENCE_AND_TECH),
                    p.where(p.quality.educational_value >= 1.0),
                ]
            )
            self.assertEqual(educational.preview()[0]["source_key"], "science")
            language_mix = snapshot.query().mix(
                domains=p.Language, tokenizer=p.ByteTokenizer(), tokens=8, sequence_length=4
            )
            self.assertEqual(language_mix.weights, [{"en": 1.0}] * 3)
            sequence = query.dataset(tokenizer=p.ByteTokenizer(), sequence_length=64)[0]
            ids = sequence.document_ids()
            joined = snapshot.query(steps=[p.where(p.document_id.is_in(ids))]).preview()
            self.assertEqual({row["id"] for row in joined}, set(ids))
            self.assertTrue(all(query._provenance()[id]["source_key"] for id in ids))
            mix_id = mixtures.id
        self.db.close()
        with (
            p.PremixDB(storage=self.root) as reopened,
            patch.object(enrichment, "producer", side_effect=AssertionError("recomputed models")),
        ):
            query = reopened.corpus("c4").query(steps=steps)
            self.assertEqual(
                query.mix(
                    domains=p.Topic,
                    n_candidates=3,
                    tokenizer=p.ByteTokenizer(),
                    tokens=32,
                    sequence_length=64,
                ).id,
                mix_id,
            )
            self.assertEqual(query.profile().output_documents, 2)
            with self.assertRaises(TypeError):
                invalid_call(query.mix, strata=p.Topic)

    def test_capture_limits_preview_pages_and_document_selection_are_deterministic(self) -> None:
        consumed = []

        def sources() -> Iterator[p.Source]:
            for i in range(3):
                consumed.append(i)
                yield p.Source(str(i), f"é🌍 document {i}")

        snapshot = self.db.corpus("pages", sources(), limit=2)
        self.assertEqual(consumed, [0, 1])
        query = snapshot.query()
        page = query.preview(limit=2, max_characters=2)
        self.assertEqual([row["ordinal"] for row in page], [0, 1])
        self.assertEqual([row["text"] for row in page], ["é🌍"] * 2)
        self.assertEqual(query.preview(offset=1, limit=2, max_characters=2), page[1:])
        self.assertTrue(all(row["truncated"] for row in page))
        ids = [row["id"] for row in page]
        chosen = snapshot.query(steps=[p.where(p.document_id.is_in(ids))])
        self.assertEqual(
            chosen.id, snapshot.query(steps=[p.where(p.document_id.is_in(ids[::-1] + ids))]).id
        )
        self.assertEqual({row["id"] for row in chosen.preview(limit=len(ids))}, set(ids))
        self.assertEqual(snapshot.query(steps=[p.where(p.document_id.is_in([]))]).preview(), [])
        self.assertEqual(
            snapshot.query(steps=[p.where(p.document_id.is_in(["ff" * 32]))])
            .profile()
            .output_documents,
            0,
        )
        self.assertEqual(query.preview(offset=100), [])
        self.assertEqual(query.preview(limit=0), [])
        self.assertEqual(query.preview(max_characters=0)[0]["text"], "")
        self.assertEqual(self.db.corpus("zero", sources(), limit=0).profile().documents, 0)
        self.assertEqual(consumed, [0, 1])
        for options in (
            {"limit": -1},
            {"offset": True},
            {"limit": 1001},
            {"max_characters": 1_000_001},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                query.preview(
                    limit=options.get("limit", 10),
                    offset=options.get("offset", 0),
                    max_characters=options.get("max_characters", 1024),
                )
        with self.assertRaises(TypeError):
            p.document_id.is_in(ids[0])
        bad = p.query(
            snapshot.id, steps=[q.Operation(document_ids=q.DocumentSelection(ids=[b"bad"]))]
        )
        with self.assertRaises(ValueError):
            self.db._submit(bad)

    def test_read_only_preview_and_selection_match_saved_results(self) -> None:
        snapshot = self.db.corpus("saved", [p.Source("a", "hello world")])
        query = snapshot.query()
        ids = query.dataset(sequence_length=32)[0].document_ids()
        selected = snapshot.query(steps=[p.where(p.document_id.is_in(ids))]).wait()
        with p.PremixDB(storage=self.root, read_only=True) as reader:
            self.assertEqual(reader.corpus("saved").id, snapshot.id)
            self.assertEqual(reader._query(query.id).preview(), query.preview())
            self.assertEqual(reader._query(selected.id).preview(), selected.preview())
            with self.assertRaises(PermissionError):
                reader.corpus("write", [p.Source("x", "no")])


if __name__ == "__main__":
    unittest.main()


def test_actual_readme_python_blocks_execute_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    blocks = re.findall(r"```python\n(.*?)\n```", readme, re.DOTALL)
    assert blocks
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PREMIXDB_STORAGE", str(tmp_path / ".premixdb"))
    with p.PremixDB() as shell_db:
        shell_db.corpus("demo", SHELL_SOURCES)
    api = Mock()
    revisions = {
        "datablations/c4-filter-small": "f975fa88ccfea268f412be33ed62cd3644d9d140",
        "datablations/oscar-filter-small": "f4d35f7523c2156660fa2a06b0a1b35cb4b9308e",
    }

    def dataset_info(repository: str, *, revision: str) -> SimpleNamespace:
        assert revision == "main"
        return SimpleNamespace(sha=revisions[repository])

    api.dataset_info.side_effect = dataset_info
    consumed = []

    def hub_rows(*args: str | None, **kwargs: str | int | bool) -> Iterator[dict[str, JSON]]:
        repository = args[0]
        captured = []
        consumed.append(captured)
        for index in range(1000):
            captured.append(index)
            text = (
                "Science explains how stars form and planets move. "
                if index % 2 == 0
                else "Learning works best when students discuss ideas and practise new skills. "
            )
            if repository == "datablations/oscar-filter-small":
                yield {"id": index, "text": "OSCAR archive: " + text * 5, "meta": {}}
            else:
                # Include extra columns and null URLs, as in c4-filter-small's schema.
                yield {"text": text * 5, "url": None, "perplexity": 300.0}

    real_producer = enrichment.producer
    with (
        p.PremixDB() as bootstrap,
        patch("huggingface_hub.HfApi", return_value=api),
        patch("datasets.load_dataset", side_effect=hub_rows) as load,
        patch.object(
            enrichment,
            "producer",
            side_effect=lambda policy: (
                real_producer(policy) if policy.HasField("datatrove") else TutorialModels(policy)
            ),
        ),
    ):
        # The README continues inside the shell, which supplies these bindings.
        namespace: dict[str, object] = {"p": p, "db": bootstrap}
        shell_db = None
        try:
            for index, code in enumerate(blocks):
                exec(compile(ast.parse(code), f"README.md:example-{index + 1}", "exec"), namespace)
                if shell_db is None:
                    handle = namespace.get("db")
                    assert isinstance(handle, p.PremixDB)
                    shell_db = handle
        finally:
            if shell_db is not None:
                shell_db.close()
    assert [len(rows) for rows in consumed] == [8, 8]
    assert api.dataset_info.call_count == load.call_count == 2
    mixtures = namespace["mixtures"]
    assert isinstance(mixtures, p.Mix) and len(mixtures) == 3
    batch = namespace["batch"]
    assert isinstance(batch, dict)
    from torch import Tensor

    tokens: object = batch["input_ids"]
    assert isinstance(tokens, Tensor) and tuple(tokens.shape) == (1, 64)
    expected = list(revisions.items())
    for call, (repository, revision) in zip(load.call_args_list, expected, strict=True):
        assert call.args == (repository, None)
        assert call.kwargs == {
            "split": "train",
            "revision": revision,
            "streaming": True,
        }
