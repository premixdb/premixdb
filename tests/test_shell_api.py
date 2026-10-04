"""Shell-facing APIs expose explicit contracts and bounded, readable results."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from unittest.mock import patch

import pytest
from _type_support import SHELL_SOURCES, coordinator
from blake3 import blake3

import premixdb as p
from premixdb._catalog import CorpusCollection
from premixdb._cli import _demo_sources, _shell_banner
from premixdb._field_expr import ScalarField, VectorField
from premixdb._ids import _decode_id
from premixdb._reader import Reader
from premixdb._resources import Sequence
from premixdb._unions import SnapshotUnion
from premixdb.enrichment import (
    DataTroveFields,
    DupekitIndex,
    Embeddings,
    LanguageScores,
    QuRating,
    WebOrganizer,
)
from premixdb.enrichment.language import _model_path


@pytest.mark.parametrize(
    "cls, names",
    [
        (p.PremixDB, "version close corpus"),
        (
            p.Corpus,
            "id name latest snapshot list_document list_snapshot list_query list_mixture list_dataset",
        ),
        (
            p.Snapshot,
            "id corpus_id status preview profile query union wait",
        ),
        (
            p.Query,
            "id status dataset preview profile mix wait",
        ),
        (p.Mix, "id weights profile"),
        (p.Dataset, "id status wait profile preview torch reader"),
        (Reader, "checkpoint"),
        (Sequence, "ordinal tokens mask attention_mask spans document_ids"),
        (CorpusCollection, "list"),
        (SnapshotUnion, "union query"),
    ],
)
def test_resource_public_surface_contains_only_workflow_operations(cls: type, names: str) -> None:
    public = {name for name, _ in inspect.getmembers_static(cls) if not name.startswith("_")}
    assert public == set(names.split())


def test_supported_methods_have_docstrings_and_explicit_parameters() -> None:
    for cls in (
        p.PremixDB,
        p.Corpus,
        p.Snapshot,
        p.Query,
        p.Dataset,
        p.Mix,
        p.Source,
        p.Topology,
        p.RangeReader,
        p.DistributionSummary,
        Reader,
        Sequence,
        CorpusCollection,
        SnapshotUnion,
        ScalarField,
        VectorField,
        DataTroveFields,
        DupekitIndex,
        Embeddings,
        LanguageScores,
        QuRating,
        WebOrganizer,
    ):
        for name, member in inspect.getmembers_static(cls):
            if name.startswith("_"):
                continue
            fn = member.fget if isinstance(member, property) else getattr(member, "func", member)
            if not callable(fn) or not getattr(fn, "__module__", "").startswith("premixdb"):
                continue
            assert inspect.getdoc(fn), f"{cls.__name__}.{name} has no docstring"
            assert all(
                parameter.kind != inspect.Parameter.VAR_KEYWORD
                for parameter in inspect.signature(fn).parameters.values()
            ), f"{cls.__name__}.{name} hides its parameters"


def test_preview_pages_and_snapshot_surface(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        old = db.corpus("pages", [p.Source(str(i), "sample text") for i in range(2)])
        query = old.query().wait()
        from premixdb._shell import _PublicCompleter

        completer = _PublicCompleter(namespace={"query": query})
        names = {
            match.removeprefix("query.").split("(", 1)[0]
            for match in completer.attr_matches("query.")
        }
        assert names == {"id", "status", "dataset", "preview", "profile", "mix", "wait"}
        assert completer.attr_matches("query._") == []
        for removed in ("estimate", "describe", "provenance", "list_document"):
            assert not hasattr(query, removed)
        assert [row["ordinal"] for row in old.preview()] == [0, 1]
        assert [row["ordinal"] for row in old.preview(limit=1)] == [0]
        assert [row["ordinal"] for row in old.preview(offset=1)] == [1]
        assert [row["ordinal"] for row in query.preview(offset=1)] == [1]
        assert old.preview(offset=2) == query.preview(offset=2) == []
        newer = db.corpus("pages", [p.Source("new", "a different snapshot")])
        assert db.corpus("pages").id == newer.id
        assert old.profile().documents == 2
        assert newer.preview()[0]["source_key"] == "new"
        assert newer.profile().added == 1
        for removed in (
            "changes",
            "latest",
            "describe",
            "list_document",
            "list_snapshot",
            "list_query",
            "list_mixture",
            "list_dataset",
        ):
            assert not hasattr(old, removed)


def test_execution_history_has_readable_identifiers_and_second_timestamps(tmp_path: Path) -> None:
    db = p.PremixDB(storage=tmp_path)
    try:
        snapshot = db.corpus("history", [p.Source("a", "hello")])
        events = db._executions(snapshot.id)
        assert events
        for event in events:
            assert event.resource_id == snapshot.id
            assert len(_decode_id(event.request_digest)) == 32
            assert event.started_at and event.ended_at
            assert len(event.started_at) == len(event.ended_at) == 20
            assert event.started_at.endswith("Z") and "." not in event.started_at
    finally:
        db.close()


def test_demo_is_the_complete_bundled_dataset() -> None:
    sources = _demo_sources()
    text = "".join(source.text for source in sources).encode()
    assert len(sources) == 7222
    assert len(text) == 1115394
    assert (
        blake3(text).hexdigest()
        == "5bd8f6749d3cda816828aabf1aebdc595f673de8102e521bbd8369ad4b7917e9"
    )


def test_nine_speech_demo_upgrades_and_keeps_its_old_snapshot(tmp_path: Path) -> None:
    excerpt = (Path(p.__file__).parent / "data/tiny_shakespeare_excerpt.txt").read_text()
    db = p.PremixDB(storage=tmp_path)
    try:
        old = db.corpus(
            "demo",
            [
                p.Source(f"speech/{i:04d}", block + "\n\n")
                for i, block in enumerate(excerpt.strip().split("\n\n"))
            ],
        )
        with patch("premixdb._cli._demo_sources", return_value=SHELL_SOURCES) as demo_sources:
            _shell_banner(db)
            current = db.corpus("demo")
            assert current.profile().documents == len(SHELL_SOURCES)
            assert old.profile().documents == 9
            assert db._snapshot(old.id).profile().documents == 9
            _shell_banner(db)
            assert db.corpus("demo").id == current.id
        demo_sources.assert_called_once_with()
    finally:
        db.close()


def test_fasttext_download_is_quiet_and_reuses_the_existing_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    def download(url: str, filename: str, *, progress: bool) -> None:
        assert progress is False
        Path(filename).write_bytes(b"model")

    with (
        patch("datatrove.io.cached_assets_path", return_value=str(tmp_path)),
        patch("datatrove.io.download_file", side_effect=download) as transfer,
        caplog.at_level(logging.DEBUG),
    ):
        path = _model_path("https://example.org/lid.bin", "ft176")
        assert _model_path("https://example.org/lid.bin", "ft176") == path
        assert path.read_bytes() == b"model"
        transfer.assert_called_once()
    captured = capsys.readouterr()
    assert not captured.out and not captured.err
    assert not caplog.records


def test_fasttext_download_failure_is_logged_and_can_be_retried(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with (
        patch("datatrove.io.cached_assets_path", return_value=str(tmp_path)),
        patch("datatrove.io.download_file", side_effect=OSError("download failed")),
        pytest.raises(OSError, match="download failed"),
    ):
        _model_path("https://example.org/lid.bin", "ft176")
    assert "fastText model download failed" in caplog.text
    assert not list(tmp_path.glob("*.completed"))


def test_profile_display_is_bounded_and_preserves_typed_data(tmp_path: Path) -> None:
    from premixdb.v1.query_pb2 import QueryProfile, QueryStepProfile

    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.corpus("summary", [p.Source("a", "hello"), p.Source("b", "world")])
        profile = snapshot.profile()
        assert profile.documents == profile.added == 2
        original = profile.SerializeToString()
        for display in (str(profile), repr(profile)):
            assert len(display.splitlines()) <= 15
            assert "Documents: 2" in display and "2 added" in display
        assert profile.SerializeToString() == original
        query = snapshot.query(steps=[p.where(p.text.characters > 0)])
        assert len(str(query.profile()).splitlines()) <= 15
        assert "Step 1: 2 → 2 documents" in str(query.profile())
    crowded = QueryProfile(
        input_documents=100,
        output_documents=10,
        steps=[QueryStepProfile(input_documents=100, output_documents=10)] * 100,
    )
    crowded.sampling.requested = 10
    crowded.decontamination.removed_documents = 90
    assert len(str(crowded).splitlines()) <= 15


def test_query_planning_and_reopening_do_not_execute(tmp_path: Path) -> None:
    from premixdb.execution import Coordinator

    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.corpus("lazy", [p.Source("a", "hello"), p.Source("b", "world")])
        assert isinstance(coordinator(db), Coordinator)
        with (
            patch.object(
                coordinator(db), "_execute_query", side_effect=AssertionError("execution")
            ),
            patch.object(
                coordinator(db), "_schedule_query", side_effect=AssertionError("scheduling")
            ),
            patch.object(
                coordinator(db), "_snapshot", side_effect=AssertionError("document reads")
            ),
        ):
            query = snapshot.query(steps=[p.where(p.text.characters > 0)])
            assert query.status is p.ExecutionStatus.PENDING
            assert query.id == snapshot.query(steps=[p.where(p.text.characters > 0)]).id
            assert db._query(query.id).status is p.ExecutionStatus.PENDING
            assert "Query" in repr(query)
            assert query._estimate.output.upper == 2
        identity = query.id
    with p.PremixDB(storage=tmp_path) as reopened:
        restored = reopened._query(identity)
        assert restored.status is p.ExecutionStatus.PENDING
        assert sorted(row["text"] for row in restored.preview()) == ["hello", "world"]
        assert restored.status is p.ExecutionStatus.COMPLETED


def test_default_policies_preserve_data_and_are_reproducible(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.corpus("defaults", [p.Source("a", "hello"), p.Source("b", "world")])
        query = snapshot.query()
        assert not query._proto.HasField("sampling")
        assert not query._proto.HasField("decontaminate")
        assert query.profile().output_documents == 2
        assert p.sample() == p.sample(seed=0, fraction=1)
        reference = db.corpus("reference", [p.Source("ref", "hello")])
        policy = p.decontaminate(reference)
        assert policy.n == 13
        assert policy.algorithm == policy.ALGORITHM_EXACT_NGRAM
        assert policy.granularity == policy.DOCUMENT
        dataset = query.dataset(sequence_length=16)
        assert len(dataset) == 1
        ids = {"hello": 31373, "world": 6894}
        expected = [token for row in query.preview() for token in [ids[row["text"]], 50256]]
        assert dataset[0].tokens[:4] == expected
        assert dataset[0].tokens[4:] == [50256] * 12
        assert dataset[0].mask == [True] * 4 + [False] * 12


def test_mix_profile_display_is_bounded_without_losing_candidates() -> None:
    from premixdb._profiles import _MixProfiles
    from premixdb.v1.dataset_pb2 import DatasetProfile

    profiles = _MixProfiles(
        DatasetProfile(planned_content_tokens=12, content_tokens=12, source_documents=index)
        for index in range(11)
    )
    assert isinstance(profiles, list) and len(profiles) == 11
    original = [profile.SerializeToString() for profile in profiles]
    for display in (str(profiles), repr(profiles)):
        assert len(display.splitlines()) <= 15
        assert "Candidates: 11" in display
        assert "+1 more candidates" in display
        assert all(
            name in display
            for name in ("Content / budget", "Sequences", "Unique docs", "Repeats", "Padding")
        )
    assert len(str(profiles[10]).splitlines()) <= 15
    assert [profile.SerializeToString() for profile in profiles] == original
    empty = _MixProfiles()
    assert len(str(empty).splitlines()) <= 15
    crowded = DatasetProfile(document_occurrences=20, source_documents=2, padding_tokens=10)
    crowded.source_tokens.update({str(i): i for i in range(11)})
    assert "Repeats: 18" in str(crowded)
    assert len(repr(crowded).splitlines()) <= 15
    assert len(crowded.source_tokens) == 11


def test_dataset_planning_is_lazy_and_survives_reopening(tmp_path: Path) -> None:
    from premixdb.execution import Coordinator
    from premixdb.v1.dataset_pb2 import Dataset

    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.corpus("lazy-dataset", [p.Source("a", "hello world")])
        query = snapshot.query()
        assert isinstance(coordinator(db), Coordinator)
        with (
            patch.object(
                coordinator(db), "_execute_query", side_effect=AssertionError("selection")
            ),
            patch.object(
                coordinator(db), "_snapshot", side_effect=AssertionError("document reads")
            ),
            patch.object(
                coordinator(db), "_tokenizer", side_effect=AssertionError("tokenizer load")
            ),
            patch.object(
                coordinator(db), "_profile_dataset", side_effect=AssertionError("profiling")
            ),
            patch.object(coordinator(db), "_dataset_handle", side_effect=AssertionError("packing")),
            patch.object(p.Query, "wait", side_effect=AssertionError("query wait")),
        ):
            dataset = query.dataset(sequence_length=8)
            assert dataset.status is query.status is p.ExecutionStatus.PENDING
            assert not dataset._proto.HasField("profile")
            assert "GPT2Tokenizer()" in repr(dataset)
            assert query.dataset(sequence_length=8).id == dataset.id
            assert db._dataset(dataset.id).status is p.ExecutionStatus.PENDING
            assert not coordinator(db)._storage.list("dataset", Dataset)
        identity, query_id = dataset.id, query.id
    with p.PremixDB(storage=tmp_path) as db:
        dataset = db._dataset(identity)
        assert dataset.status is db._query(query_id).status is p.ExecutionStatus.PENDING
        with patch.object(
            coordinator(db), "_dataset_handle", side_effect=AssertionError("packing")
        ):
            profile = dataset.profile()
        assert profile.content_tokens == 2
        assert profile.separator_tokens == 1
        assert profile.padding_tokens == 5
        assert dataset.status is p.ExecutionStatus.PENDING
        assert not coordinator(db)._storage.list("dataset", Dataset)
        with p.PremixDB(storage=tmp_path, read_only=True) as reader:
            assert reader._dataset(identity).profile() == profile
            assert reader._dataset(identity).status is p.ExecutionStatus.PENDING
        assert dataset[0].tokens == [31373, 995] + [50256] * 6
        assert dataset[0].mask == [True] * 3 + [False] * 5
        assert dataset.profile() == profile
        assert dataset.status is p.ExecutionStatus.COMPLETED
    with p.PremixDB(storage=tmp_path, read_only=True) as db:
        assert db._dataset(identity)[0].tokens == [31373, 995] + [50256] * 6


def test_default_token_budget_uses_bpe_and_keeps_byte_policy_explicit(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.corpus("bpe-budget", [p.Source("a", "hello world")])
        policy = p.sample(tokens=2, replacement=False)
        assert policy.tokenizer_asset == p.GPT2Tokenizer().hugging_face.asset
        profile = snapshot.query(sampling=policy).profile()
        assert profile.output_documents == 1
        assert profile.sampling.realized == 2
        assert profile.sampling.overshoot == 0
        byte_profile = snapshot.query(
            sampling=p.sample(tokens=2, tokenizer=p.ByteTokenizer())
        ).profile()
        assert byte_profile.output_documents == 1
        assert byte_profile.sampling.realized == 11
        assert byte_profile.sampling.overshoot == 9


def test_database_instance_and_completion_expose_only_three_members(tmp_path: Path) -> None:
    from premixdb._shell import _PublicCompleter

    with p.PremixDB(storage=tmp_path) as db:
        assert {name for name in dir(db) if not name.startswith("_")} == {
            "version",
            "close",
            "corpus",
        }
        completer = _PublicCompleter(namespace={"db": db})
        assert set(completer.attr_matches("db.")) == {"db.version", "db.close", "db.corpus"}
        assert completer.attr_matches("db._") == []
        for removed in (
            "snapshot",
            "query",
            "dataset",
            "datasets",
            "executions",
            "storage",
            "read_only",
            "timeout",
            "poll_interval",
            "open",
        ):
            assert not hasattr(db, removed)
