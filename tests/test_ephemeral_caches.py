"""Temporary caches release files immediately, even while failures remain inspectable."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest
from _type_support import coordinator
from blake3 import blake3
from test_enrichment_service import ControlledFields
from tokenizers import Tokenizer, models

import premixdb as p
from premixdb.engine import curation
from premixdb.engine.dataset_plan import PackingPlan
from premixdb.engine.datasets import HuggingFaceTokenizer, TokenList
from premixdb.engine.identity import CodeVersion
from premixdb.engine.mixtures import MixturePool
from premixdb.engine.plans import Step
from premixdb.engine.queries import CorpusIndex, Query, Row
from premixdb.engine.snapshots import Snapshot
from premixdb.engine.spill import cosine_edges
from premixdb.engine.token_cache import TokenCache, token_pool
from premixdb.engine.value_cache import ValueCache
from premixdb.runtime import enrichment
from premixdb.v1 import query_pb2 as q


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, sqlite3.Connection]]:
    connections: list[tuple[Path, sqlite3.Connection]] = []
    connect = sqlite3.connect

    def tracked(
        path: str | Path,
        *,
        check_same_thread: bool = True,
        uri: bool = False,
        timeout: float = 5,
    ) -> sqlite3.Connection:
        database = connect(path, check_same_thread=check_same_thread, uri=uri, timeout=timeout)
        if Path(path).name == "evidence.sqlite3":
            connections.append((Path(path), database))
        return database

    monkeypatch.setattr(sqlite3, "connect", tracked)
    return connections


def assert_closed(opened: list[tuple[Path, sqlite3.Connection]]) -> None:
    assert len(opened) == 1
    path, database = opened[0]
    assert not path.parent.exists()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        database.execute("SELECT 1")


@pytest.fixture
def model_query() -> tuple[Query, HuggingFaceTokenizer]:
    asset = (Path(__file__).parent / "fixtures" / "wordpiece.json").read_bytes()
    tokenizer = HuggingFaceTokenizer.from_bytes(asset, blake3(asset).hexdigest(), 1024)
    code = CodeVersion("local://cleanup", "a" * 40, "09" * 32)
    snapshot = Snapshot("01" * 16, [("a", "hello world"), ("b", "playing")], code)
    return CorpusIndex([snapshot]).execute([], code), tokenizer


@pytest.mark.parametrize("stream", [False, True])
def test_completed_datasets_release_owned_pools(
    model_query: tuple[Query, HuggingFaceTokenizer],
    opened: list[tuple[Path, sqlite3.Connection]],
    stream: bool,
) -> None:
    query, tokenizer = model_query
    dataset = query.dataset(2, None, None, tokenizer, stream=stream)
    assert len(list(dataset.iter_sequences())) == 2
    assert_closed(opened)
    dataset.close()
    dataset.close()
    assert dataset.summary()["content_tokens"] == 4


def test_explicit_close_releases_a_partially_read_dataset_pool(
    model_query: tuple[Query, HuggingFaceTokenizer], opened: list[tuple[Path, sqlite3.Connection]]
) -> None:
    query, tokenizer = model_query
    dataset = query.dataset(2, None, None, tokenizer, stream=True)
    sequences = dataset.iter_sequences()
    assert next(sequences).ordinal == 0
    assert opened[0][0].parent.exists()
    dataset.close()
    assert_closed(opened)


@pytest.mark.parametrize("stage", ["construct", "pack", "mixture"])
def test_failures_release_pools_while_their_tracebacks_remain_alive(
    model_query: tuple[Query, HuggingFaceTokenizer],
    opened: list[tuple[Path, sqlite3.Connection]],
    stage: str,
) -> None:
    query, tokenizer = model_query
    target = {
        "construct": "premixdb.engine.datasets.Dataset.__init__",
        "pack": "premixdb.engine.token_cache.TokenCache.__getitem__",
        "mixture": "premixdb.engine.token_cache.TokenCache.length",
    }[stage]
    with patch(target, side_effect=RuntimeError("owned pool failed")):
        with pytest.raises(RuntimeError, match="owned pool failed") as failure:
            if stage == "mixture":
                MixturePool(query, "object.uri", {}, tokenizer)
            else:
                query.dataset(2, None, None, tokenizer)
    assert failure.value.__traceback__ is not None
    assert_closed(opened)


def test_mixture_datasets_keep_their_borrowed_pool_open(
    model_query: tuple[Query, HuggingFaceTokenizer], opened: list[tuple[Path, sqlite3.Connection]]
) -> None:
    query, tokenizer = model_query
    pool = MixturePool(query, "object.uri", {}, tokenizer)
    first = pool.dataset(pool.inventory(), "04" * 32, 0, False, None, 2, None, None)
    first.close()
    assert pool._encoded is not None
    assert pool._encoded.length(query.row(0).id) == 2
    second = pool.dataset(pool.inventory(), "04" * 32, 0, False, None, 2, None, None)
    assert first.id == second.id
    assert [sequence.tokens for sequence in first.iter_sequences()] == [
        sequence.tokens for sequence in second.iter_sequences()
    ]
    pool.close()
    pool.close()
    assert_closed(opened)


@pytest.mark.parametrize("fail", [False, True])
def test_profile_only_pools_close_before_returning_or_raising(
    tmp_path: Path, opened: list[tuple[Path, sqlite3.Connection]], fail: bool
) -> None:
    asset = Path(__file__).parent / "fixtures" / "wordpiece.json"
    with p.PremixDB(storage=tmp_path) as db:
        dataset = (
            db.Corpus("profile", [p.Source("a", "hello world")])
            .query()
            .mix(
                tokenizer=p.hugging_face_tokenizer(
                    asset, digest=blake3(asset.read_bytes()).digest()
                ),
                sequence_length=2,
            )[0]
        )
        if fail:
            with patch.object(PackingPlan, "profile", side_effect=RuntimeError("profile failed")):
                with pytest.raises(RuntimeError, match="profile failed") as failure:
                    dataset.profile()
            assert failure.value.__traceback__ is not None
        else:
            assert dataset.profile().content_tokens == 2
        assert_closed(opened)


def test_failed_publication_releases_the_native_dataset_pool(
    tmp_path: Path, opened: list[tuple[Path, sqlite3.Connection]]
) -> None:
    asset = Path(__file__).parent / "fixtures" / "wordpiece.json"
    with p.PremixDB(storage=tmp_path) as db:
        dataset = (
            db.Corpus("publication", [p.Source("a", "hello world")])
            .query()
            .mix(
                tokenizer=p.hugging_face_tokenizer(
                    asset, digest=blake3(asset.read_bytes()).digest()
                ),
                sequence_length=2,
            )[0]
        )
        dataset.profile()
        assert_closed(opened)
        opened.clear()
        with patch(
            "premixdb.runtime.datasets.tokens.publish", side_effect=OSError("publish failed")
        ):
            with pytest.raises(OSError, match="publish failed") as failure:
                dataset.wait()
        assert failure.value.__traceback__ is not None
        assert_closed(opened)


@pytest.mark.parametrize("cache_bytes", [0, 128 * 1024 * 1024])
def test_session_close_releases_cached_and_evicted_mixture_pools(
    tmp_path: Path, opened: list[tuple[Path, sqlite3.Connection]], cache_bytes: int
) -> None:
    asset = Path(__file__).parent / "fixtures" / "wordpiece.json"
    with p.PremixDB(storage=tmp_path, cache_bytes=cache_bytes) as db:
        mix = (
            db.Corpus("mixture", [p.Source("a", "hello world")])
            .query()
            .mix(
                tokenizer=p.hugging_face_tokenizer(
                    asset, digest=blake3(asset.read_bytes()).digest()
                ),
                tokens=2,
                sequence_length=2,
                n_candidates=1,
            )
        )
        recipe = mix._configs[0]
        pool = coordinator(db)._mix_pool(recipe, recipe.sampling.domains)
        assert pool._encoded is not None
        assert pool._encoded.length(next(iter(pool.labels))) == 2
        path, database = opened[-1]
        assert path.parent.exists()
        db.close()
        assert not path.parent.exists()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            database.execute("SELECT 1")
        pool.close()
    for connection in opened:
        assert_closed([connection])


@pytest.mark.parametrize("factory", [TokenCache, ValueCache])
def test_close_is_idempotent(
    factory: type[TokenCache] | type[ValueCache], opened: list[tuple[Path, sqlite3.Connection]]
) -> None:
    cache = factory()
    cache.close()
    cache.close()
    assert_closed(opened)


@pytest.mark.parametrize("factory", [TokenCache, ValueCache])
def test_schema_failure_releases_resources_before_propagating(
    factory: type[TokenCache] | type[ValueCache],
    opened: list[tuple[Path, sqlite3.Connection]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connect = sqlite3.connect

    def authorize(
        action: int,
        _first: str | None,
        _second: str | None,
        _database: str | None,
        _trigger: str | None,
    ) -> int:
        return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_CREATE_TABLE else sqlite3.SQLITE_OK

    def deny_schema(path: Path, *, check_same_thread: bool = True) -> sqlite3.Connection:
        database = connect(path, check_same_thread=check_same_thread)
        database.set_authorizer(authorize)
        return database

    monkeypatch.setattr(sqlite3, "connect", deny_schema)
    with pytest.raises(sqlite3.DatabaseError, match="not authorized") as failure:
        factory()
    assert failure.value.__traceback__ is not None
    assert_closed(opened)


def test_invalid_field_rows_release_the_partially_loaded_cache(
    opened: list[tuple[Path, sqlite3.Connection]],
) -> None:
    with pytest.raises(ValueError) as failure:
        ValueCache([("valid", 1), ("invalid", float("nan"))])
    assert failure.value.__traceback__ is not None
    assert_closed(opened)


def test_close_releases_storage_with_an_unfinished_items_iterator(
    opened: list[tuple[Path, sqlite3.Connection]],
) -> None:
    cache = ValueCache((f"{index:03}", index) for index in range(65))
    rows = iter(cache.items())
    assert next(rows) == ("000", 0)
    cache.close()
    assert_closed(opened)
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        list(rows)


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_failed_encoding_releases_a_partially_filled_pool(
    tmp_path: Path,
    opened: list[tuple[Path, sqlite3.Connection]],
    error: type[RuntimeError] | type[KeyboardInterrupt],
) -> None:
    asset = tmp_path / "tokenizer.json"
    Tokenizer(models.WordLevel({"[UNK]": 0, "text": 1}, unk_token="[UNK]")).save(str(asset))
    tokenizer = HuggingFaceTokenizer(asset, blake3(asset.read_bytes()).hexdigest(), 1024)
    code = CodeVersion("local://test", "a" * 40, "09" * 32)
    snapshot = Snapshot("01" * 16, [("a", "one"), ("b", "two")], code)
    query = CorpusIndex([snapshot]).execute([], code)
    calls = 0

    def encode(row: Row, _tokenizer: HuggingFaceTokenizer) -> TokenList:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise error("encoder failed")
        return TokenList([1], [[(0, row.document.size)]])

    query._encoding_provider = encode
    with pytest.raises(error, match="encoder failed") as failure:
        token_pool(query, tokenizer)
    assert calls == 2
    assert failure.value.__traceback__ is not None
    assert_closed(opened)


@pytest.mark.parametrize("stage", ["success", "prepare", "estimate", "execute", "publish"])
def test_query_columns_close_while_indexes_and_failures_remain_alive(
    tmp_path: Path, opened: list[tuple[Path, sqlite3.Connection]], stage: str
) -> None:
    indexes: list[CorpusIndex] = []
    execute = CorpusIndex.execute

    def tracked(
        index: CorpusIndex, plan: Iterable[Step], version: CodeVersion, fields: tuple[bytes, ...]
    ) -> Query:
        indexes.append(index)
        if stage == "execute":
            raise RuntimeError("columns failed")
        return execute(index, plan, version, fields)

    intrinsic = curation.intrinsic_value
    calls = 0

    def measured(document: curation.SelectedDocument, field: q.IntrinsicField) -> int | str:
        nonlocal calls
        calls += 1
        if stage == "prepare" and calls == 3:
            raise RuntimeError("columns failed")
        return intrinsic(document, field)

    with p.PremixDB(storage=tmp_path) as db:
        query = db.Corpus("columns", [p.Source("a", "one"), p.Source("b", "two")]).query(
            sampling=p.sample(seed=3, documents=1, domains=(p.text.characters, p.object.uri))
        )
        with (
            patch.object(CorpusIndex, "execute", autospec=True, side_effect=tracked),
            patch.object(curation, "intrinsic_value", side_effect=measured),
            ExitStack() as failures,
        ):
            target = {
                "estimate": "premixdb.storage.profiles.estimate_query",
                "publish": "premixdb.storage.preview.inline",
            }.get(stage)
            if target is not None:
                failures.enter_context(patch(target, side_effect=RuntimeError("columns failed")))
            if stage == "success":
                query.wait()
                assert query.profile().output_documents == 1
            else:
                with pytest.raises(RuntimeError, match="columns failed") as failure:
                    query.wait()
                assert failure.value.__traceback__ is not None
        assert len(opened) == 2
        for connection in opened:
            assert_closed([connection])
        if stage in ("success", "execute", "publish"):
            assert len(indexes) == 1
            assert len(indexes[0].field_values) == 2


def test_failed_projection_closes_the_partially_loaded_column(
    tmp_path: Path, opened: list[tuple[Path, sqlite3.Connection]]
) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.Corpus("projection", [p.Source("a", "one"), p.Source("b", "two")])
        with patch.object(enrichment, "producer", ControlledFields):
            snapshot.query()._with_fields([p.quality.educational_value]).wait()
        assert_closed(opened)
        opened.clear()
        query = snapshot.query(
            sampling=p.sample(seed=3, documents=1, domains=p.quality.educational_value)
        )
        with patch.object(
            enrichment, "project", side_effect=[0.2, RuntimeError("projection failed")]
        ):
            with pytest.raises(RuntimeError, match="projection failed") as failure:
                query.wait()
        assert failure.value.__traceback__ is not None
        assert_closed(opened)


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_failed_cosine_stream_releases_partially_written_evidence(
    opened: list[tuple[Path, sqlite3.Connection]],
    error: type[RuntimeError] | type[KeyboardInterrupt],
) -> None:
    def vectors() -> Iterable[tuple[str, list[float]]]:
        yield "a", [1.0, 0.0]
        raise error("vector stream failed")

    with pytest.raises(error, match="vector stream failed") as failure:
        list(cosine_edges(vectors(), 1.0))
    assert failure.value.__traceback__ is not None
    assert_closed(opened)
