"""Cold reads preserve published selections and verified token alignment."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from _type_support import coordinator

import premixdb as p
from premixdb import _runtime
from premixdb._ids import _decode_id
from premixdb.engine.datasets import HuggingFaceTokenizer
from premixdb.internal import derivation_pb2 as d


@pytest.mark.parametrize("mode", ["filtered", "span", "replacement", "empty"])
def test_completed_selection_reopens_without_running_kernels(tmp_path: Path, mode: str) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.corpus(
            "selection",
            [p.Source(str(i), f"é first {i}\nremove me\n🌍 tail") for i in range(3)],
        )
        if mode == "span":
            reference = db.corpus("reference", [p.Source("ref", "remove me")])
            query = snapshot.query(
                decontaminate=p.decontaminate(reference, algorithm="line", granularity="span")
            )
        elif mode == "replacement":
            query = snapshot.query(sampling=p.sample(seed=42, documents=5, replacement=True))
        else:
            query = snapshot.query(
                steps=[p.where(p.text.characters > (10_000 if mode == "empty" else 0))]
            )
        expected = query.preview(offset=1, max_characters=2048)
        lineage = query._provenance()
        identity = query.id
        dataset = query.dataset(tokenizer=p.ByteTokenizer(), sequence_length=16).wait()
        packed = [(s.tokens, s.mask, s.spans) for s in dataset]
    with p.PremixDB(storage=tmp_path, cache_bytes=0) as db:
        with (
            patch.object(
                coordinator(db), "_execute_query", side_effect=AssertionError("reexecuted")
            ),
            patch.object(
                coordinator(db), "_snapshot", side_effect=AssertionError("loaded full snapshot")
            ),
        ):
            query = db._query(identity)
            assert query.preview(offset=1, max_characters=2048) == expected
            assert query._provenance() == lineage
            # A new packing length forces the restored handle through the packer.
            rebuilt = query.dataset(tokenizer=p.ByteTokenizer(), sequence_length=8).wait()
            assert rebuilt.profile().content_tokens == dataset.profile().content_tokens
            assert [(s.tokens, s.mask, s.spans) for s in db._dataset(dataset.id)] == packed


def test_missing_selection_shard_is_an_error_not_a_reexecution(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        query = db.corpus("missing", [p.Source("a", "hello")]).query().wait()
        id = _decode_id(query.id)
        manifest = coordinator(db)._storage.load("query", id, d.QuerySelection, suffix=".selection")
        ref = manifest.shards[0]
        (tmp_path / "query/objects" / ref.blake3_digest.hex()).unlink()
    with p.PremixDB(storage=tmp_path, cache_bytes=0) as db:
        with patch.object(
            coordinator(db), "_execute_query", side_effect=AssertionError("reexecuted")
        ):
            with pytest.raises(ValueError, match="selection shard is missing"):
                db._query(id).preview(max_characters=2048)


def test_profiles_mixes_packing_and_restart_share_token_encodings(tmp_path: Path) -> None:
    from blake3 import blake3

    asset = Path(__file__).parent / "fixtures/wordpiece.json"
    tokenizer = p.hugging_face_tokenizer(asset, digest=blake3(asset.read_bytes()).hexdigest())
    original = HuggingFaceTokenizer.encode_with_offsets
    calls = []

    def encode(self: HuggingFaceTokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        calls.append(text)
        return original(self, text)

    with patch.object(HuggingFaceTokenizer, "encode_with_offsets", encode):
        with p.PremixDB(storage=tmp_path) as db:
            query = db.corpus(
                "tokens", [p.Source("a", "hello world"), p.Source("b", "hello")]
            ).query()
            first = query.dataset(tokenizer=tokenizer, sequence_length=4).wait()
            expected = [s.tokens for s in first]
            assert len(calls) == 2
            query.dataset(tokenizer=tokenizer, sequence_length=8).wait()
            query.mix(tokenizer=tokenizer, tokens=4, sequence_length=4)[0].wait()
            assert len(calls) == 2
            identity = query.id
        with p.PremixDB(storage=tmp_path) as db:
            query = db._query(identity)
            query.dataset(tokenizer=tokenizer, sequence_length=2).wait()
            assert len(calls) == 2
            assert [s.tokens for s in db._dataset(first.id)] == expected


def test_runtime_changes_create_new_recipes_but_keep_completed_inputs(tmp_path: Path) -> None:
    from blake3 import blake3

    asset = Path(__file__).parent / "fixtures/wordpiece.json"
    tokenizer = p.hugging_face_tokenizer(asset, digest=blake3(asset.read_bytes()).hexdigest())
    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.corpus("runtime", [p.Source("a", "hello world")])
        query = snapshot.query()
        first = query.dataset(tokenizer=tokenizer, sequence_length=4).wait()
        identity = query.id
    changed = replace(_runtime.current_code(), environment="ab" * 32)
    with patch.object(_runtime, "current_code", return_value=changed):
        with p.PremixDB(storage=tmp_path) as db:
            with patch.object(
                coordinator(db), "_execute_query", side_effect=AssertionError("reexecuted")
            ):
                old = db._query(identity)
                assert old.preview(max_characters=2048)[0]["text"] == "hello world"
                packed = old.dataset(tokenizer=tokenizer, sequence_length=4).wait()
                assert packed.id != first.id
                assert [s.tokens for s in packed] == [s.tokens for s in db._dataset(first.id)]
            assert db._snapshot(snapshot.id).query().id != identity


def test_durable_token_corruption_is_not_a_cache_miss(tmp_path: Path) -> None:
    from blake3 import blake3

    asset = Path(__file__).parent / "fixtures/wordpiece.json"
    tokenizer = p.hugging_face_tokenizer(asset, digest=blake3(asset.read_bytes()).hexdigest())
    with p.PremixDB(storage=tmp_path) as db:
        query = db.corpus("corruption", [p.Source("a", "hello")]).query()
        query.dataset(tokenizer=tokenizer, sequence_length=4).wait()
        manifest = coordinator(db)._storage.list("tokenizer", d.TokenEncoding, suffix=".encoding")[
            0
        ]
        (tmp_path / "tokenizer/objects" / manifest.shards[0].blake3_digest.hex()).write_bytes(
            b"bad"
        )
        with patch.object(HuggingFaceTokenizer, "encode_with_offsets", side_effect=AssertionError):
            with pytest.raises(ValueError, match="integrity"):
                query.dataset(tokenizer=tokenizer, sequence_length=8).profile()


def test_local_resources_can_be_opened_read_only(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.corpus("published", [p.Source("a", "hello world")])
        query = snapshot.query()
        dataset = query.dataset(tokenizer=p.ByteTokenizer(), sequence_length=4).wait()
        expected = dataset.profile()
    with p.PremixDB(storage=tmp_path, read_only=True) as db:
        assert db.corpus("published").id == snapshot.id
        assert db._query(query.id).preview(max_characters=2048)[0]["text"] == "hello world"
        assert db._query(query.id)._provenance()
        restored = db._dataset(dataset.id)
        assert restored.profile() == expected
        assert len(restored[0].tokens) == 4


def test_public_recipes_reuse_cached_results_after_restarting(tmp_path: Path) -> None:
    from blake3 import blake3

    asset = Path(__file__).parent / "fixtures/wordpiece.json"
    tokenizer = p.hugging_face_tokenizer(asset, digest=blake3(asset.read_bytes()).hexdigest())
    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.corpus("automatic-cache", [p.Source("a", "hello world")])
        query = snapshot.query(steps=[p.where(p.text.characters > 0)])
        dataset = query.dataset(tokenizer=tokenizer, sequence_length=8).wait()
        expected = [(sequence.tokens, sequence.mask) for sequence in dataset]
        expected_profile = dataset.profile()
    with p.PremixDB(storage=tmp_path) as db:
        with (
            patch.object(
                coordinator(db), "_execute_query", side_effect=AssertionError("reexecuted query")
            ),
            patch.object(
                coordinator(db), "_tokenizer", side_effect=AssertionError("loaded tokenizer")
            ),
            patch.object(
                coordinator(db), "_dataset_handle", side_effect=AssertionError("repacked dataset")
            ),
            patch.object(
                coordinator(db),
                "_profile_dataset",
                side_effect=AssertionError("recomputed profile"),
            ),
        ):
            for _ in range(2):
                dataset = (
                    db.corpus("automatic-cache")
                    .query(steps=[p.where(p.text.characters > 0)])
                    .dataset(tokenizer=tokenizer, sequence_length=8)
                )
                assert dataset.status is p.ExecutionStatus.COMPLETED
                assert dataset.profile() == expected_profile
                assert [(sequence.tokens, sequence.mask) for sequence in dataset] == expected
