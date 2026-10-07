"""Previews consume exact prefixes without publishing full selections or packing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest
from _type_support import coordinator, tokenizer_packing, wordpiece_tokenizer

import premixdb as p
from premixdb.engine.datasets import HuggingFaceTokenizer
from premixdb.engine.snapshots import StoredDocument
from premixdb.v1 import data_mixture_pb2 as m
from premixdb.v1 import query_pb2 as q


def test_snapshot_inspection_reuses_capture_statistics_and_bounded_text(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path, progress=False) as db:
        snapshot = db.Corpus("captured", [p.Source(str(i), "hello" * 1000) for i in range(8)])
        service = coordinator(db)
        with (
            patch.object(service, "_execute_query", side_effect=AssertionError("query scan")),
            patch.object(service, "_snapshot", side_effect=AssertionError("inventory scan")),
            patch.object(
                StoredDocument,
                "text",
                new_callable=PropertyMock,
                side_effect=AssertionError("whole document"),
            ),
        ):
            assert snapshot.profile().documents == 8
            assert len(snapshot.preview()) == 3
            assert all(row["text"] == "hello"[:2] for row in snapshot.preview(max_characters=2))
            assert snapshot.preview(limit=0) == []


def test_query_preview_stops_after_three_survivors_and_leaves_profile_unknown(
    tmp_path: Path,
) -> None:
    from premixdb.engine.queries import _value

    with p.PremixDB(storage=tmp_path, progress=False) as db:
        snapshot = db.Corpus("filtered", [p.Source(str(i), f"hello {i}") for i in range(8)])
        query = snapshot.query(steps=[p.where(p.text.characters > 0)])
        service = coordinator(db)
        with (
            patch.object(service, "_execute_query", side_effect=AssertionError("full query")),
            patch("premixdb.engine.queries._value", wraps=_value) as predicate,
        ):
            examples = query.preview()
            assert len(examples) == 3
            assert predicate.call_count == 3
            assert query.status is p.ExecutionStatus.PENDING
            assert not query._proto.HasField("profile")
            assert not service._storage.list("query", q.Query)
        assert query.profile().output_documents == 8
        assert query.preview() == examples


def test_fewer_than_three_results_can_require_exhausting_the_query(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path, progress=False) as db:
        snapshot = db.Corpus(
            "sparse", [p.Source(str(i), "yes" if i == 7 else "no") for i in range(8)]
        )
        query = snapshot.query(steps=[p.where(p.text.characters == 3)])
        examples = query.preview()
        assert [row["text"] for row in examples] == ["yes"]
        assert query.status is p.ExecutionStatus.PENDING
        assert query.profile().output_documents == 1
        assert query.preview() == examples


def test_global_dedupe_preview_keeps_the_same_winners_without_publishing(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path, progress=False) as db:
        query = db.Corpus(
            "dedupe", [p.Source("a", "same"), p.Source("z", "same"), p.Source("b", "unique")]
        ).query(steps=[p.dedupe(order_by=[p.object.uri.desc()])])
        examples = query.preview()
        assert {row["source_key"] for row in examples} == {"z", "b"}
        assert query.status is p.ExecutionStatus.PENDING
        assert not coordinator(db)._storage.list("query", q.Query)
        assert query.profile().output_documents == 2
        assert query.preview() == examples


def test_decontamination_preview_stops_after_three_target_documents(tmp_path: Path) -> None:
    from premixdb.engine.curation import decontaminate

    with p.PremixDB(storage=tmp_path, progress=False) as db:
        target = db.Corpus("target", [p.Source(str(i), f"hello {i}\nDROP\nend") for i in range(8)])
        reference = db.Corpus("reference", [p.Source("ref", "DROP")])
        query = target.query(
            decontaminate=p.Decontaminate(reference, algorithm="line", granularity="span")
        )
        with (
            patch.object(
                coordinator(db), "_execute_query", side_effect=AssertionError("full target query")
            ),
            patch("premixdb.engine.curation.decontaminate", wraps=decontaminate) as curate,
        ):
            examples = query.preview()
            assert len(examples) == 3
            assert curate.call_count == 3
            assert all("DROP" not in row["text"] for row in examples)
        assert query.status is p.ExecutionStatus.PENDING
        assert query.profile().output_documents == 8
        assert query.preview() == examples


@pytest.mark.parametrize(
    "tokenizer", [p.ByteTokenizer(), wordpiece_tokenizer()], ids=["bytes", "model"]
)
def test_dataset_preview_encodes_only_needed_documents_and_matches_full_packing(
    tmp_path: Path, tokenizer: m.Tokenizer
) -> None:
    with p.PremixDB(storage=tmp_path, progress=False) as db:
        query = db.Corpus("packed", [p.Source(str(i), "hello " * 64) for i in range(8)]).query()
        mixture = query.mix(
            tokenizer=tokenizer, packing=tokenizer_packing(tokenizer), sequence_length=4
        )
        dataset = mixture[0]
        service = coordinator(db)
        original = HuggingFaceTokenizer.encode_with_offsets
        encoded = []

        def encode(
            model: HuggingFaceTokenizer, text: str
        ) -> tuple[list[int], list[tuple[int, int]]]:
            encoded.append(text)
            return original(model, text)

        with (
            patch.object(service, "_execute_query", side_effect=AssertionError("full query")),
            patch.object(service, "_profile_dataset", side_effect=AssertionError("exact profile")),
            patch("premixdb.storage.tokens.publish", side_effect=AssertionError("full packing")),
            patch.object(HuggingFaceTokenizer, "encode_with_offsets", encode),
        ):
            examples = dataset.preview()
        assert len(examples) == 3
        assert len(encoded) == (1 if tokenizer.HasField("hugging_face") else 0)
        assert dataset.status is query.status is p.ExecutionStatus.PENDING
        assert not service._storage.list("dataset", m.Dataset)
        assert not dataset._proto.HasField("profile")
        dataset.wait()
        assert dataset.preview() == examples
        assert dataset.profile().source_documents == 8


@pytest.mark.parametrize(
    "packing",
    [p.Concat(separator=256), p.Concat(separator=256, drop_remainder=False, pad_token=257)],
)
def test_prefix_packing_handles_empty_documents_boundaries_and_final_tails(
    tmp_path: Path, packing: m.Packing
) -> None:
    with p.PremixDB(storage=tmp_path, progress=False) as db:
        dataset = (
            db.Corpus("tails", [p.Source("a", "é"), p.Source("b", ""), p.Source("c", "🌍end")])
            .query()
            .mix(tokenizer=p.ByteTokenizer(), packing=packing, sequence_length=5)[0]
        )
        examples = dataset.preview(limit=3)
        second = dataset.preview(offset=1, limit=1)
        assert dataset.status is p.ExecutionStatus.PENDING
        dataset.wait()
        assert dataset.preview(limit=3) == examples
        assert dataset.preview(offset=1, limit=1) == second


def test_sampled_preview_packs_a_prefix_without_dataset_profiling(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path, progress=False) as db:
        dataset = (
            db.Corpus("sampled", [p.Source("a", "abcdef"), p.Source("b", "uvwxyz")])
            .query()
            .mix(
                domains=p.object.uri,
                weights={"a": 0.5, "b": 0.5},
                tokens=12,
                tokenizer=p.ByteTokenizer(),
                sequence_length=3,
            )[0]
        )
        with patch.object(
            coordinator(db), "_profile_dataset", side_effect=AssertionError("exact profile")
        ):
            examples = dataset.preview()
        assert len(examples) == 3
        assert dataset.status is p.ExecutionStatus.PENDING
        dataset.wait()
        assert dataset.preview() == examples


def test_mixture_registration_and_empty_windows_do_not_execute(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path, progress=False) as db:
        query = db.Corpus("lazy", [p.Source("a", "hello")]).query()
        service = coordinator(db)
        with (
            patch.object(service, "_mix_pool", side_effect=AssertionError("inventory")),
            patch.object(service, "_tokenizer", side_effect=AssertionError("model loading")),
            patch.object(service, "_query", side_effect=AssertionError("query execution")),
        ):
            mixture = query.mix(weights=p.RegMix(), n_candidates=5)
            assert len(mixture) == 5
            assert "unresolved" in repr(mixture)
            assert len(mixture[1:3]) == 2
            assert not mixture.preview(limit=0).candidates
            assert not mixture[:0].preview().candidates
            with pytest.raises(IndexError):
                mixture[5]
        assert query.status is p.ExecutionStatus.PENDING


def test_preview_of_new_packing_reads_completed_selection_incrementally(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path, progress=False) as db:
        target = db.Corpus(
            "target", [p.Source(str(i), "hello " * 32 + "\nDROP\nend") for i in range(8)]
        )
        reference = db.Corpus("reference", [p.Source("ref", "DROP")])
        query = target.query(
            decontaminate=p.Decontaminate(reference, algorithm="line", granularity="span")
        ).wait()
        dataset = query.mix(tokenizer=p.ByteTokenizer(), sequence_length=4)[0]
        with patch.object(
            coordinator(db), "_query", side_effect=AssertionError("restored complete selection")
        ):
            examples = dataset.preview()
        assert len(examples) == 3
        dataset.wait()
        assert dataset.preview() == examples
