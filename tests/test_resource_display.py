"""Shell inspection explains stored construction without executing recipes."""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from _type_support import coordinator
from google.protobuf.message import Message

import premixdb as p
import premixdb as sdk
from premixdb import local
from premixdb._resources import Dataset, Mix, Query, Snapshot
from premixdb.v1 import dataset_pb2 as d
from premixdb.v1 import query_pb2 as q
from premixdb.v1 import snapshot_pb2 as s
from premixdb.v1 import status_pb2 as status
from premixdb.v1 import storage_pb2 as storage
from premixdb.v1.storage_pb2 import Source


def shell_display(
    value: local.Snapshot
    | local.Query
    | local.Dataset
    | sdk.Corpus
    | sdk.Snapshot
    | sdk.Query
    | sdk.Dataset
    | sdk.Mix
    | Message,
) -> str:
    output = io.StringIO()
    with contextlib.redirect_stdout(output), patch("builtins._", None, create=True):
        sys.__displayhook__(value)
    return output.getvalue().rstrip("\n")


def test_shell_plans_survive_reopening_and_do_not_materialize(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.corpus("display", [p.Source("a", "abcd"), p.Source("b", "xyz")])
        query = snapshot.query(
            steps=[p.where(p.text.characters >= 3), p.dedupe(order_by=[p.object.uri.desc()])]
        ).wait()
        mixture = query.mix(domains=p.object.uri, tokens=4, sequence_length=4, n_candidates=2)
        dataset = mixture[0]
        assert dataset.status is p.ExecutionStatus.PENDING
        reopened = [
            db._snapshot(snapshot.id),
            db._query(query.id),
            db._datasets(mixture.id),
            db._dataset(dataset.id),
        ]
        originals = [snapshot, query, mixture, dataset]
        with (
            patch.object(db, "_get", side_effect=AssertionError("display fetched metadata")),
            patch.object(db, "_submit", side_effect=AssertionError("display submitted work")),
            patch.object(db, "_dataset", side_effect=AssertionError("display fetched candidate")),
            patch.object(
                db._object_reader, "read", side_effect=AssertionError("display read data")
            ),
        ):
            for original, again in zip(originals, reopened, strict=True):
                assert shell_display(original) == repr(original) == str(original) == repr(again)
            assert "Source: captured inventory (2 documents)" in repr(snapshot)
            plan = repr(query)
            assert plan.index("1. where text.characters >= 3") < plan.index(
                "2. dedupe exact_document"
            )
            assert "order by object.uri desc" in plan
            assert "Domains: object.uri" in repr(mixture)
            assert "Candidates: 2 of 2" in repr(mixture)
            assert "Candidates: 1 of 2" in repr(mixture[1:])
            assert "Budget: 4 content tokens" in repr(mixture)
            assert "Sampler: RegMixSampler(" in repr(mixture)
            assert "Weights: {" in repr(dataset)
            assert "Tokenizer: GPT2Tokenizer()" in repr(dataset)
            assert "Sequence length: 4" in repr(dataset)
            assert "Packing: Concat(" in repr(dataset)
            assert snapshot.id in repr(snapshot.union(snapshot))
        assert dataset.status is p.ExecutionStatus.PENDING
        assert not coordinator(db)._storage.list("dataset", d.Dataset)


def test_pending_query_explains_projections_and_optional_stages_without_io() -> None:
    resource = q.Query(
        id=b"a" * 32,
        snapshot_ids=[b"b" * 32],
        status=status.STATUS_PENDING,
        operations=[
            p.where(p.topic[p.Topic.SCIENCE_AND_TECH] >= 0.7),
            p.where(p.embedding.harrier.component(0) < 0.5),
            p.where(p.quality.educational_value.is_null()),
            q.Operation(document_ids=q.DocumentSelection(ids=[b"c" * 32])),
            q.Operation(indexed_dedupe=q.IndexedDedupe(index_name="dupekit.lsh", threshold=0.8)),
            q.Operation(
                similarity_dedupe=q.SimilarityDedupe(
                    algorithm=q.SimilarityDedupe.JACCARD, threshold=0.9, n=3
                )
            ),
        ],
        decontaminate=q.Decontaminate(
            snapshot_ids=[b"d" * 32],
            algorithm=q.Decontaminate.ALGORITHM_EXACT_NGRAM,
            n=5,
            granularity=q.Decontaminate.SPAN,
        ),
        sampling=q.QuerySampling(documents=0, seed=0, replacement=False),
        fields=[
            q.FieldComparison(
                field=q.FIELD_WEBORGANIZER_TOPIC, projection=q.FieldComparison.TOP_CLASS
            )
        ],
        field_snapshot_ids=[b"e" * 32],
    )
    client = Mock()
    query = Query(client, resource)
    before = query._proto.SerializeToString()
    plan = shell_display(query)
    assert "status='pending'" in plan
    assert "weborganizer.topic.probability('Science & Tech.') >= 0.7" in plan
    assert "embedding.harrier.component(0) < 0.5" in plan
    assert "quality.educational_value.is_null() == True" in plan
    assert "select document IDs: " in plan
    assert "indexed dedupe dupekit.lsh; threshold=0.8" in plan
    assert "similarity dedupe jaccard; threshold=0.9; n=3" in plan
    assert "Decontaminate: exact_ngram; span; n=5" in plan
    assert "Sampling: 0 documents; seed=0; replacement=False" in plan
    assert "Fields: weborganizer.topic.label" in plan
    assert "Field snapshots: " in plan
    assert query._proto.SerializeToString() == before
    assert client.mock_calls == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            storage.Source(
                files=storage.FileSources(documents=[storage.FileSource(path="/tmp/docs")])
            ),
            "files: '/tmp/docs'",
        ),
        (
            storage.Source(
                hugging_face=storage.HuggingFaceDataset(
                    repository="org/data", split="train", revision="v1", text_column="body"
                )
            ),
            "Hugging Face 'org/data' (split='train', revision='v1', text_column='body')",
        ),
        (
            storage.Source(
                manifest=storage.SourceManifest(
                    objects={"doc": storage.ObjectRef(uri="file:///data/doc")}
                )
            ),
            "manifest: 'doc'",
        ),
        (
            storage.Source(
                memory=storage.MemorySources(
                    documents=[storage.MemorySource(text="DO NOT PRINT DOCUMENT TEXT", uri="a")]
                )
            ),
            "memory (1 documents)",
        ),
    ],
)
def test_snapshot_source_and_base(source: Source, expected: str) -> None:
    source.limit = 0
    snapshot = Snapshot(
        Mock(),
        s.Snapshot(id=b"a" * 32, corpus_id=b"b" * 16, parent_snapshot_id=b"c" * 32, source=source),
    )
    plan = repr(snapshot)
    assert expected + "; limit=0" in plan
    assert "Base snapshot: " in plan
    assert "DO NOT PRINT DOCUMENT TEXT" not in plan


def test_dataset_zero_token_options_and_model_assets() -> None:
    dataset = Dataset(
        Mock(),
        d.Dataset(
            id=b"a" * 32,
            query_id=b"b" * 32,
            status=status.STATUS_ERROR,
            error="packing failed",
            sequence_length=32,
            tokenizer=d.Tokenizer(
                hugging_face=d.HuggingFaceTokenizer(
                    asset=storage.ObjectRef(
                        uri="file:///data/tokenizer.json", blake3_digest=b"c" * 32
                    ),
                    json=b"DO NOT PRINT TOKENIZER JSON",
                    max_document_bytes=1024,
                )
            ),
            packing=d.Packing(
                concat=d.Concat(separator_token_id=0, pad_token_id=0, drop_remainder=False)
            ),
            sampling=d.Sampling(
                tokens=100,
                seed=0,
                replacement=False,
                max_epochs=1,
                domains=d.Domains(field=q.FIELD_OBJECT_URI),
                weights={"b": 0.75, "a": 0.25},
            ),
        ),
    )
    plan = repr(dataset)
    assert "Concat(separator=0, pad=0, drop_remainder=False)" in plan
    assert "HuggingFaceTokenizer(asset='file:///data/tokenizer.json'" in plan
    assert "max_document_bytes=1024" in plan
    assert "DO NOT PRINT TOKENIZER JSON" not in plan
    assert "Weights: {'a': 0.25, 'b': 0.75}" in plan
    assert "Max epochs: 1" in plan
    assert "Error: 'packing failed'" in plan


def test_large_mixture_display_is_bounded_and_does_not_fetch_candidates() -> None:
    client = Mock()
    mixture = Mix(
        client,
        d.Mix(id=b"a" * 32, dataset_ids=[i.to_bytes(32) for i in range(100)], n_candidates=100),
    )
    assert "Candidates: 100 of 100" in repr(mixture)
    assert "... (92 more)" in repr(mixture)
    assert "Candidates: 0 of 100" in repr(mixture[:0])
    assert client.mock_calls == []


def test_direct_local_api_retains_plan_after_input_list_is_mutated(tmp_path: Path) -> None:
    db = local.PremixDB(storage=tmp_path)
    snapshot = db.corpus("display").snapshot(source=[local.Source("a", "abcd")])
    steps = [local.where(local.text.bytes > 0), local.dedupe(order_by=[local.object.uri.asc()])]
    query = snapshot.query(steps=steps)
    steps.clear()
    dataset = query.dataset(
        tokenizer=local.ByteTokenizer(),
        sequence_length=4,
        packing=local.Concat(separator=0, pad_token=0, drop_remainder=False),
    )
    assert "Source keys: 'a'" in shell_display(snapshot)
    assert "1. where text.bytes > 0" in shell_display(query)
    assert "2. dedupe exact_document; remove document; order by object.uri asc" in repr(query)
    assert "ByteTokenizer()" in shell_display(dataset)
    assert "Concat(separator=0, pad=0, drop_remainder=False)" in repr(dataset)


def test_query_default_sampling_is_visible_without_execution(tmp_path: Path) -> None:
    with p.PremixDB(storage=tmp_path) as db:
        snapshot = db.corpus("sampling-display", [p.Source("a", "hello")])
        query = snapshot.query()
        with patch.object(p.Query, "wait", side_effect=AssertionError("execution")):
            assert "Sampling: all selected documents once; replacement=False" in repr(query)
            assert repr(db._query(query.id)) == repr(query)
        query.wait()
        assert "Sampling: all selected documents once; replacement=False (1 documents)" in repr(
            query
        )
