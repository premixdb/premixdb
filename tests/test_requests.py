"""Typed requests remain independent of execution and internal storage schemas."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from _type_support import invalid_call
from google.protobuf.descriptor import FileDescriptor
from google.protobuf.message import Message

import premixdb
from premixdb._protobuf import descriptor
from premixdb.v1 import corpus_pb2 as corpora
from premixdb.v1 import dataset_pb2 as datasets
from premixdb.v1 import query_pb2 as queries
from premixdb.v1 import snapshot_pb2 as snapshots
from premixdb.v1 import storage_pb2 as source_types


class RequestTests(unittest.TestCase):
    def test_public_protocol_is_independent_of_worker_and_storage_protocols(self) -> None:
        seen = set()

        def inspect(file: FileDescriptor) -> None:
            if file.name in seen:
                return
            seen.add(file.name)
            self.assertFalse(file.name.startswith("premixdb/server/"))
            self.assertNotIn(file.name, ("premixdb/v1/common.proto", "premixdb/v1/execution.proto"))
            for dependency in file.dependencies:
                inspect(dependency)

        inspect(datasets.DESCRIPTOR)
        inspect(queries.DESCRIPTOR)
        self.assertIn("premixdb/v1/storage.proto", seen)
        self.assertIn("premixdb/v1/dataset.proto", seen)
        self.assertNotIn("premixdb/v1/source.proto", seen)
        self.assertNotIn("Source", snapshots.DESCRIPTOR.message_types_by_name)
        for resource in (queries.Query, datasets.Dataset):
            fields = descriptor(resource).fields_by_name
            self.assertNotIn("spec", fields)
            self.assertIn("profile", fields)
            self.assertNotIn("manifest", fields)
        for spec in (queries.CreateQueryRequest, datasets.CreateDatasetRequest):
            self.assertNotIn("recipe", descriptor(spec).fields_by_name)
        self.assertFalse(datasets.DESCRIPTOR.services_by_name)
        self.assertFalse(queries.DESCRIPTOR.services_by_name)

    def test_each_stage_produces_a_serializable_request_and_uses_server_ids(self) -> None:
        create_corpus = premixdb.corpus("web")
        self.assertEqual(
            create_corpus,
            corpora.CreateCorpusRequest(
                name="web",
            ),
        )
        corpus = corpora.Corpus(id=b"c" * 16, name="web")
        create_snapshot = premixdb.snapshot(
            corpus,
            source=source_types.Source(
                memory=premixdb.MemorySources(
                    documents=[premixdb.MemorySource(uri="a", text="training")]
                )
            ),
        )
        snapshot = snapshots.Snapshot(id=b"s" * 32, corpus_id=corpus.id)
        create_query = premixdb.query(
            snapshot, steps=[premixdb.where(premixdb.text.characters > 0)]
        )
        query = queries.Query(id=b"q" * 32)
        create_dataset = premixdb.dataset(
            query, tokenizer=premixdb.byte_tokenizer(), sequence_length=256
        )
        for request in (create_corpus, create_snapshot, create_query, create_dataset):
            self.assertIsInstance(request, Message)
            if "request_id" in descriptor(request).fields_by_name:
                self.assertIsInstance(request.request_id, str)  # Optional durable retry key.
            self.assertEqual(type(request).FromString(request.SerializeToString()), request)
        self.assertEqual(create_snapshot.corpus_id, corpus.id)
        self.assertEqual(list(create_query.snapshot_ids), [snapshot.id])
        self.assertEqual(create_dataset.query_id, query.id)
        self.assertNotIn("recipe", descriptor(create_snapshot).fields_by_name)
        self.assertNotIn("recipe", descriptor(create_query).fields_by_name)
        self.assertNotIn("recipe", descriptor(create_dataset).fields_by_name)
        self.assertEqual(create_dataset.tokenizer.WhichOneof("kind"), "byte")
        self.assertEqual(create_dataset.tokenizer.definition_digest, b"")
        self.assertIsInstance(datasets.Dataset(), datasets.Dataset)

    def test_query_canonicalizes_only_union_and_copies_ordered_steps(self) -> None:
        first = premixdb.where(premixdb.text.bytes >= 12)
        second = premixdb.dedupe(
            algorithm=premixdb.DedupeAlgorithm.EXACT_LINE,
            removal=premixdb.SourceGroup("→"),
            order_by=[premixdb.object.uri.desc()],
        )
        third = premixdb.where(premixdb.object.uri != "excluded")
        steps = [first, second, third]
        request = premixdb.query(b"b" * 32, b"a" * 32, b"b" * 32, steps=steps)
        self.assertEqual(list(request.snapshot_ids), [b"a" * 32, b"b" * 32])
        self.assertEqual(
            [op.WhichOneof("kind") for op in request.operations],
            ["where", "dedupe", "where"],
        )
        compare = request.operations[0].where
        self.assertEqual(compare.operator, queries.Comparison.OPERATOR_GE)
        self.assertEqual(compare.field, queries.FIELD_TEXT_BYTES)
        self.assertEqual(compare.WhichOneof("value"), "count")
        self.assertEqual(compare.count, 12)
        dedupe = request.operations[1].dedupe
        self.assertEqual(dedupe.algorithm, queries.Dedupe.ALGORITHM_EXACT_LINE)
        self.assertEqual(dedupe.source_group_separator, "→")
        self.assertEqual(dedupe.order_by[0].direction, queries.OrderBy.DIRECTION_DESC)
        self.assertEqual(dedupe.order_by[0].field, queries.FIELD_OBJECT_URI)
        self.assertFalse(premixdb.dedupe().dedupe.HasField("source_group_separator"))
        original = request.SerializeToString()
        steps.clear()
        first.Clear()
        second.Clear()
        self.assertEqual(request.SerializeToString(), original)
        self.assertEqual(queries.CreateQueryRequest.FromString(original), request)

    def test_query_step_errors_identify_the_position_and_correction(self) -> None:
        first = premixdb.where(premixdb.language.en >= 0.75)
        cases = [
            (
                premixdb.decontaminate(b"r" * 32),
                "steps[1] contains p.decontaminate(...); pass it as "
                "query(decontaminate=p.decontaminate(...)) instead of inside steps",
            ),
            (
                premixdb.sample(documents=5),
                "steps[1] contains p.sample(...); pass it as "
                "query(sampling=p.sample(...)) instead of inside steps",
            ),
            (
                None,
                "steps[1] is NoneType; expected a query step "
                "created by p.where(...), p.dedupe(...), or another step constructor",
            ),
            (
                queries.Operation(),
                "steps[1] is an empty query step; "
                "use p.where(...), p.dedupe(...), or another step constructor",
            ),
        ]
        for invalid, message in cases:
            with self.subTest(message=message), self.assertRaises(ValueError) as error:
                invalid_call(premixdb.query, b"s" * 32, steps=iter([first, invalid]))
            self.assertEqual(str(error.exception), message)

    def test_decontamination_is_passed_separately_from_ordered_steps(self) -> None:
        operation = premixdb.where(premixdb.language.en >= 0.75)
        policy = premixdb.decontaminate(b"r" * 32)
        request = premixdb.query(b"s" * 32, steps=[operation], decontaminate=policy)
        self.assertEqual(list(request.operations), [operation])
        self.assertEqual(request.decontaminate, policy)

    def test_remote_sources_and_assets_are_descriptions_and_are_copied(self) -> None:
        asset = premixdb.ObjectRef(
            uri="file:///models/tokenizer.json",
            blake3_digest=b"a" * 32,
            size_bytes=100,
        )
        tokenizer = premixdb.hugging_face_tokenizer(asset, max_document_bytes=1024)
        self.assertEqual(tokenizer.hugging_face.asset, asset)
        asset.uri = "file:///elsewhere/changed"
        self.assertEqual(tokenizer.hugging_face.asset.uri, "file:///models/tokenizer.json")
        src = source_types.Source(manifest=premixdb.SourceManifest(objects={"a": asset}))
        request = premixdb.snapshot(b"c" * 16, source=src)
        src.manifest.objects["a"].uri = "file:///elsewhere/replaced"
        self.assertEqual(request.source.manifest.objects["a"].uri, "file:///elsewhere/changed")
        self.assertEqual(list(request.source.manifest.objects), ["a"])
        base = snapshots.Snapshot(id=b"b" * 32, corpus_id=b"c" * 16)
        for src in (
            source_types.Source(
                files=premixdb.FileSources(documents=[premixdb.FileSource(path="data")])
            ),
            source_types.Source(
                hugging_face=premixdb.HuggingFaceDataset(
                    repository="org/data",
                    revision="a" * 40,
                    split="train",
                )
            ),
            source_types.Source(manifest=premixdb.SourceManifest()),
        ):
            request = premixdb.snapshot(b"c" * 16, source=src, base=base)
            self.assertEqual(request.parent_snapshot_id, base.id)
        with self.assertRaises(ValueError):
            premixdb.snapshot(b"d" * 16, source=src, base=base)

    def test_zero_tokens_and_false_flags_retain_presence_on_wire(self) -> None:
        request = premixdb.dataset(
            b"d" * 32,
            tokenizer=premixdb.byte_tokenizer(),
            sequence_length=4,
            packing=premixdb.concat(separator=0, drop_remainder=False, pad_token=0),
        )
        restored = datasets.CreateDatasetRequest.FromString(request.SerializeToString())
        policy = restored.packing.concat
        self.assertTrue(policy.HasField("drop_remainder"))
        self.assertFalse(policy.drop_remainder)
        self.assertTrue(policy.HasField("pad_token_id"))
        self.assertEqual(policy.pad_token_id, 0)
        self.assertTrue(policy.HasField("separator_token_id"))
        self.assertEqual(policy.separator_token_id, 0)
        default = premixdb.concat().concat
        self.assertTrue(default.HasField("drop_remainder"))
        self.assertTrue(default.drop_remainder)
        self.assertFalse(default.HasField("pad_token_id"))

    def test_invalid_builder_inputs_fail_before_submission(self) -> None:
        cases = [
            lambda: premixdb.corpus(" "),
            lambda: invalid_call(premixdb.snapshot, b"c" * 16, source="local/file.txt"),
            lambda: premixdb.snapshot(b"c" * 16, source=source_types.Source()),
            lambda: premixdb.snapshot(
                b"c" * 16,
                source=source_types.Source(
                    files=premixdb.FileSources(documents=[premixdb.FileSource()])
                ),
            ),
            lambda: premixdb.snapshot(
                b"c" * 16,
                source=source_types.Source(
                    hugging_face=premixdb.HuggingFaceDataset(
                        repository="org/data",
                        revision="main",
                        split="train",
                    )
                ),
            ),
            lambda: premixdb.query(),
            lambda: premixdb.query("not-an-id"),
            lambda: premixdb.query(snapshots.Snapshot()),
            lambda: premixdb.query(b"s" * 32, steps=[queries.Operation()]),
            lambda: premixdb.where(premixdb.text.bytes > True),
            lambda: premixdb.where(premixdb.text.bytes > -1),
            lambda: invalid_call(premixdb.where, premixdb.object.uri == 1),
            lambda: invalid_call(premixdb.where, 0 < premixdb.text.bytes < 10),
            lambda: invalid_call(premixdb.dedupe, algorithm="paragraph"),
            lambda: premixdb.SourceGroup("//"),
            lambda: premixdb.SourceGroup("\ud800"),
            lambda: premixdb.hugging_face_tokenizer(
                premixdb.ObjectRef(uri="file:///local/tokenizer.json")
            ),
            lambda: premixdb.hugging_face_tokenizer(premixdb.ObjectRef(uri="file:///models/model")),
            lambda: premixdb.concat(drop_remainder=False),
            lambda: premixdb.concat(pad_token=0),
            lambda: premixdb.concat(separator=True),
            lambda: premixdb.dataset(b"q" * 32, tokenizer=datasets.Tokenizer(), sequence_length=1),
            lambda: premixdb.dataset(
                b"q" * 32, tokenizer=premixdb.byte_tokenizer(), sequence_length=0
            ),
        ]
        for index, build in enumerate(cases):
            with self.subTest(case=index), self.assertRaises((TypeError, ValueError)):
                build()

    def test_import_and_build_need_neither_execution_engine_nor_grpc(self) -> None:
        # Run in a clean interpreter so earlier local-engine tests cannot mask imports.
        code = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ("premixdb.engine.execution", "premixdb._runtime", "grpc"):
            raise AssertionError("unexpected dependency: " + fullname)
sys.meta_path.insert(0, Block())
import premixdb
from google.protobuf.message import Message
assert isinstance(premixdb.corpus("web"), Message)
assert isinstance(premixdb.query(b"s" * 32, steps=[premixdb.dedupe()]), Message)
assert isinstance(premixdb.dataset(b"q" * 32, tokenizer=premixdb.byte_tokenizer(), sequence_length=8), Message)
"""
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_local_bindings_match_schemas(self) -> None:
        root = Path(__file__).resolve().parents[1]
        subprocess.run(
            [sys.executable, str(root / "scripts/generate_protos.py"), "--check"], check=True
        )


if __name__ == "__main__":
    unittest.main()
