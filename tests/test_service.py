"""Fluent API -> public protobuf queries -> Python engine."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from typing import cast
from unittest.mock import patch

from _type_support import coordinator

import premixdb
from premixdb.engine import execution
from premixdb.runtime import Coordinator, compile_query
from premixdb.schemas import requests as _requests
from premixdb.schemas.ids import _decode_id, _encode_id
from premixdb.schemas.protobuf import descriptor
from premixdb.v1 import corpus_pb2 as corpora
from premixdb.v1 import data_mixture_pb2 as datasets
from premixdb.v1 import query_pb2 as queries
from premixdb.v1 import snapshot_pb2 as snapshots
from premixdb.v1 import status_pb2 as common
from premixdb.v1 import storage_pb2 as source_types


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.client = premixdb.PremixDB(storage=self.root / "local")
        self.addCleanup(self.client.close)

    def snapshot(self, client: premixdb.PremixDB | None = None) -> premixdb.Snapshot:
        client = client or self.client
        return client._create_corpus("test").snapshot(
            source=[
                premixdb.Source("a", "é\nshared"),
                premixdb.Source("b", "shared"),
                premixdb.Source("empty", ""),
            ]
        )

    def recipe(self, snapshot: premixdb.Snapshot) -> premixdb.Query:
        return snapshot.query(
            steps=[
                premixdb.where(premixdb.text.characters > 0),
                premixdb.dedupe(
                    algorithm=premixdb.DedupeAlgorithm.EXACT_LINE,
                    order_by=[premixdb.object.uri.asc()],
                ),
            ]
        )

    def test_minimal_researcher_api_and_defaults(self) -> None:
        import os

        with patch.dict(os.environ, {"PREMIXDB_STORAGE": str(self.root / "default")}):
            with premixdb.PremixDB() as client:
                snapshot = client.Corpus("shakespeare", [premixdb.Source("a", "To be")])
                query = snapshot.query(steps=[premixdb.where(premixdb.text.characters > 0)])
                mixture = query.mix()
                self.assertEqual(len(mixture), 1)
                candidate = mixture[0]
                self.assertIs(candidate.status, premixdb.ExecutionStatus.PENDING)
                self.assertFalse(hasattr(coordinator(client), "_dataset_handles"))
                self.assertEqual(candidate.profile().planned_content_tokens, 2)
                self.assertEqual(candidate._proto.sequence_length, 2048)
                self.assertEqual(len(candidate), 1)
                self.assertEqual(candidate[0].tokens[:3], [2514, 307, 50256])
                self.assertEqual(candidate.profile().padding_tokens, 2045)
                self.assertEqual(query._proto.git_commit, snapshot._proto.git_commit)
                self.assertEqual(candidate._proto.git_commit, query._proto.git_commit)

    def test_local_client_uses_coordinator_and_reuses_policy_independent_index(self) -> None:
        self.assertIsInstance(coordinator(self.client), Coordinator)
        snapshot = self.snapshot()
        self.assertIs(snapshot.status, premixdb.ExecutionStatus.COMPLETED)
        first = snapshot.query(
            steps=[premixdb.dedupe(algorithm=premixdb.DedupeAlgorithm.EXACT_LINE)]
        )
        first.wait()
        index = next(iter(coordinator(self.client)._indexes.values()))
        second = snapshot.query(
            steps=[
                premixdb.where(premixdb.object.uri != "a"),
                premixdb.dedupe(algorithm=premixdb.DedupeAlgorithm.EXACT_LINE),
            ]
        )
        self.assertIs(index, next(iter(coordinator(self.client)._indexes.values())))
        self.assertEqual(second.profile().output_documents, 2)
        self.assertNotEqual(first.id, second.id)
        self.client.close()
        self.client.close()
        with self.assertRaises(ValueError):
            self.client._create_corpus("closed")

    def test_fluent_local_path_builds_proto_and_executes_python(self) -> None:
        snapshot = self.snapshot()
        query = self.recipe(snapshot)
        self.assertEqual(query.profile().output_documents, 1)
        self.assertIsInstance(query._request, queries.CreateQueryRequest)
        self.assertIsInstance(query._proto, queries.Query)
        self.assertEqual(query._proto.status, common.STATUS_COMPLETED)
        self.assertEqual(
            (
                query.profile().output_documents,
                query.profile().output_content_bytes,
                query.profile().output_characters,
            ),
            (1, 9, 8),
        )
        self.assertNotIn("engine", descriptor(query._proto).fields_by_name)
        query._proto.Clear()
        self.assertTrue(query.id)
        dataset = query.mix(
            tokenizer=premixdb.ByteTokenizer(),
            sequence_length=4,
            packing=premixdb.Concat(separator=256, drop_remainder=False, pad_token=257),
        )[0]
        self.assertIsInstance(dataset._request, datasets.CreateDatasetRequest)
        self.assertEqual(
            [t for sequence in dataset for t in sequence.tokens],
            list("é\nshared".encode()) + [256, 257, 257],
        )
        self.assertEqual(
            [m for sequence in dataset for m in sequence.mask], [True] * 10 + [False] * 2
        )
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.profile().padding_tokens, 2)
        self.assertEqual(self.client._query(query.id).profile(), query.profile())
        self.assertEqual(self.client._snapshot(snapshot.id).profile(), snapshot.profile())
        self.assertEqual(
            self.client._dataset(dataset.id)._tokenizer_definition, dataset._tokenizer_definition
        )

    def test_differential_capture_and_client_ownership(self) -> None:
        path = self.root / "input.txt"
        path.write_bytes(b"abc\r\n")
        corpus = self.client._create_corpus("files")
        first = corpus.snapshot(source=path)
        self.assertIs(corpus.snapshot(source=path, base=first), first)
        path.write_bytes(b"xyz")
        changed = corpus.snapshot(source=path, base=first)
        self.assertNotEqual(first.id, changed.id)
        self.assertEqual(first.query().profile().output_content_bytes, 5)
        other = premixdb.PremixDB(storage=self.root / "other")
        self.addCleanup(other.close)
        with self.assertRaises(ValueError):
            first.union(self.snapshot(other))

    def test_pathlike_sources_use_the_filesystem_protocol(self) -> None:
        path = self.root / "input.txt"
        path.write_text("captured text", encoding="utf-8")

        class FilePath:
            def __fspath__(self) -> str:
                return str(path)

            def __str__(self) -> str:
                raise AssertionError("source paths must use __fspath__")

        expected = self.client.Corpus("paths", path)
        actual = self.client.Corpus("paths", FilePath(), base=expected)
        self.assertEqual(actual.id, expected.id)
        self.assertEqual(actual.preview()[0]["text"], "captured text")

    def test_resolve_without_execution_and_worker_validates_query_identity(self) -> None:
        snapshot = self.snapshot()
        request = premixdb.query(snapshot.id, steps=[premixdb.dedupe()])
        original = request.SerializeToString()
        with patch.object(
            execution, "execute", side_effect=AssertionError("resolver executed data")
        ):
            resolved = compile_query(request)
        self.assertEqual(request.SerializeToString(), original)
        restored = queries.Query.FromString(resolved.SerializeToString())
        result = coordinator(self.client).run_query(restored)
        self.assertEqual(result.id, resolved.id)
        repeated = premixdb.query(snapshot.id, snapshot.id, steps=[premixdb.dedupe()])
        self.assertEqual(resolved, compile_query(repeated))
        for mutate in (
            lambda q: setattr(q, "id", b"x" * 32),
            lambda q: q.operations.add().CopyFrom(premixdb.where(premixdb.text.bytes > 100)),
            lambda q: setattr(q, "status", common.STATUS_COMPLETED),
        ):
            broken = queries.Query.FromString(resolved.SerializeToString())
            mutate(broken)
            with self.assertRaises(ValueError):
                coordinator(self.client).run_query(broken)
        with patch(
            "premixdb.runtime.environment.current_code",
            return_value=execution.CodeVersion("premixdb://repository", "f" * 40, "00" * 32),
        ):
            with self.assertRaisesRegex(NotImplementedError, "Git revision"):
                coordinator(self.client).run_query(restored)

    def test_server_rejects_unsupported_or_ambiguous_wire_semantics(self) -> None:
        plan = premixdb.query(self.snapshot().id)
        cases = []
        for mutate in (
            lambda p: p.operations.add(where=queries.Comparison()),
            lambda p: p.operations.add(dedupe=queries.Dedupe()),
            lambda p: p.operations.add(
                where=queries.Comparison(
                    operator=queries.Comparison.OPERATOR_EQ,
                    field=queries.FIELD_TEXT_BYTES,
                    text="wrong type",
                )
            ),
            lambda p: p.operations.add(
                where=queries.Comparison(
                    operator=queries.Comparison.OPERATOR_EQ,
                    field=cast(queries.IntrinsicField, 999),
                    count=0,
                )
            ),
        ):
            value = queries.CreateQueryRequest.FromString(plan.SerializeToString())
            mutate(value)
            cases.append(value)
        # Unknown fields must fail rather than silently changing query semantics.
        cases.append(
            queries.CreateQueryRequest.FromString(plan.SerializeToString() + b"\x98\x06\x01")
        )
        for case in cases:
            with self.assertRaises((ValueError, NotImplementedError)):
                compile_query(case)

    def test_concurrent_idempotent_submissions_execute_once_and_keep_results_owned(self) -> None:
        request = premixdb.query(self.snapshot().id, steps=[premixdb.dedupe()])
        service = coordinator(self.client)
        with patch.object(service, "run_query", wraps=service.run_query) as execute:
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda _: service.CreateQuery(request), range(4)))
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(len(service._query_handles), 1)
        self.assertTrue(all(r == results[0] for r in results))
        results[0].Clear()
        self.assertTrue(service.CreateQuery(request).id)
        request.operations.add().CopyFrom(premixdb.where(premixdb.text.bytes > 10))
        self.assertNotEqual(service.CreateQuery(request).id, results[1].id)

    def test_failed_query_submission_can_retry_and_cache_success(self) -> None:
        service = coordinator(self.client)
        request = premixdb.query(self.snapshot().id)
        with patch.object(service, "run_query", wraps=service.run_query) as execute:
            execute.side_effect = RuntimeError("temporary worker failure")
            with self.assertRaisesRegex(RuntimeError, "temporary worker failure"):
                service.CreateQuery(request)
            failed = service.GetQuery(queries.GetQueryRequest(id=compile_query(request).id)).query
            self.assertEqual(failed.status, common.STATUS_ERROR)
            listed = service.ListQuery(queries.ListQueryRequest()).queries
            self.assertEqual(
                [(item.id, item.status, item.error) for item in listed],
                [(failed.id, common.STATUS_ERROR, failed.error)],
            )
            self.assertFalse(service._query_handles)
            execute.side_effect = None
            result = service.CreateQuery(request)
            self.assertEqual(service.CreateQuery(request), result)
            self.assertEqual(execute.call_count, 2)
        query = service.GetQuery(queries.GetQueryRequest(id=result.id)).query
        self.assertEqual(query.status, common.STATUS_COMPLETED)
        self.assertEqual(service.ListQuery(queries.ListQueryRequest()).queries, [query])

    def test_concurrent_failed_submissions_share_failure_then_allow_retry(self) -> None:
        service = coordinator(self.client)
        request = premixdb.query(self.snapshot().id)
        started, release = Event(), Event()
        waiters = Barrier(4)

        class WaitingFuture(Future[queries.Query]):
            def result(self, timeout: float | None = None) -> queries.Query:
                if not self.done():
                    waiters.wait(timeout=5)
                return super().result(timeout=timeout)

        def fail(query: queries.Query) -> None:
            started.set()
            if not release.wait(timeout=5):
                raise AssertionError("failed submission was not released")
            raise RuntimeError("temporary worker failure")

        with (
            patch("premixdb.runtime.materialization.Future", WaitingFuture),
            patch.object(service, "run_query", wraps=service.run_query) as execute,
        ):
            execute.side_effect = fail
            with ThreadPoolExecutor(max_workers=4) as pool:
                attempts = [pool.submit(service.CreateQuery, request)]
                try:
                    self.assertTrue(started.wait(timeout=5))
                    attempts.extend(pool.submit(service.CreateQuery, request) for _ in range(3))
                    waiters.wait(timeout=5)
                    self.assertEqual(execute.call_count, 1)
                finally:
                    release.set()
                for attempt in attempts:
                    with self.assertRaisesRegex(RuntimeError, "temporary worker failure"):
                        attempt.result(timeout=5)
            execute.side_effect = None
            result = service.CreateQuery(request)
            self.assertEqual(service.CreateQuery(request), result)
            self.assertEqual(execute.call_count, 2)

    def test_failed_dataset_publication_can_retry_and_cache_success(self) -> None:
        service = coordinator(self.client)
        request = _requests.dataset(
            self.snapshot().query().id, tokenizer=premixdb.ByteTokenizer(), sequence_length=4
        )
        id = service._dataset_id(service._resolve_dataset(request))
        with patch.object(service._storage, "save", wraps=service._storage.save) as save:
            save.side_effect = OSError("temporary publication failure")
            with self.assertRaisesRegex(OSError, "temporary publication failure"):
                service.CreateDataset(request)
            self.assertNotIn(id, service._datasets)
            self.assertFalse(service._storage.metadata.contains("dataset", id))
            with self.assertRaises(KeyError):
                service.GetDataset(datasets.GetDatasetRequest(id=id))
            save.side_effect = None
            result = service.CreateDataset(request)
            self.assertEqual(result.id, id)
            self.assertEqual(service.CreateDataset(request), result)
            self.assertEqual(
                sum(
                    call.kwargs.get("suffix")
                    not in (".recipe", ".profile", ".pending", ".selection")
                    and call.args[0] == "dataset"
                    for call in save.call_args_list
                ),
                1,
            )
        dataset = service.GetDataset(datasets.GetDatasetRequest(id=id)).dataset
        self.assertEqual(dataset.status, common.STATUS_COMPLETED)

    def test_sequence_indexes_are_stored_and_reads_skip_the_control_plane(self) -> None:
        snapshot = self.client._create_corpus("pages").snapshot(
            source=[premixdb.Source("a", "x" * 257)]
        )
        dataset = (
            snapshot.query()
            .mix(tokenizer=premixdb.ByteTokenizer(), sequence_length=1, packing=premixdb.Concat())[
                0
            ]
            .wait()
        )
        self.assertEqual(len(dataset._proto.sequences), 3)
        with patch.object(self.client, "_get", side_effect=AssertionError("reader called RPC")):
            self.assertEqual(dataset[128].tokens, [120])
            self.assertEqual([sequence.ordinal for sequence in dataset], list(range(257)))

    def test_resource_lists_follow_parent_ids_and_return_detached_pages(self) -> None:
        first = self.snapshot()
        second = self.client._create_corpus("other").snapshot(
            source=[premixdb.Source("other", "text")]
        )
        query = first.union(second).query()
        dataset = query.mix(tokenizer=premixdb.ByteTokenizer(), sequence_length=1)[0]
        rpc = coordinator(self.client)
        request = corpora.ListCorpusRequest()
        page = rpc.ListCorpus(request)
        self.assertEqual(len(page.corpora), 2)
        snapshots_page = rpc.ListSnapshot(
            snapshots.ListSnapshotRequest(corpus_id=first._proto.corpus_id)
        )
        self.assertEqual([s.id for s in snapshots_page.snapshots], [_decode_id(first.id)])
        for snapshot in (first, second):
            page = rpc.ListQuery(queries.ListQueryRequest(snapshot_id=_decode_id(snapshot.id)))
            self.assertEqual([q.id for q in page.queries], [_decode_id(query.id)])
        page = rpc.ListDatasets(datasets.ListDatasetRequest(query_id=_decode_id(query.id)))
        self.assertEqual(page.datasets[0], dataset._proto)
        page.datasets[0].profile.Clear()
        self.assertGreater(dataset.profile().sequences, 0)

    def test_unknown_fields_are_rejected_inside_nested_requests(self) -> None:
        snapshot = self.snapshot()
        query = snapshot.query()
        request = _requests.dataset(query.id, tokenizer=premixdb.ByteTokenizer(), sequence_length=4)
        # Unknown nested configuration must not be silently ignored.
        unknown = datasets.Concat.FromString(
            request.packing.concat.SerializeToString() + b"\x98\x06\x01"
        )
        request.packing.concat.CopyFrom(unknown)
        with self.assertRaises(NotImplementedError):
            coordinator(self.client).CreateDataset(request)

    def test_read_only_session_uses_saved_resources_without_importing_engine(self) -> None:
        snapshot = self.snapshot()
        query = self.recipe(snapshot)
        dataset = query.mix(tokenizer=premixdb.ByteTokenizer(), sequence_length=2)[0].wait()
        with premixdb.PremixDB(storage=self.root / "local", read_only=True) as reader:
            self.assertEqual(reader._query(query.id).profile(), query.profile())
            self.assertEqual(
                [s.tokens for s in reader._dataset(dataset.id)], [s.tokens for s in dataset]
            )
        code = """
import json, sys, premixdb
with premixdb.PremixDB(storage=sys.argv[1], read_only=True) as db:
    print(json.dumps(db._dataset(sys.argv[2])[0].tokens))
assert "premixdb.engine.execution" not in sys.modules
assert "premixdb.runtime.coordinator" not in sys.modules
assert "premixdb.runtime.environment" not in sys.modules
assert "grpc" not in sys.modules
"""
        output = subprocess.check_output(
            [sys.executable, "-c", code, str(self.root / "local"), dataset.id], text=True
        )
        self.assertEqual(json.loads(output), dataset[0].tokens)

    def test_local_sources_are_disabled_by_default_and_root_is_enforced(self) -> None:
        service = Coordinator(self.root / "restricted", source_root=self.root / "inputs")
        self.addCleanup(service.close)
        inputs = (self.root / "inputs").resolve()
        (inputs / "nested").mkdir(parents=True)
        document = inputs / "nested/data.txt"
        document.write_text("text")
        (inputs / "alias.txt").symlink_to(document)
        source = source_types.Source(
            files=premixdb.FileSources(documents=[premixdb.FileSource(path=str(inputs))])
        )
        self.assertEqual(
            service._sources(source),
            ((), [("alias.txt", inputs / "alias.txt"), ("nested/data.txt", document)]),
        )
        outside = self.root / "outside.txt"
        outside.write_text("outside")
        (inputs / "escape.txt").symlink_to(outside)
        with self.assertRaisesRegex(
            ValueError, "source symlink escapes the configured source root"
        ):
            service._sources(source)
        with self.assertRaises(ValueError):
            service._sources(
                source_types.Source(
                    files=premixdb.FileSources(documents=[premixdb.FileSource(path="/etc/passwd")])
                )
            )
        service = Coordinator(self.root / "disabled")
        self.addCleanup(service.close)
        with self.assertRaises(NotImplementedError):
            service._sources(
                source_types.Source(
                    files=premixdb.FileSources(documents=[premixdb.FileSource(path=str(self.root))])
                )
            )

    def test_fluent_reader_partitions_and_checkpoints_match_execution(self) -> None:
        from _reference import Topology as NativeTopology

        dataset = (
            self.snapshot().query().mix(tokenizer=premixdb.ByteTokenizer(), sequence_length=1)[0]
        )
        native = coordinator(self.client)._query(dataset._proto.query_id).dataset(1, 256, 257)
        all_ordinals = []
        for rank in range(2):
            for worker in range(3):
                topology = premixdb.Topology(rank, 2, worker, 3)
                reader = dataset._reader(topology=topology)
                reference = native.reader(NativeTopology(rank, 2, worker, 3))
                first = next(reader, None)
                native_first = next(reference, None)
                self.assertEqual(
                    first.tokens if first else None, native_first.tokens if native_first else None
                )
                checkpoint = json.loads(json.dumps(reader.checkpoint()))
                self.assertEqual(checkpoint, dict(reference.checkpoint(), dataset=dataset.id))
                rest = list(dataset._reader(topology=topology, checkpoint=checkpoint))
                self.assertEqual([s.tokens for s in rest], [s.tokens for s in reference])
                all_ordinals.extend(([first.ordinal] if first else []) + [s.ordinal for s in rest])
        self.assertEqual(sorted(all_ordinals), list(range(len(dataset))))
        reader = dataset._reader()
        list(reader)
        self.assertEqual(list(dataset._reader(checkpoint=reader.checkpoint())), [])
        self.assertEqual(dataset[-1].tokens, native[-1].tokens)
        with self.assertRaises(ValueError):
            dataset._reader(topology=premixdb.Topology(world_size=0))
        with self.assertRaises(ValueError):
            dataset._reader(checkpoint=reader.checkpoint() | {"dataset": "00" * 32})

    def test_pending_dependencies_wait_and_execution_errors_are_explicit(self) -> None:
        snapshot = self.snapshot()
        pending = snapshot._proto
        pending.status = common.STATUS_PENDING
        handle = premixdb.Snapshot(self.client, pending)
        with patch.object(self.client, "_get", return_value=snapshot._proto) as get:
            self.assertEqual(handle.wait().profile(), snapshot.profile())
            get.assert_called_once()
        failed = snapshot._proto
        failed.status = common.STATUS_ERROR
        with self.assertRaisesRegex(premixdb.ExecutionError, "failed"):
            premixdb.Snapshot(self.client, failed).wait()
        with patch.object(self.client, "_get", return_value=pending):
            handle = premixdb.Snapshot(self.client, pending)
            with self.assertRaises(TimeoutError):
                handle.wait(timeout=0.01)

    def test_worker_receives_only_a_resolved_public_query(self) -> None:
        service = coordinator(self.client)
        snapshot = self.snapshot()
        with patch.object(service, "run_query", wraps=service.run_query) as execute:
            result = self.recipe(snapshot).wait()
            plan = execute.call_args.args[0]
            self.assertEqual(_encode_id(plan.id), result.id)
            self.assertEqual(plan.snapshot_ids, [_decode_id(snapshot.id)])
            self.assertNotIn("documents", descriptor(plan).fields_by_name)

    def test_failed_execution_does_not_publish_query_completion(self) -> None:
        service = coordinator(self.client)
        resolved = compile_query(premixdb.query(self.snapshot().id))
        with (
            patch.object(execution, "CorpusIndex") as index,
            patch.object(service._storage, "save", wraps=service._storage.save) as save,
        ):
            index.return_value.execute.side_effect = RuntimeError("input span failed")
            with self.assertRaisesRegex(RuntimeError, "input span failed"):
                service.run_query(resolved)
            save.assert_not_called()
        self.assertNotIn(resolved.id, service._queries)
        self.assertNotIn(resolved.id, service._query_handles)

    def test_worker_must_return_matching_completed_output(self) -> None:
        service = coordinator(self.client)
        request = premixdb.query(self.snapshot().id)
        with patch.object(service, "run_query", side_effect=lambda query: query):
            with self.assertRaisesRegex(ValueError, "completed result"):
                service.CreateQuery(request)
            self.assertEqual(
                service.GetQuery(
                    queries.GetQueryRequest(id=compile_query(request).id)
                ).query.status,
                common.STATUS_ERROR,
            )


if __name__ == "__main__":
    unittest.main()
