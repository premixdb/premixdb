"""Execute recipes and publish durable results in local storage."""

from __future__ import annotations

from collections.abc import MutableMapping
from concurrent.futures import Future, wait
from contextlib import ExitStack, closing
from functools import wraps
from pathlib import Path
from threading import Lock
from types import TracebackType
from typing import Callable, Concatenate, Hashable, Iterable, Literal, Self, cast
from weakref import WeakSet

from blake3 import blake3
from google.protobuf.message import Message

from .. import _requests, _runtime
from .._identity import corpus_id as _corpus_id
from .._inputs import source_files
from .._protobuf import copy_message, descriptor_name
from ..engine import execution
from ..engine.contracts import QuerySummary
from ..engine.dataset_plan import BYTE_DEFINITION, DatasetPlan, PackingPlan
from ..v1 import corpus_pb2 as corpora
from ..v1 import dataset_pb2 as datasets
from ..v1 import query_pb2 as queries
from ..v1 import snapshot_pb2 as snapshots
from ..v1 import status_pb2 as status
from ..v1 import storage_pb2 as storage
from . import mixing, profiles, tokens
from .cache import MemoryCache, _Namespace
from .catalog_reader import Catalog, _read
from .materialization import Materializer, SingleFlight
from .planner import compile_query, copy_fields, execution_steps, field_definitions, reject_unknown
from .storage import ObjectStore

type _CreateRequest = (
    corpora.CreateCorpusRequest
    | snapshots.CreateSnapshotRequest
    | queries.CreateQueryRequest
    | datasets.CreateDatasetRequest
    | datasets.CreateMixRequest
)
type _CreateResponse = (
    corpora.CreateCorpusResponse
    | snapshots.CreateSnapshotResponse
    | queries.CreateQueryResponse
    | datasets.CreateDatasetResponse
    | datasets.CreateMixResponse
)


def _operation[Request: _CreateRequest, Response: _CreateResponse, **P](
    method: Callable[Concatenate[Coordinator, Request, P], Response],
) -> Callable[Concatenate[Coordinator, Request, P], Response]:
    name = str(getattr(method, "__name__"))

    @wraps(method)
    def call(self: Coordinator, request: Request, *args: P.args, **kwargs: P.kwargs) -> Response:
        event = None
        try:
            reject_unknown(request)
            request = copy_message(request)
            digest = blake3(request.SerializeToString(deterministic=True)).digest()
            event = self._storage.metadata.begin_execution(name, digest)
            key = request.request_id
            cache_hit = False

            def run() -> Response:
                nonlocal cache_hit
                if not key:
                    return method(self, request, *args, **kwargs)
                identity = blake3(name.encode() + b"\0" + key.encode()).digest()
                try:
                    previous = self._storage.metadata.load(
                        "submission", identity, status.Submission
                    )
                except KeyError:
                    self._storage.metadata.save(
                        "submission", identity, status.Submission(request_digest=digest)
                    )
                    previous = self._storage.metadata.load(
                        "submission", identity, status.Submission
                    )
                if previous.request_digest != digest:
                    raise ValueError("idempotency key was reused for a different request")
                if previous.response:
                    from google.protobuf import symbol_database

                    response_type = cast(
                        type[Message], symbol_database.Default().GetSymbol(previous.response_type)
                    )
                    cached: Message = response_type()
                    cached.ParseFromString(previous.response)
                    cache_hit = True
                    # The submission key binds the request and operation to the
                    # response type recorded by the first successful invocation.
                    return cast(Response, cached)
                result = method(self, request, *args, **kwargs)
                self._storage.metadata.save(
                    "submission",
                    identity,
                    status.Submission(
                        request_digest=digest,
                        response=result.SerializeToString(deterministic=True),
                        response_type=descriptor_name(result),
                    ),
                    mutable=True,
                )
                return result

            result = self._submissions.run(("durable", name, key, digest), run) if key else run()
            if isinstance(result, snapshots.CreateSnapshotResponse):
                resource_id = result.snapshot.id
            elif isinstance(
                result,
                (
                    corpora.CreateCorpusResponse,
                    queries.CreateQueryResponse,
                    datasets.CreateDatasetResponse,
                    datasets.CreateMixResponse,
                ),
            ):
                resource_id = result.id
            else:
                raise TypeError("create operation returned an unsupported response")
            self._storage.metadata.end_execution(
                event, resource_id=resource_id, cache_hit=cache_hit
            )
            return result
        except Exception as error:
            if event is not None:
                self._storage.metadata.end_execution(event, error=str(error))
            raise

    return call


class Coordinator(Catalog):
    """Execute recipes locally."""

    def __init__(
        self,
        storage_path: str | Path | ObjectStore,
        *,
        workers: int = 1,
        source_root: str | Path | None = None,
        allow_local_files: bool = False,
        cache_bytes: int = 128 * 1024 * 1024,
        metadata_path: str | Path | None = None,
        process_workers: int = 0,
    ) -> None:
        if type(workers) is not int or workers <= 0:
            raise ValueError("workers must be a positive integer")
        if type(process_workers) is not int or process_workers < 0:
            raise ValueError("process_workers must be a nonnegative integer")
        self._source_root = Path(source_root).resolve() if source_root is not None else None
        self._allow_local_files = allow_local_files
        self._cache = MemoryCache(cache_bytes)
        self._corpora: _Namespace[bytes, corpora.Corpus] = self._cache.namespace("corpora")
        self._snapshots: _Namespace[bytes, snapshots.Snapshot] = self._cache.namespace("snapshots")
        self._queries: _Namespace[bytes, queries.Query] = self._cache.namespace("queries")
        self._datasets: _Namespace[bytes, datasets.Dataset] = self._cache.namespace("datasets")
        self._snapshot_handles: _Namespace[bytes, execution.Snapshot] = self._cache.namespace(
            "snapshot_handles"
        )
        self._snapshot_objects: _Namespace[bytes, dict[str, storage.ObjectRef]] = (
            self._cache.namespace("snapshot_objects")
        )
        self._dataset_profiles: _Namespace[bytes, datasets.DatasetProfile] = self._cache.namespace(
            "dataset_profiles"
        )
        self._query_handles: _Namespace[bytes, execution.Query] = self._cache.namespace(
            "query_handles"
        )
        self._indexes: _Namespace[tuple[bytes, ...], execution.CorpusIndex] = self._cache.namespace(
            "indexes"
        )
        self._mixes: _Namespace[bytes, datasets.Mix] = self._cache.namespace("mixes")
        self._mix_pools: _Namespace[tuple[bytes, bytes, bytes, bytes], execution.MixturePool] = (
            self._cache.namespace("mix_pools")
        )
        self._owned_mix_pools: WeakSet[execution.MixturePool] = WeakSet()
        self._profile_lock = Lock()
        self._index_lock = Lock()
        self._mix_pool_lock = Lock()
        self._submissions = SingleFlight()
        self._lock = Lock()
        self.pipeline = None
        with ExitStack() as startup:
            if isinstance(storage_path, ObjectStore):
                self._storage = storage_path
            else:
                self._storage = ObjectStore(storage_path, metadata_path=metadata_path)
                startup.callback(self._storage.close)
            self._jobs = Materializer[queries.Query | datasets.Dataset](workers)
            startup.callback(self._jobs.close)
            self._store = execution.Store(self._storage.root / "snapshot")
            from .encodings import Encodings

            self._encodings = Encodings(self._storage)
            if process_workers:
                from .partitions import PartitionStore
                from .pipeline import PartitionPipeline

                self.pipeline = PartitionPipeline(
                    PartitionStore(self._storage.root / "partitions"), workers=process_workers
                )
                startup.callback(self.pipeline.close)
            startup.pop_all()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        with ExitStack() as closing:
            closing.callback(self._storage.close)
            if self.pipeline is not None:
                closing.callback(self.pipeline.close)
            closing.callback(self._close_mix_pools)
            closing.callback(self._jobs.close)

    def _close_mix_pools(self) -> None:
        with self._mix_pool_lock:
            pools = tuple(self._owned_mix_pools)
        with ExitStack() as closing:
            for pool in pools:
                closing.callback(pool.close)

    def _once[R: Message](
        self, request: Message, run: Callable[[], R], *, mode: Hashable | None = None
    ) -> R:
        if isinstance(request, snapshots.CreateSnapshotRequest):
            return run()  # Source locations can contain new bytes on every capture.
        key = (descriptor_name(request), request.SerializeToString(deterministic=True), mode)
        return copy_message(self._submissions.run(key, run))

    @_operation
    def CreateCorpus(
        self, request: corpora.CreateCorpusRequest, *, timeout: float | None = None
    ) -> corpora.CreateCorpusResponse:
        _requests.corpus(request.name)

        def run() -> corpora.CreateCorpusResponse:
            result = corpora.Corpus(
                id=_corpus_id(request.name),
                name=request.name,
            )
            with self._lock:
                try:
                    result = self._storage.load("corpus", result.id, corpora.Corpus)
                except KeyError:
                    self._storage.save("corpus", result.id, result)
                result = self._corpora.setdefault(result.id, result)
            return corpora.CreateCorpusResponse(id=result.id)

        return self._once(request, run)

    def _snapshot(self, id: bytes) -> execution.Snapshot:
        _requests._id(id, 32)
        with self._lock:
            handle = self._snapshot_handles.get(id)
        if handle is None:
            handle = self._store.load(id.hex(), lazy=True)
            with self._lock:
                self._snapshot_handles[id] = handle
        return handle

    def _sources(
        self, source: storage.Source
    ) -> tuple[Iterable[tuple[str, str]], list[tuple[str, Path]]]:
        kind = source.WhichOneof("location")
        if kind == "memory":
            return [(doc.uri, doc.text) for doc in source.memory.documents], []
        if kind == "files":
            if not self._allow_local_files and self._source_root is None:
                raise NotImplementedError("local source paths are disabled for this executor")
            result = []
            for document in source.files.documents:
                path = Path(document.path).resolve()
                if self._source_root is not None and not path.is_relative_to(self._source_root):
                    raise ValueError("source path escapes the configured source root")
                for key, file in source_files(path):
                    if self._source_root is not None and not file.resolve().is_relative_to(
                        self._source_root
                    ):
                        raise ValueError("source symlink escapes the configured source root")
                    result.append((key, file))
            return (), result
        from .sources import capture

        return capture(source, self._source_root), []

    def _snapshot_resource(
        self, id: bytes, source: storage.Source | None = None
    ) -> snapshots.Snapshot:
        self._storage.verify_snapshot_metadata(id)
        info = self._store.describe(id.hex())
        result = snapshots.Snapshot(
            id=id,
            corpus_id=bytes.fromhex(info["corpus_id"]),
            status=status.STATUS_COMPLETED,
        )
        result.parent_snapshot_id = bytes.fromhex(info["parent_snapshot_id"])
        if source is not None:
            result.source.CopyFrom(source)
            # Capture input is a recipe artifact, not part of the statistics catalog.
            if result.source.HasField("memory"):
                result.ClearField("source")
        for field, value in info["changes"].items():
            setattr(result.profile, field, value)
        objects = self._objects(result.id)
        result.profile.documents = len(objects)
        result.profile.objects = len(objects)
        for object in objects.values():
            for field in ("spans", "content_bytes", "characters", "newlines"):
                setattr(
                    result.profile,
                    field,
                    getattr(result.profile, field) + getattr(object.profile, field),
                )
        result.profile.fields.extend(profiles.text_profiles(objects, result.corpus_id))
        return result

    def _objects(self, id: bytes) -> dict[str, storage.ObjectRef]:
        _requests._id(id, 32)
        if id not in self._snapshot_objects:
            result = {}
            for key, info in self._store.objects(id.hex()).items():
                digest = bytes.fromhex(info["content_digest"])
                result[key] = storage.ObjectRef(
                    uri=f"premixdb://content/{digest.hex()}",
                    blake3_digest=digest,
                    size_bytes=info["content_bytes"],
                    profile=storage.ObjectProfile(
                        spans=info["spans"],
                        content_bytes=info["content_bytes"],
                        characters=info["characters"],
                        newlines=info["newlines"],
                    ),
                )
            self._snapshot_objects[id] = result
            return result
        return self._snapshot_objects[id]

    @_operation
    def CreateSnapshot(
        self, request: snapshots.CreateSnapshotRequest, *, timeout: float | None = None
    ) -> snapshots.CreateSnapshotResponse:
        _requests.snapshot(
            request.corpus_id,
            source=request.source,
            base=request.parent_snapshot_id or None,
            git_commit=request.git_commit,
        )

        code = _runtime.resolve_code(request.git_commit)
        self.GetCorpus(corpora.GetCorpusRequest(id=request.corpus_id))

        def run() -> snapshots.CreateSnapshotResponse:
            base = (
                self._snapshot(request.parent_snapshot_id) if request.parent_snapshot_id else None
            )
            texts, files = self._sources(request.source)
            if request.source.HasField("limit"):
                from itertools import islice

                limit = request.source.limit
                texts = list(islice(texts, limit))
                files = files[: max(0, limit - len(texts))]
            handle = self._store.capture_inputs(
                request.corpus_id.hex(), texts, files, code, base, stream=True
            )
            id = bytes.fromhex(handle.id)
            with self._lock:
                self._snapshot_handles[id] = handle
                resource = self._snapshots.get(id)
                if resource is None:
                    try:
                        resource = self._storage.load("snapshot", id, snapshots.Snapshot)
                    except KeyError:
                        resource = self._snapshot_resource(id, request.source)
                        resource.git_commit = bytes.fromhex(code.commit)
                        preview = execution.execute([handle], [], code)
                        from .previewing import inline, publish

                        inline(self._storage, resource.preview, preview)
                        publish(self._storage, "snapshot", id, preview)
                        self._storage.save("snapshot", id, resource)
                    self._snapshots[id] = resource
                corpus = self._storage.load("corpus", request.corpus_id, corpora.Corpus)
                corpus.latest_snapshot_id = id
                self._storage.save("corpus", corpus.id, corpus, suffix=".latest")
                self._corpora[corpus.id] = corpus
            return snapshots.CreateSnapshotResponse(snapshot=resource)

        return self._once(request, run)

    @_read
    def GetSnapshot(
        self, request: snapshots.GetSnapshotRequest, *, timeout: float | None = None
    ) -> snapshots.GetSnapshotResponse:
        _requests._id(request.id, 32)
        with self._lock:
            resource = self._snapshots.get(request.id)
        if resource is None:
            try:
                resource = self._storage.load("snapshot", request.id, snapshots.Snapshot)
            except KeyError:
                resource = self._snapshot_resource(request.id)
        return snapshots.GetSnapshotResponse(snapshot=resource)

    def run_query(self, query: queries.Query) -> queries.Query:
        return self._execute_query(query)[0]

    def _execute_query(self, query: queries.Query) -> tuple[queries.Query, execution.Query]:
        """Verify the resolved recipe before publishing any kernel result."""
        reject_unknown(query)
        expected = compile_query(query)
        if query != expected:
            raise ValueError("query identity, execution revision, or inputs were modified")
        with ExitStack() as resources:
            if (
                query.field_snapshot_ids
                or query.index_snapshot_ids
                or query.HasField("sampling")
                or any(op.HasField("similarity_dedupe") for op in query.operations)
            ):
                from .enrichment import query_inputs

                index, steps = resources.enter_context(query_inputs(self, query))
            else:
                index = self._query_index(query.snapshot_ids)
                steps = execution_steps(query)
            from ..engine import plans

            for i, step in enumerate(steps):
                if step.kind != "Policy":
                    continue
                assert step.payload is not None
                if step.payload[0] == "decontaminate":
                    _, policy, _ = step.payload
                    reference_index = self._query_index(tuple(policy.snapshot_ids))
                    steps[i] = plans.policy(
                        step.definition,
                        ("decontaminate", policy, list(reference_index.documents.values())),
                    )
                elif step.payload[0] == "sample":
                    _, sampling, tokenizer = step.payload
                    if sampling.HasField("tokenizer_asset"):
                        tokenizer = self._tokenizer_asset(
                            sampling.tokenizer_asset,
                            sampling.max_document_bytes or 8 * 1024 * 1024,
                            sampling.tokenizer_json or None,
                        )
                    steps[i] = plans.policy(step.definition, ("sample", sampling, tokenizer))
            if self.pipeline is not None:
                index.class_provider = self.pipeline.exact_classes
                index.reference_provider = self.pipeline.references
            estimate = profiles.estimate_query(self, query)
            handle = index.execute(
                steps, _runtime.resolve_code(query.git_commit), field_definitions(query)
            )
        handle.field_snapshot_ids = tuple(query.field_snapshot_ids)
        handle._encoding_provider = self._encodings
        if bytes.fromhex(handle.id) != query.id:
            raise RuntimeError("Python output identity does not match the resolved query")
        result = copy_message(query)
        result.status = status.STATUS_COMPLETED
        result.estimate.CopyFrom(estimate)
        summary = handle.summary()
        result.profile.CopyFrom(query_profile(summary, len(query.snapshot_ids)))
        from .profiles import output_profiles

        result.profile.fields.extend(output_profiles(self, query, handle))
        for row in handle:
            result.profile.source_documents[row.corpus_id] += 1
            result.profile.source_content_bytes[row.corpus_id] += row.document.size
        if "decontamination" in summary:
            result.profile.decontamination.CopyFrom(
                queries.DecontaminationProfile(**summary["decontamination"])
            )
        if "sampling" in summary:
            result.profile.sampling.CopyFrom(queries.SamplingProfile(**summary["sampling"]))
        from .previewing import inline
        from .previewing import publish as publish_preview

        inline(self._storage, result.preview, handle)
        import json

        result.lineage.CopyFrom(
            self._storage.put(
                "query",
                json.dumps(handle.provenance(), sort_keys=True, separators=(",", ":")).encode(),
            )
        )
        from .selections import publish

        publish(self._storage, handle)
        publish_preview(self._storage, "query", result.id, handle)
        self._storage.save("query", result.id, result)
        with self._lock:
            self._query_handles[result.id] = handle
            self._queries[result.id] = result
        # Exact classes are lazy; refresh the retained size after construction.
        self._indexes.refresh(tuple(query.snapshot_ids))
        return copy_message(result), handle

    def _query_index(self, snapshot_ids: Iterable[bytes]) -> execution.CorpusIndex:
        key = tuple(snapshot_ids)
        with self._index_lock:
            index = self._indexes.get(key)
            if index is None:
                index = execution.CorpusIndex([self._snapshot(id) for id in key])
                self._indexes[key] = index
        return index

    @_operation
    def CreateQuery(
        self,
        request: queries.CreateQueryRequest,
        *,
        _lazy: bool = False,
        timeout: float | None = None,
    ) -> queries.CreateQueryResponse:
        resolved = compile_query(request)
        if (
            not _lazy
            and resolved.HasField("sampling")
            and resolved.sampling.HasField("tokenizer_asset")
        ):
            policy = resolved.sampling
            tokenizer = self._tokenizer_asset(
                policy.tokenizer_asset,
                policy.max_document_bytes or 8 * 1024 * 1024,
                policy.tokenizer_json or None,
            )
            policy.tokenizer_asset.CopyFrom(self._storage.put("tokenizer", tokenizer.asset_bytes))
            policy.ClearField("tokenizer_json")

        def run() -> queries.CreateQueryResponse:
            try:
                self._storage.load("query", resolved.id, queries.Query)
            except KeyError:
                try:
                    self._storage.load("query", resolved.id, queries.Query, suffix=".pending")
                except KeyError:
                    self._storage.save("query", resolved.id, resolved, suffix=".pending")
                if not _lazy:
                    future = self._schedule_query(resolved)
                    if not (resolved.field_snapshot_ids or resolved.index_snapshot_ids):
                        future.result()
            return queries.CreateQueryResponse(id=resolved.id)

        return self._once(resolved, run, mode="plan" if _lazy else "execute")

    def _plan_query(self, request: queries.CreateQueryRequest) -> queries.Query:
        """Save a query recipe and attach population bounds from published metadata."""
        request = copy_message(request)
        if request.HasField("sampling") and request.sampling.tokenizer_json:
            policy = request.sampling
            if blake3(policy.tokenizer_json).digest() != policy.tokenizer_asset.blake3_digest:
                raise ValueError("sampling tokenizer JSON differs from its digest")
            policy.tokenizer_asset.CopyFrom(self._storage.put("tokenizer", policy.tokenizer_json))
            policy.ClearField("tokenizer_json")
        id = self.CreateQuery(request, _lazy=True).id
        result = Catalog.GetQuery(self, queries.GetQueryRequest(id=id)).query
        # Creating the recipe again is an explicit retry, deferred until wait().
        if result.status == status.STATUS_ERROR:
            result.status = status.STATUS_PENDING
            result.ClearField("error")
        return result

    def _materialize[R: queries.Query | datasets.Dataset](
        self, kind: Literal["query", "dataset"], recipe: R, execute: Callable[[], R]
    ) -> Future[R]:
        if (kind == "query") != isinstance(recipe, queries.Query):
            raise TypeError("materialization kind differs from its recipe")
        resources = cast(
            MutableMapping[bytes, R], self._queries if kind == "query" else self._datasets
        )
        request_type = (
            queries.CreateQueryRequest if kind == "query" else datasets.CreateDatasetRequest
        )
        running = copy_message(recipe)
        running.status = status.STATUS_RUNNING
        with self._lock:
            resources[recipe.id] = running

        def run() -> R:
            event = self._storage.metadata.begin_execution(
                "materialize_" + kind, resource_id=recipe.id
            )
            try:
                try:
                    result = self._storage.load(kind, recipe.id, type(recipe))
                except KeyError:
                    result = execute()
                if result.id != recipe.id or copy_fields(result, request_type()) != copy_fields(
                    recipe, request_type()
                ):
                    raise ValueError(f"{kind} worker returned a result for a different recipe")
                if result.status != status.STATUS_COMPLETED or not result.HasField("profile"):
                    raise ValueError("worker did not return a completed result")
                self._storage.save(kind, result.id, result)
            except Exception as exc:
                self._storage.metadata.end_execution(event, error=str(exc))
                failed = copy_message(recipe)
                failed.status = status.STATUS_ERROR
                failed.error = str(exc)
                with self._lock:
                    resources[recipe.id] = failed
                # Failure state is not a cache entry: eviction must not silently
                # restart an expensive failed job. Completion always takes priority.
                try:
                    self._storage.save(kind, recipe.id, failed, suffix=".failed", failure=True)
                except Exception:
                    # Preserve the worker/publication failure if storage is unavailable.
                    pass
                raise
            self._storage.metadata.end_execution(event)
            with self._lock:
                resources[result.id] = copy_message(result)
            return result

        return cast(Future[R], self._jobs.ensure(kind, recipe.id, run))

    def _schedule_query(self, resolved: queries.Query) -> Future[queries.Query]:
        return self._materialize("query", resolved, lambda: self.run_query(copy_message(resolved)))

    def _wait_for_materialization(
        self, kind: Literal["query", "dataset"], identity: bytes, timeout: float
    ) -> bool:
        """Wake when a local job finishes; the catalog supplies its result or failure."""
        future = self._jobs.active(kind, identity)
        if future is None:
            return False
        done, _ = wait((future,), timeout=timeout)
        return bool(done)

    def _materialized_resource[R: queries.Query | datasets.Dataset](
        self,
        kind: Literal["query", "dataset"],
        identity: bytes,
        message_type: type[R],
        cache: MutableMapping[bytes, R],
        recipe_suffix: str,
    ) -> R:
        _requests._id(identity, 32)
        with self._lock:
            cached = cache.get(identity)
            if cached is not None and cached.status == status.STATUS_COMPLETED:
                return copy_message(cached)
        try:
            return self._storage.load(kind, identity, message_type)
        except KeyError:
            resource = self._storage.load(kind, identity, message_type, suffix=recipe_suffix)
        # Completion takes priority; an active retry supersedes an earlier failure.
        if self._jobs.active(kind, identity) is not None:
            resource.status = status.STATUS_RUNNING
            return resource
        try:
            return self._storage.load(kind, identity, message_type, suffix=".failed")
        except KeyError:
            return resource

    @_read
    def GetQuery(
        self, request: queries.GetQueryRequest, *, timeout: float | None = None
    ) -> queries.GetQueryResponse:
        result = self._materialized_resource(
            "query", request.id, queries.Query, self._queries, ".pending"
        )
        if result.status != status.STATUS_COMPLETED or not result.HasField("estimate"):
            result.estimate.CopyFrom(profiles.estimate_query(self, result))
        return queries.GetQueryResponse(query=result)

    def _query(self, id: bytes) -> execution.Query:
        with self._lock:
            handle = self._query_handles.get(id)
        if handle is None:
            resource = self.GetQuery(queries.GetQueryRequest(id=id)).query
            if resource.status != status.STATUS_COMPLETED:
                with self._lock:
                    future = self._jobs.active("query", id)
                if future is None and resource.status in (
                    status.STATUS_PENDING,
                    status.STATUS_RUNNING,
                ):
                    future = self._schedule_query(compile_query(resource))
                if future is not None:
                    resource = future.result()
                    assert isinstance(resource, queries.Query)
                if resource.status == status.STATUS_ERROR:
                    raise ValueError(resource.error)
            with self._lock:
                ready = self._query_handles.get(id)
            if ready is not None:
                handle = ready
            else:
                from .selections import restore

                if self._storage.metadata.contains("query", id, suffix=".selection"):
                    handle = restore(self, resource)
                else:
                    # Resources published before selection receipts were introduced.
                    handle = self._execute_query(compile_query(resource))[1]
                with self._lock:
                    self._query_handles[id] = handle
        return handle

    def _resolve_dataset(
        self, request: datasets.CreateDatasetRequest, *, _lazy: bool = False
    ) -> datasets.CreateDatasetRequest:
        parent = (
            Catalog.GetQuery(self, queries.GetQueryRequest(id=request.query_id))
            if _lazy
            else self.GetQuery(queries.GetQueryRequest(id=request.query_id))
        ).query
        code = _runtime.resolve_code(request.git_commit or parent.git_commit)
        if parent.git_commit != bytes.fromhex(code.commit):
            raise ValueError("dataset revision must match its query")
        spec = _requests.dataset(
            request.query_id,
            tokenizer=request.tokenizer if request.HasField("tokenizer") else None,
            sequence_length=request.sequence_length or 2048,
            packing=request.packing if request.HasField("packing") else None,
            sampling=request.sampling if request.HasField("sampling") else None,
            git_commit=code.commit,
        )
        if _lazy and spec.tokenizer.HasField("hugging_face"):
            from ..engine.datasets import _tokenizer_definition
            from .assets import read

            policy = spec.tokenizer.hugging_face
            relative = "tokenizer/objects/" + policy.asset.blake3_digest.hex()
            inline = policy.json or None
            if inline is None and policy.asset.uri == self._storage.object_uri(relative):
                inline = self._storage._get(relative)
            data = read(policy.asset, local_root=self._source_root, inline=inline)
            policy.asset.CopyFrom(self._storage.put("tokenizer", data))
            policy.ClearField("json")
            definition = bytes.fromhex(_tokenizer_definition(policy.asset.blake3_digest.hex()))
        else:
            tokenizer = self._tokenizer(spec)
            definition = bytes.fromhex(tokenizer.definition if tokenizer else BYTE_DEFINITION)
            if tokenizer is not None:
                policy = spec.tokenizer.hugging_face
                policy.asset.CopyFrom(self._storage.put("tokenizer", tokenizer.asset_bytes))
                policy.ClearField("json")
        if spec.tokenizer.definition_digest and spec.tokenizer.definition_digest != definition:
            raise ValueError("tokenizer definition does not match its policy")
        spec.tokenizer.definition_digest = definition
        if spec.HasField("sampling"):
            sampling = spec.sampling
            if not sampling.HasField("seed"):
                sampling.seed = 0
            if not sampling.HasField("replacement"):
                sampling.replacement = True
            if sampling.domains.WhichOneof("kind") is None:
                sampling.domains.field = queries.FIELD_SOURCE_CORPUS_ID
            mixing.validate_sampling(sampling)
        return spec

    def _tokenizer_asset(
        self, asset: storage.ObjectRef, limit: int, inline: bytes | None = None
    ) -> execution.HuggingFaceTokenizer:
        from .assets import read

        relative = "tokenizer/objects/" + asset.blake3_digest.hex()
        owned_uri = self._storage.object_uri(relative)
        if inline is None and asset.uri == owned_uri:
            inline = self._storage._get(relative)
        data = read(asset, local_root=self._source_root, inline=inline)
        return execution.HuggingFaceTokenizer.from_bytes(data, asset.blake3_digest.hex(), limit)

    def _tokenizer(
        self, spec: datasets.CreateDatasetRequest | datasets.CreateMixRequest | datasets.Dataset
    ) -> execution.HuggingFaceTokenizer | None:
        if spec.tokenizer.HasField("byte"):
            return None
        policy = spec.tokenizer.hugging_face
        return self._tokenizer_asset(policy.asset, policy.max_document_bytes, policy.json or None)

    def _mix_pool(
        self,
        spec: datasets.CreateDatasetRequest | datasets.CreateMixRequest | datasets.Dataset,
        strata: datasets.Domains,
    ) -> execution.MixturePool:
        if strata.HasField("fields"):
            from .._field_ids import field_name, selector_field
            from .catalog import build_id, plan
            from .planner import validate_projection

            query = self.GetQuery(queries.GetQueryRequest(id=spec.query_id)).query
            for selector in strata.fields.selectors:
                selector.field = selector_field(selector)
                selector.ClearField("field_name")
                if selector.operator or selector.WhichOneof("value") is not None:
                    raise ValueError("mixture domains must be field projections, not predicates")
                validate_projection(selector)
                if selector.field in (1, 2, 3, 4):
                    if selector.field_snapshot_id:
                        raise ValueError("intrinsic domains cannot contain a derivation pin")
                    continue
                definition = plan(query.snapshot_ids, field_name(selector.field), query.git_commit)
                pin = build_id(definition, field_name(selector.field))
                if selector.field_snapshot_id and selector.field_snapshot_id != pin:
                    raise ValueError("domain pin differs from its built-in recipe")
                selector.field_snapshot_id = pin
        key = (
            spec.query_id,
            spec.tokenizer.definition_digest,
            mixing.canonical_digest("mixture-strata", strata),
            _runtime.current_code().canonical_digest(),
        )
        with self._mix_pool_lock:
            pool = self._mix_pools.get(key)
            if pool is None:
                from ..engine.mixtures import MixturePool

                query = self._packing_query(spec)
                assignments = dict(strata.assignments.documents)
                if strata.HasField("fields"):
                    from .._mixing import _domain_key
                    from ..engine.curation import selector_key
                    from .enrichment import projections

                    values = projections(self, query, strata.fields.selectors)
                    assignments = {
                        r.id: _domain_key(
                            [values[selector_key(s)][r.id] for s in strata.fields.selectors]
                        )
                        for r in query
                    }
                pool = MixturePool(
                    query, mixing.domains_name(strata), assignments, self._tokenizer(spec)
                )
                self._owned_mix_pools.add(pool)
                self._mix_pools[key] = pool
            return pool

    def _sampling(
        self, spec: datasets.CreateDatasetRequest | datasets.Dataset
    ) -> tuple[execution.MixturePool, tuple[dict[str, int], str, int, bool, int | None]]:
        sampling = spec.sampling
        pool = self._mix_pool(spec, sampling.domains)
        counts = mixing.allocations(sampling.weights, sampling.tokens)
        cap = sampling.max_epochs if sampling.HasField("max_epochs") else None
        mixing.capacity_check(counts, pool.inventory(), sampling.replacement, cap)
        return pool, (
            counts,
            mixing.canonical_digest("mixture-sampling", sampling).hex(),
            sampling.seed,
            sampling.replacement,
            cap,
        )

    @staticmethod
    def _packing(spec: datasets.CreateDatasetRequest) -> tuple[int, int | None, int | None]:
        policy = spec.packing.concat
        return (
            spec.sequence_length,
            policy.separator_token_id if policy.HasField("separator_token_id") else None,
            policy.pad_token_id if policy.HasField("pad_token_id") else None,
        )

    def _dataset_plan(self, spec: datasets.CreateDatasetRequest) -> DatasetPlan:
        identity = spec.query_id.hex()
        if spec.HasField("sampling"):
            pool, args = self._sampling(spec)
            identity = pool.identity(*args)
        return DatasetPlan(
            identity,
            spec.tokenizer.definition_digest.hex(),
            _runtime.resolve_code(spec.git_commit),
            PackingPlan(*self._packing(spec)),
        )

    def _dataset_id(self, spec: datasets.CreateDatasetRequest) -> bytes:
        return bytes.fromhex(self._dataset_plan(spec).id)

    def _packing_query(
        self, spec: datasets.CreateDatasetRequest | datasets.CreateMixRequest | datasets.Dataset
    ) -> execution.Query:
        from copy import copy

        query = self._query(spec.query_id)
        code = _runtime.resolve_code(spec.git_commit)
        if query.code != code:
            # Selection is a frozen input. New packing pins the executing runtime
            # even when that input was produced by an older environment.
            query = copy(query)
            query.code = code
        return query

    def _dataset_handle(self, spec: datasets.CreateDatasetRequest) -> execution.Dataset:
        if spec.HasField("sampling"):
            pool, args = self._sampling(spec)
            return pool.dataset(*args, *self._packing(spec), stream=True)
        query = self._packing_query(spec)
        if self.pipeline is None:
            return query.dataset(*self._packing(spec), tokenizer=self._tokenizer(spec), stream=True)
        import time

        from ..engine.datasets import Dataset

        plan = self._dataset_plan(spec)
        lengths, encoded = self.pipeline.tokenize(query, spec.tokenizer)
        return Dataset(
            query,
            plan,
            query.source_counts(),
            encoded,
            start=time.monotonic(),
            lengths=lengths,
            stream=True,
        )

    def _profile_dataset(self, spec: datasets.CreateDatasetRequest) -> datasets.DatasetProfile:
        spec = self._resolve_dataset(spec)
        key = blake3(
            b"premixdb-dataset-profile-runtime/v1\0"
            + mixing.canonical_digest("dataset-profile", spec)
            + _runtime.current_code().canonical_digest()
        ).digest()
        with self._profile_lock:
            if key in self._dataset_profiles:
                return copy_message(self._dataset_profiles[key])
            try:
                profile = self._storage.load(
                    "dataset", key, datasets.DatasetProfile, suffix=".profile"
                )
            except KeyError:
                pass
            else:
                self._dataset_profiles[key] = profile
                return copy_message(profile)
            planned = {}
            if spec.HasField("sampling"):
                pool, args = self._sampling(spec)
                planned = args[0]
                packing = PackingPlan(*self._packing(spec))
                prepared = pool._prepare(*args)
                values = prepared.profile(packing)
                geometry = prepared.geometry(packing)
            else:
                handle = self._query(spec.query_id)
                tokenizer = self._tokenizer(spec)
                if tokenizer:
                    if self.pipeline is not None:
                        lengths, _ = self.pipeline.tokenize(handle, spec.tokenizer)
                    else:
                        from ..engine.token_cache import token_pool

                        with closing(token_pool(handle, tokenizer)) as tokens:
                            lengths = [tokens.length(row.id) for row in handle]
                    values = PackingPlan(*self._packing(spec)).profile(
                        handle.source_counts(),
                        lengths,
                    )
                else:
                    values = handle.profile(*self._packing(spec))
                    lengths = handle.lengths()
                geometry = PackingPlan(*self._packing(spec)).geometry(
                    (r.id, r.corpus_id, length) for r, length in zip(handle, lengths, strict=True)
                )
            profile = datasets.DatasetProfile(**values, **geometry, planned_stratum_tokens=planned)
            self._storage.save("dataset", key, profile, suffix=".profile")
            self._dataset_profiles[key] = profile
            return copy_message(profile)

    @_operation
    def CreateMix(
        self, request: datasets.CreateMixRequest, *, timeout: float | None = None
    ) -> datasets.CreateMixResponse:
        def run() -> datasets.CreateMixResponse:
            from .._mixing import RegMixSampler

            template = self._resolve_dataset(copy_fields(request, datasets.CreateDatasetRequest()))
            spec = copy_fields(request, datasets.CreateMixRequest())
            for name in ("tokenizer", "packing"):
                getattr(spec, name).CopyFrom(getattr(template, name))
            spec.git_commit = template.git_commit
            spec.sequence_length = template.sequence_length
            if spec.domains.WhichOneof("kind") is None:
                spec.domains.field = queries.FIELD_SOURCE_CORPUS_ID
            if spec.algorithm.WhichOneof("kind") is None:
                spec.algorithm.CopyFrom(RegMixSampler()._to_proto())
            else:
                policy = spec.algorithm.regmix
                defaults = RegMixSampler()._to_proto().regmix
                for name in (
                    "prior_power",
                    "min_concentration",
                    "max_concentration",
                    "concentration_steps",
                    "oversample",
                ):
                    if not getattr(policy, name):
                        setattr(policy, name, getattr(defaults, name))
                if not policy.HasField("seed"):
                    policy.seed = 0
            if not spec.HasField("seed"):
                spec.seed = 0
            if not spec.HasField("replacement"):
                spec.replacement = True
            spec.n_candidates = spec.n_candidates or 3
            inventory = self._mix_pool(template, spec.domains).inventory()
            spec.tokens = spec.tokens or sum(inventory.values())
            mixing.validate_mix(spec)
            candidates = mixing.generate(spec, inventory)
            result = copy_fields(spec, datasets.Mix())
            recipes = []
            single_domain = sum(value > 0 for value in inventory.values()) == 1
            for index, weights in enumerate(candidates):
                recipe = copy_message(template)
                recipe.sampling.CopyFrom(
                    datasets.Sampling(
                        domains=spec.domains,
                        weights=weights,
                        tokens=spec.tokens,
                        seed=(spec.seed + index) % (2**64) if single_domain else spec.seed,
                        replacement=spec.replacement,
                    )
                )
                if spec.bounds.HasField("max_epochs"):
                    recipe.sampling.max_epochs = spec.bounds.max_epochs
                resource = copy_fields(
                    recipe,
                    datasets.Dataset(
                        id=self._dataset_id(recipe),
                        status=status.STATUS_PENDING,
                        profile=self._profile_dataset(recipe),
                    ),
                )
                recipes.append(resource)
                result.dataset_ids.append(resource.id)
            result.id = mixing.canonical_digest("mix", result)
            if result.ByteSize() + sum(r.ByteSize() for r in recipes) > 3 * 1024 * 1024:
                raise ValueError("mixture metadata exceeds 3 MiB; use smaller candidate batches")
            for resource in recipes:
                self._storage.save("dataset", resource.id, resource, suffix=".recipe")
            self._storage.save("mixture", result.id, result)
            with self._lock:
                self._mixes[result.id] = result
            return datasets.CreateMixResponse(id=result.id)

        return self._once(request, run)

    @_operation
    def CreateDataset(
        self,
        request: datasets.CreateDatasetRequest,
        *,
        _lazy: bool = False,
        timeout: float | None = None,
    ) -> datasets.CreateDatasetResponse:
        spec = self._resolve_dataset(request, _lazy=_lazy)
        if _lazy:
            identity = self._dataset_id(spec)
            try:
                Catalog.GetDataset(self, datasets.GetDatasetRequest(id=identity))
            except KeyError:
                resource = copy_fields(
                    spec, datasets.Dataset(id=identity, status=status.STATUS_PENDING)
                )
                self._storage.save("dataset", identity, resource, suffix=".recipe")
            return datasets.CreateDatasetResponse(id=identity)

        def run() -> datasets.Dataset:
            id = self._dataset_id(spec)
            try:
                existing = self._storage.load("dataset", id, datasets.Dataset)
                if existing.status == status.STATUS_COMPLETED:
                    return existing
            except KeyError:
                pass
            with closing(self._dataset_handle(spec)) as native:
                handle = self.pipeline.pack(native) if self.pipeline is not None else native
                if bytes.fromhex(handle.id) != id:
                    raise RuntimeError("kernel dataset identity does not match its recipe")
                result = copy_fields(
                    spec,
                    datasets.Dataset(
                        id=id, status=status.STATUS_COMPLETED, profile=self._profile_dataset(spec)
                    ),
                )
                spans, batches, preview = tokens.publish(
                    self._storage,
                    handle,
                    profile=result.profile,
                    lineage=self._query(spec.query_id).provenance(),
                    tokenizer=self._tokenizer(spec),
                )
                result.tokens.extend(spans)
                result.sequences.extend(batches)
                result.preview.CopyFrom(preview)
                return result

        # Mix candidates and direct builds share the same registered recipe.
        id = self._dataset_id(spec)
        try:
            self._storage.load("dataset", id, datasets.Dataset)
        except KeyError:
            pass
        else:
            return datasets.CreateDatasetResponse(id=id)
        resource = copy_fields(
            spec,
            datasets.Dataset(
                id=id, status=status.STATUS_PENDING, profile=self._profile_dataset(spec)
            ),
        )
        try:
            self._storage.load("dataset", id, datasets.Dataset, suffix=".recipe")
        except KeyError:
            self._storage.save("dataset", id, resource, suffix=".recipe")
        return self._once(
            spec,
            lambda: datasets.CreateDatasetResponse(
                id=self._materialize("dataset", resource, run).result().id
            ),
        )

    def _planned_dataset_profile(self, resource: datasets.Dataset) -> datasets.DatasetProfile:
        profile = self._profile_dataset(copy_fields(resource, datasets.CreateDatasetRequest()))
        self._storage.save("dataset", resource.id, profile, suffix=".planned-profile")
        return profile

    def _plan_dataset(self, request: datasets.CreateDatasetRequest) -> datasets.Dataset:
        """Register a recipe without executing its query, profiling, or packing."""
        identity = self.CreateDataset(request, _lazy=True).id
        result = Catalog.GetDataset(self, datasets.GetDatasetRequest(id=identity)).dataset
        if result.status == status.STATUS_ERROR:
            result.status = status.STATUS_PENDING
            result.ClearField("error")
        return result

    @_read
    def GetDataset(
        self, request: datasets.GetDatasetRequest, *, timeout: float | None = None
    ) -> datasets.GetDatasetResponse:
        resource = self._materialized_resource(
            "dataset", request.id, datasets.Dataset, self._datasets, ".recipe"
        )
        return datasets.GetDatasetResponse(dataset=self._dataset_profile(resource))


def query_profile(summary: QuerySummary, snapshots: int) -> queries.QueryProfile:
    return queries.QueryProfile(
        snapshots=snapshots,
        input_documents=summary["input"]["documents"],
        input_content_bytes=summary["input"]["bytes"],
        input_characters=summary["input"]["characters"],
        output_documents=summary["output"]["documents"],
        output_content_bytes=summary["output"]["bytes"],
        output_characters=summary["output"]["characters"],
        steps=[
            queries.QueryStepProfile(
                input_documents=step["before"]["documents"],
                input_content_bytes=step["before"]["bytes"],
                input_characters=step["before"]["characters"],
                output_documents=step["after"]["documents"],
                output_content_bytes=step["after"]["bytes"],
                output_characters=step["after"]["characters"],
            )
            for step in summary["steps"]
        ],
    )
