"""Execute recipes and publish durable results in local storage."""

from __future__ import annotations

from collections.abc import MutableMapping
from concurrent.futures import Future, wait
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from threading import Lock
from types import TracebackType
from typing import (
    Callable,
    Concatenate,
    Generator,
    Hashable,
    Iterable,
    Literal,
    Mapping,
    Self,
    cast,
)
from weakref import WeakSet

from blake3 import blake3
from google.protobuf.message import Message

from premixdb.contracts import PreviewSequence
from premixdb.engine import execution
from premixdb.engine.contracts import QuerySummary
from premixdb.engine.dataset_plan import DatasetPlan
from premixdb.engine.names import corpus_id as _corpus_id
from premixdb.engine.sources import source_files
from premixdb.runtime import datasets as datasets_execution
from premixdb.runtime import environment as _runtime
from premixdb.runtime import mixtures as mixtures_execution
from premixdb.runtime.materialization import Materializer, SingleFlight
from premixdb.runtime.planner import (
    compile_query,
    copy_fields,
    execution_steps,
    field_definitions,
    reject_unknown,
)
from premixdb.schemas import requests as _requests
from premixdb.schemas.protobuf import copy_message, descriptor_name
from premixdb.storage import profiles
from premixdb.storage.cache import MemoryCache, _Namespace
from premixdb.storage.catalog import Catalog, _read
from premixdb.storage.objects import ObjectStore
from premixdb.v1 import corpus_pb2 as corpora
from premixdb.v1 import data_mixture_pb2 as datasets
from premixdb.v1 import query_pb2 as queries
from premixdb.v1 import snapshot_pb2 as snapshots
from premixdb.v1 import status_pb2 as status
from premixdb.v1 import storage_pb2 as storage

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
            from premixdb.runtime.encodings import Encodings

            self._encodings = Encodings(self._storage)
            if process_workers:
                from premixdb.runtime.partitions import PartitionStore
                from premixdb.runtime.pipeline import PartitionPipeline

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
    def CreateCorpus(self, request: corpora.CreateCorpusRequest) -> corpora.CreateCorpusResponse:
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
        from premixdb.runtime.sources import capture

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
        self, request: snapshots.CreateSnapshotRequest
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
                        from premixdb.storage.preview import inline, publish

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
    def GetSnapshot(self, request: snapshots.GetSnapshotRequest) -> snapshots.GetSnapshotResponse:
        _requests._id(request.id, 32)
        with self._lock:
            resource = self._snapshots.get(request.id)
        if resource is None:
            try:
                resource = self._storage.load("snapshot", request.id, snapshots.Snapshot)
            except KeyError:
                resource = self._snapshot_resource(request.id)
        return snapshots.GetSnapshotResponse(snapshot=resource)

    def _preview_rows(self, resource: queries.Query) -> Generator[execution.Row, None, None]:
        from premixdb.runtime.preview_execution import query_rows

        yield from query_rows(self, resource)

    @_read
    def Preview(self, request: queries.PreviewRequest) -> queries.PreviewResponse:
        from itertools import islice

        from premixdb.storage.preview import bounded_text

        limit, offset, width = _requests._preview_options(
            request.limit if request.HasField("limit") else 3,
            request.offset,
            request.max_characters if request.HasField("max_characters") else 1024,
            unit="documents",
        )
        if request.WhichOneof("input") != "query_id":
            return Catalog.Preview(self, request)
        resource = Catalog.GetQuery(self, queries.GetQueryRequest(id=request.query_id)).query
        if resource.status == status.STATUS_COMPLETED:
            return Catalog.Preview(self, request)
        result = queries.PreviewResponse()
        if not limit:
            return result
        rows = self._preview_rows(resource)
        try:
            for row in islice(rows, offset, offset + limit):
                text, truncated = bounded_text(self._storage, row, width)
                result.preview.documents.add(
                    id=bytes.fromhex(row.id),
                    corpus_id=bytes.fromhex(row.corpus_id),
                    source_key=row.source_key,
                    ordinal=row.ordinal,
                    text=text,
                    truncated=truncated,
                )
        finally:
            rows.close()
        return result

    def _preview_dataset(
        self, resource: datasets.Dataset, *, limit: int, offset: int, max_characters: int
    ) -> list[PreviewSequence]:
        from premixdb.runtime.preview_execution import dataset_preview

        return dataset_preview(
            self, resource, limit=limit, offset=offset, max_characters=max_characters
        )

    def run_query(self, query: queries.Query) -> queries.Query:
        return self._execute_query(query)[0]

    def _execute_query(
        self, query: queries.Query, *, publish: bool = True
    ) -> tuple[queries.Query, execution.Query]:
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
                from premixdb.runtime.enrichment import query_inputs

                index, steps = resources.enter_context(query_inputs(self, query))
            else:
                index = self._query_index(query.snapshot_ids)
                steps = execution_steps(query)
            from premixdb.engine import plans

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
        if not publish:
            return copy_message(query), handle
        result = copy_message(query)
        result.status = status.STATUS_COMPLETED
        result.estimate.CopyFrom(estimate)
        summary = handle.summary()
        result.profile.CopyFrom(query_profile(summary, len(query.snapshot_ids)))
        from premixdb.runtime.profiles import output_profiles

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
        from premixdb.storage.preview import inline
        from premixdb.storage.preview import publish as publish_preview

        inline(self._storage, result.preview, handle)
        import json

        result.lineage.CopyFrom(
            self._storage.put(
                "query",
                json.dumps(handle.provenance(), sort_keys=True, separators=(",", ":")).encode(),
            )
        )
        from premixdb.storage.selections import publish as publish_selection

        publish_selection(self._storage, handle)
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
    def GetQuery(self, request: queries.GetQueryRequest) -> queries.GetQueryResponse:
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
                from premixdb.storage.selections import restore

                if self._storage.metadata.contains("query", id, suffix=".selection"):
                    handle = restore(self._storage, resource)
                    handle._encoding_provider = self._encodings
                else:
                    # Resources published before selection receipts were introduced.
                    handle = self._execute_query(compile_query(resource))[1]
                with self._lock:
                    self._query_handles[id] = handle
        return handle

    def _resolve_dataset(
        self, request: datasets.CreateDatasetRequest, *, _lazy: bool = False
    ) -> datasets.CreateDatasetRequest:
        return datasets_execution.resolve_recipe(self, request, _lazy=_lazy)

    def _tokenizer_asset(
        self, asset: storage.ObjectRef, limit: int, inline: bytes | None = None
    ) -> execution.HuggingFaceTokenizer:
        return datasets_execution.load_tokenizer_asset(self, asset, limit, inline)

    def _tokenizer(
        self, spec: datasets.CreateDatasetRequest | datasets.CreateMixRequest | datasets.Dataset
    ) -> execution.HuggingFaceTokenizer | None:
        return datasets_execution.tokenizer(self, spec)

    def _mix_pool(
        self,
        spec: datasets.CreateDatasetRequest | datasets.CreateMixRequest | datasets.Dataset,
        strata: datasets.Domains,
    ) -> execution.MixturePool:
        return datasets_execution.mixture_pool(self, spec, strata)

    def _sampling(
        self, spec: datasets.CreateDatasetRequest | datasets.Dataset
    ) -> tuple[execution.MixturePool, tuple[dict[str, int], str, int, bool, int | None]]:
        return datasets_execution.sampling(self, spec)

    @staticmethod
    def _packing(spec: datasets.CreateDatasetRequest) -> tuple[int, int | None, int | None]:
        return datasets_execution.packing(spec)

    def _dataset_plan(self, spec: datasets.CreateDatasetRequest) -> DatasetPlan:
        return datasets_execution.plan(self, spec)

    def _dataset_id(self, spec: datasets.CreateDatasetRequest) -> bytes:
        return datasets_execution.identity(self, spec)

    def _packing_query(
        self, spec: datasets.CreateDatasetRequest | datasets.CreateMixRequest | datasets.Dataset
    ) -> execution.Query:
        return datasets_execution.packing_query(self, spec)

    def _dataset_handle(self, spec: datasets.CreateDatasetRequest) -> execution.Dataset:
        return datasets_execution.build(self, spec)

    def _profile_dataset(self, spec: datasets.CreateDatasetRequest) -> datasets.DatasetProfile:
        return datasets_execution.profile(self, spec)

    @_operation
    def CreateMix(self, request: datasets.CreateMixRequest) -> datasets.CreateMixResponse:
        """Pin and register the recipe without running its query or token inventory."""
        return mixtures_execution.create(self, request)

    def _resolve_mix(self, identity: bytes) -> datasets.Mix:
        """Freeze concrete candidate recipes; profiles and packed output stay lazy."""
        return mixtures_execution.resolve(self, identity)

    def _mix_profile(
        self,
        resource: datasets.Mix,
        inventory: Mapping[str, int],
        recipes: list[datasets.Dataset],
    ) -> datasets.MixProfile:
        return mixtures_execution.profile(self, resource, inventory, recipes)

    def _profile_mix(self, identity: bytes) -> datasets.Mix:
        return mixtures_execution.profile_resource(self, identity)

    @_operation
    def CreateDataset(
        self,
        request: datasets.CreateDatasetRequest,
        *,
        _lazy: bool = False,
        timeout: float | None = None,
    ) -> datasets.CreateDatasetResponse:
        return datasets_execution.create(self, request, _lazy=_lazy, timeout=timeout)

    def _planned_dataset_profile(self, resource: datasets.Dataset) -> datasets.DatasetProfile:
        return datasets_execution.planned_profile(self, resource)

    def _plan_dataset(self, request: datasets.CreateDatasetRequest) -> datasets.Dataset:
        """Register a recipe without executing its query, profiling, or packing."""
        return datasets_execution.plan_resource(self, request)

    @_read
    def GetDataset(self, request: datasets.GetDatasetRequest) -> datasets.GetDatasetResponse:
        return datasets_execution.get(self, request)


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
