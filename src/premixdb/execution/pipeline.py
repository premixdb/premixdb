"""Lower tokenization and exact evidence to verified map partitions and disk exchange.

One reusable process pool computes partitions using local immutable files.
The coordinator reconciles coverage and publishes the final public resource.
"""

from __future__ import annotations

import base64
import json
import platform
import time
from bisect import bisect_left, bisect_right
from collections import deque
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from contextlib import closing
from itertools import islice
from multiprocessing import get_context
from pathlib import Path
from typing import Callable, ContextManager, Generator, Iterable, Iterator, Mapping
from typing import Sequence as SequenceABC
from urllib.parse import unquote, urlsplit

from blake3 import blake3

from .. import _runtime
from .._typing import (
    JSON,
    checked_record,
    field_value,
    json_list,
    json_object,
    load_json,
)
from ..engine.contracts import Occurrence, Span
from ..engine.curation import RetainedDocument, SelectedDocument, units
from ..engine.datasets import (
    ByteTokens,
    Dataset,
    Sequence,
    TokenList,
    compact_ranges,
    encoded_tokens,
)
from ..engine.queries import CorpusIndex, Query, Row
from ..engine.snapshots import Document
from ..engine.spill import Group, ReferenceLookup, evidence_groups
from ..engine.token_codec import byte_interval, decode_tokens, encode_tokens, token_length
from ..enrichment.types import ComputedRow
from ..enrichment.types import Document as FeatureDocument
from ..internal import derivation_pb2 as e
from ..v1 import dataset_pb2 as d
from ..v1 import query_pb2 as q
from .enrichment import DedupeRow, Worker
from .partition_types import (
    EvidenceRequest,
    EvidenceRow,
    FeatureRequest,
    IndexRow,
    InputRow,
    PackedRow,
    PackingRequest,
    TokenRequest,
    TokenRow,
)
from .partitions import (
    Artifact,
    Kernel,
    PartitionStore,
    PartitionTask,
    PartitionWorker,
    Receipt,
    checked_receipt,
)

PARTITION_BYTES = 8 * 1024 * 1024
_FEATURE_WORKERS: dict[bytes, Worker] = {}


def execute_tasks(
    pool: Executor,
    run: Callable[[PartitionTask], Receipt],
    tasks: Iterable[PartitionTask],
    window: int,
) -> Generator[Receipt, None, None]:
    """Bound queued futures while reconciling outputs in admitted input order."""
    pending: deque[tuple[PartitionTask, Future[Receipt]]] = deque()
    tasks = iter(tasks)
    try:
        for task in islice(tasks, window):
            pending.append((task, pool.submit(run, task)))
        while pending:
            task, future = pending.popleft()
            yield checked_receipt(task, future.result())
            next_task = next(tasks, None)
            if next_task is not None:
                pending.append((next_task, pool.submit(run, next_task)))
    finally:
        for _, future in pending:
            future.cancel()


def engine_digest() -> bytes:
    root = Path(__file__).resolve().parents[1]
    digest = blake3(b"premixdb-partition-engine/v1\0")
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    digest.update(platform.python_version().encode())
    digest.update(_runtime.current_code().canonical_digest())
    return digest.digest()


def _packing_blocks(prefixes: SequenceABC[int], total: int, low: int, high: int) -> range:
    """Locate block indices by their cumulative token boundaries."""
    if not prefixes or low >= total:
        return range(0)
    return range(max(0, bisect_right(prefixes, low) - 1), bisect_left(prefixes, high))


def processor(task: PartitionTask, inputs: tuple[Path, ...], output: Path) -> None:
    if not inputs or (
        task.kernel != Kernel.PACK
        and len(inputs) not in ((1, 2) if task.kernel == Kernel.TOKENIZE else (1,))
    ):
        raise ValueError("invalid partition inputs")
    raw = json_object(load_json(inputs[0].read_bytes()))
    if raw.get("version") != 1:
        raise ValueError("unsupported partition request")
    if task.kernel == Kernel.FEATURES:
        from ..internal import derivation_pb2 as e
        from .enrichment import producer

        feature_request = checked_record(raw, FeatureRequest)
        encoded = base64.b64decode(feature_request["producer"])
        worker = _FEATURE_WORKERS.get(encoded)
        if worker is None:
            policy = e.EnrichmentProducer()
            policy.ParseFromString(encoded)
            worker = producer(policy)
            _FEATURE_WORKERS[encoded] = worker
        definition = json.dumps(
            worker.definition, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if definition != feature_request["definition"]:
            raise ValueError("partition producer differs from its admitted definition")
        documents = [
            FeatureDocument(r["id"], r["text"], r.get("url")) for r in feature_request["rows"]
        ]
        from ..enrichment.dupekit import DupekitIndex

        if isinstance(worker, DupekitIndex):
            index_values = [
                checked_record(row, DedupeRow) for row in worker.compute(documents).to_pylist()
            ]
            wire_values: list[IndexRow] = [
                dict(
                    id=row["id"],
                    exact_hash=base64.b64encode(row["exact_hash"]).decode(),
                    minhash=row["minhash"],
                    lsh_buckets=row["lsh_buckets"],
                )
                for row in index_values
            ]
            result = dict(version=1, rows=wire_values)
        else:
            result = dict(version=1, rows=worker.compute(documents))
    elif task.kernel == Kernel.PACK:
        request = checked_record(raw, PackingRequest)
        first, length = request["first"], request["length"]
        low, high = first * length, (first + request["sequences"]) * length
        packed: dict[int, PackedRow] = {}

        def packed_sequence(ordinal: int) -> PackedRow:
            sequence = packed.get(ordinal)
            if sequence is None:
                sequence = packed[ordinal] = PackedRow(
                    ordinal=ordinal, tokens=[], spans=[], alignment=[]
                )
            return sequence

        for path, prefix in zip(inputs[1:], request["prefixes"], strict=True):
            content = json_object(load_json(path.read_bytes()))
            cursor = prefix
            for raw_row in json_list(content["rows"]):
                row = checked_record(raw_row, TokenRow)
                decoded = decode_tokens(row)
                token_count = len(decoded)
                for start, end, kind in (
                    (cursor, cursor + token_count, "content"),
                    (
                        cursor + token_count,
                        cursor + token_count + (request["separator"] is not None),
                        "separator",
                    ),
                ):
                    left, right = max(start, low), min(end, high)
                    while left < right:
                        sequence_ordinal = left // length
                        begin = left % length
                        take = min(right - left, length - begin)
                        sequence = packed_sequence(sequence_ordinal)
                        offset = left - start
                        if kind == "content":
                            tokens = decoded[offset : offset + take]
                            sequence["alignment"].extend(
                                dict(token=begin + i, occurrence=row["ordinal"], start=a, end=b)
                                for i, ranges in enumerate(tokens.ranges)
                                for a, b in ranges
                            )
                        else:
                            separator = request["separator"]
                            assert separator is not None
                            tokens = [separator]
                        span: Span = Span(
                            start=begin,
                            end=begin + take,
                            kind="content" if kind == "content" else "separator",
                            occurrence=row["ordinal"],
                        )
                        if kind == "content":
                            span["offset"] = offset
                        sequence["spans"].append(span)
                        sequence["tokens"].extend(tokens)
                        left += take
                cursor += token_count + (request["separator"] is not None)
        for ordinal in range(first, first + request["sequences"]):
            sequence = packed_sequence(ordinal)
            missing = length - len(sequence["tokens"])
            if missing:
                if request["padding"] is None or ordinal != request["total_sequences"] - 1:
                    raise ValueError("packing partition omitted tokens")
                sequence["spans"].append(
                    dict(start=len(sequence["tokens"]), end=length, kind="padding")
                )
                sequence["tokens"].extend([request["padding"]] * missing)
        result = dict(version=1, rows=[packed[i] for i in sorted(packed)])
    elif task.kernel == Kernel.TOKENIZE:
        token_request = checked_record(raw, TokenRequest)
        tokenizer = None
        policy = d.Tokenizer()
        policy.ParseFromString(base64.b64decode(token_request["tokenizer"]))
        if policy.HasField("hugging_face"):
            from ..engine.datasets import HuggingFaceTokenizer
            from .assets import read

            asset = policy.hugging_face
            data = read(
                asset.asset,
                inline=inputs[1].read_bytes() if len(inputs) == 2 else asset.json or None,
            )
            tokenizer = HuggingFaceTokenizer.from_bytes(
                data, asset.asset.blake3_digest.hex(), asset.max_document_bytes
            )
        token_rows: list[TokenRow] = []
        for row in token_request["rows"]:
            document = Document("0" * 32, row["id"], row["text"])
            ranges = tuple(byte_interval(pair) for pair in row["ranges"])
            retained = RetainedDocument(document, row["text"], ranges)
            tokens = encoded_tokens(Row(row["ordinal"], retained), tokenizer)
            token_rows.append(dict(id=row["id"], ordinal=row["ordinal"], **encode_tokens(tokens)))
        result = dict(version=1, rows=token_rows)
    elif task.kernel in (Kernel.DEDUPE_INDEX, Kernel.DECONTAMINATION_INDEX):
        from ..engine.spill import unit_key

        evidence_request = checked_record(raw, EvidenceRequest)
        evidence_rows: list[EvidenceRow] = []
        for row in evidence_request["rows"]:
            for value, start, end in units(
                Document("0" * 32, row["id"], row["text"]),
                evidence_request["algorithm"],
                evidence_request.get("n", 0),
            ):
                evidence_rows.append(
                    dict(
                        id=row["id"],
                        start=start,
                        end=end,
                        value=base64.b64encode(unit_key(value)).decode(),
                    )
                )
        result = dict(version=1, rows=evidence_rows)
    else:
        raise ValueError("kernel is not admitted by this worker")
    output.write_text(json.dumps(result, separators=(",", ":")))


_PROCESS_WORKER: PartitionWorker | None = None


def _initialize_worker(root: str) -> None:
    global _PROCESS_WORKER
    _PROCESS_WORKER = PartitionWorker(PartitionStore(root), engine_digest(), processor)


def _process(task: PartitionTask) -> Receipt:
    if _PROCESS_WORKER is None:
        raise RuntimeError("partition worker was not initialized")
    return _PROCESS_WORKER(task)


class PartitionPipeline:
    def __init__(self, storage: PartitionStore, *, workers: int = 1) -> None:
        if type(workers) is not int or workers <= 0:
            raise ValueError("workers must be a positive integer")
        self.storage = storage
        self._window = workers * 2
        self.engine = engine_digest()
        self._pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(unquote(urlsplit(storage.root).path),),
        )
        self.packing_shard_sequences = 128

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def _execute(self, tasks: Iterable[PartitionTask]) -> tuple[Artifact, ...]:
        """Verify ordered results and cancel queued work when reconciliation fails."""
        outputs = []
        with closing(execute_tasks(self._pool, _process, tasks, self._window)) as results:
            for receipt in results:
                self.storage.verify(receipt.output)
                outputs.append(receipt.output)
        return tuple(outputs)

    def _task(self, key: bytes, kernel: Kernel, inputs: tuple[Artifact, ...]) -> PartitionTask:
        return PartitionTask(
            key, self.engine, kernel, inputs, self.storage.root + "/outputs/" + key.hex()
        )

    def _input(self, data: bytes) -> Artifact:
        return self.storage.publish_bytes(
            self.storage.root + "/inputs/" + blake3(data).hexdigest(), data
        )

    def _map(
        self,
        kernel: Kernel,
        documents: Iterable[Row | SelectedDocument],
        options: Mapping[str, JSON],
        additional_inputs: tuple[Artifact, ...] = (),
    ) -> tuple[Artifact, ...]:
        tasks: list[tuple[bytes, Artifact]] = []
        batch: list[InputRow] = []
        size = 0

        def flush() -> None:
            nonlocal size
            payload: dict[str, object] = dict(version=1, rows=batch)
            payload.update(options)
            data = json.dumps(payload, separators=(",", ":")).encode()
            artifact = self._input(data)
            key = blake3(
                b"premixdb-map/v1\0"
                + self.engine
                + kernel.value.encode()
                + artifact.digest
                + b"".join(item.digest for item in additional_inputs)
            ).digest()
            tasks.append((key, artifact))
            batch.clear()
            size = 0

        for document in documents:
            original = document.document if isinstance(document, Row) else document
            text = document.text
            ranges = (
                original.source_ranges
                if isinstance(original, RetainedDocument)
                else ((0, len(text.encode())),)
            )
            row: InputRow = InputRow(
                id=document.id,
                text=text,
                ordinal=document.ordinal if isinstance(document, Row) else 0,
                ranges=[[a, b] for a, b in ranges],
            )
            amount = len(json.dumps(row).encode())
            if batch and size + amount > PARTITION_BYTES:
                flush()
            batch.append(row)
            size += amount
        if batch:
            flush()
        return self._execute(
            self._task(
                blake3(key + i.to_bytes(8, "big") + len(tasks).to_bytes(8, "big")).digest(),
                kernel,
                (artifact, *additional_inputs),
            )
            for i, (key, artifact) in enumerate(tasks)
        )

    def rows(self, artifacts: Iterable[Artifact]) -> Iterator[dict[str, JSON]]:
        for artifact in artifacts:
            result = json_object(load_json(self.storage.read(artifact)))
            if result.get("version") != 1:
                raise ValueError("unsupported partition output")
            yield from (json_object(row) for row in json_list(result["rows"]))

    def features(
        self,
        policy: e.EnrichmentProducer,
        definition: bytes,
        cohorts: Iterable[SequenceABC[FeatureDocument]],
    ) -> Iterator[list[ComputedRow] | list[DedupeRow]]:
        """Dispatch fixed computation cohorts; physical layout never changes batching."""
        producer = base64.b64encode(policy.SerializeToString(deterministic=True)).decode()
        definition_text = definition.decode()
        inputs = []
        for cohort in cohorts:
            request = dict(
                version=1,
                producer=producer,
                definition=definition_text,
                rows=[dict(id=doc.id, text=doc.text, url=doc.url) for doc in cohort],
            )
            inputs.append(self._input(json.dumps(request, separators=(",", ":")).encode()))
        tasks = []
        for i, artifact in enumerate(inputs):
            key = blake3(
                b"premixdb-features/v1\0"
                + self.engine
                + artifact.digest
                + i.to_bytes(8, "big")
                + len(inputs).to_bytes(8, "big")
            ).digest()
            tasks.append(self._task(key, Kernel.FEATURES, (artifact,)))
        for artifact in self._execute(tasks):
            raw_rows = list(self.rows((artifact,)))
            if policy.HasField("dupekit"):
                indexes = [checked_record(row, IndexRow) for row in raw_rows]
                yield [
                    dict(
                        id=row["id"],
                        exact_hash=base64.b64decode(row["exact_hash"], validate=True),
                        minhash=row["minhash"],
                        lsh_buckets=row["lsh_buckets"],
                    )
                    for row in indexes
                ]
            else:
                yield [{key: field_value(value) for key, value in row.items()} for row in raw_rows]

    def tokenize(
        self, query: Query, tokenizer: d.Tokenizer
    ) -> tuple[list[int], Iterator[tuple[Row, ByteTokens | TokenList]]]:
        additional = ()
        if tokenizer.HasField("hugging_face"):
            from urllib.parse import unquote, urlsplit

            from .assets import read

            policy = tokenizer.hugging_face
            location = urlsplit(policy.asset.uri)
            local_root = Path(unquote(location.path)).parent if location.scheme == "file" else None
            data = read(policy.asset, local_root=local_root, inline=policy.json or None)
            additional = (self._input(data),)
        artifacts = self._map(
            Kernel.TOKENIZE,
            query,
            dict(
                tokenizer=base64.b64encode(tokenizer.SerializeToString(deterministic=True)).decode()
            ),
            additional,
        )
        lengths: list[int] = []
        expected = iter(query)
        for raw_row in self.rows(artifacts):
            row = checked_record(raw_row, TokenRow)
            original = next(expected, None)
            if original is None or row["ordinal"] != original.ordinal or row["id"] != original.id:
                raise ValueError("token partition does not cover ordered query occurrences")
            lengths.append(token_length(row))
        if next(expected, None) is not None:
            raise ValueError("token partition omitted query occurrences")

        def encoded() -> Iterator[tuple[Row, ByteTokens | TokenList]]:
            for original, row in zip(query, self.rows(artifacts), strict=True):
                yield original, decode_tokens(checked_record(row, TokenRow))

        return lengths, encoded()

    def pack(self, handle: Dataset) -> PackedPartitions:
        packing = handle.plan.packing
        blocks: list[Artifact] = []
        prefixes: list[int] = []
        prefix, size = 0, 0
        current: list[TokenRow] = []
        occurrences: list[Occurrence] = []

        def flush() -> None:
            nonlocal size
            data = json.dumps(dict(version=1, rows=current), separators=(",", ":")).encode()
            blocks.append(self._input(data))
            current.clear()
            size = 0

        for ordinal, (row, tokens) in enumerate(handle._encoded):
            if ordinal >= len(handle._lengths) or len(tokens) != handle._lengths[ordinal]:
                raise ValueError("tokenization does not match packing lengths")
            if current and size + len(tokens) * 16 > PARTITION_BYTES:
                flush()
            if not current:
                prefixes.append(prefix)
            current.append(dict(id=row.id, ordinal=ordinal, **encode_tokens(tokens)))
            occurrences.append(
                dict(
                    ordinal=ordinal,
                    document=row.id,
                    source=handle._query._provenance[row.id],
                    tokens=len(tokens),
                )
            )
            prefix += len(tokens) + (packing.separator is not None)
            size += len(tokens) * 16
        if current:
            flush()
        if len(occurrences) != len(handle._lengths):
            raise ValueError("incomplete tokenization coverage")
        tasks = []
        totals = handle._totals
        for first in range(0, totals.sequences, self.packing_shard_sequences):
            sequences = min(self.packing_shard_sequences, totals.sequences - first)
            low, high = first * packing.length, (first + sequences) * packing.length
            selected = _packing_blocks(prefixes, prefix, low, high)
            data = json.dumps(
                dict(
                    version=1,
                    first=first,
                    sequences=sequences,
                    total_sequences=totals.sequences,
                    length=packing.length,
                    separator=packing.separator,
                    padding=packing.padding,
                    prefixes=[prefixes[j] for j in selected],
                ),
                separators=(",", ":"),
            ).encode()
            control = self._input(data)
            inputs = (control, *(blocks[j] for j in selected))
            key = blake3(
                b"premixdb-pack/v1\0" + self.engine + b"".join(a.digest for a in inputs)
            ).digest()
            tasks.append(self._task(key, Kernel.PACK, inputs))
        return PackedPartitions(handle, self, self._execute(tasks), occurrences)

    def exact_classes(self, index: CorpusIndex, unit: str) -> Generator[Group, None, None]:
        artifacts = self._map(
            Kernel.DEDUPE_INDEX,
            index.documents.values(),
            dict(algorithm=1 if unit == "Document" else 2),
        )
        yield from evidence_groups(
            (base64.b64decode(row["value"]), row["id"], row["start"], row["end"])
            for raw_row in self.rows(artifacts)
            for row in [checked_record(raw_row, EvidenceRow)]
        )

    def references(
        self, documents: SequenceABC[Document], policy: q.Decontaminate
    ) -> ContextManager[ReferenceLookup]:
        from ..engine.spill import reference_rows

        artifacts = self._map(
            Kernel.DECONTAMINATION_INDEX, documents, dict(algorithm=policy.algorithm, n=policy.n)
        )
        return reference_rows(
            (base64.b64decode(row["value"]), row["id"], row["start"], row["end"])
            for raw_row in self.rows(artifacts)
            for row in [checked_record(raw_row, EvidenceRow)]
        )


class PackedPartitions(Dataset):
    def __init__(
        self,
        handle: Dataset,
        pipeline: PartitionPipeline,
        outputs: SequenceABC[Artifact],
        occurrences: list[Occurrence],
    ) -> None:
        super().__init__(
            handle._query,
            handle.plan,
            handle._input_counts,
            (),
            time.monotonic(),
            handle._lengths,
            stream=True,
        )
        self.elapsed_seconds = handle.elapsed_seconds
        handle.close()
        self.pipeline, self.outputs = pipeline, outputs
        self._occurrences = occurrences

    def _pack_sequences(self) -> Iterator[Sequence]:
        if self._consumed:
            raise RuntimeError("packing stream has already been consumed")
        self._consumed = True
        expected = 0
        for raw_row in self.pipeline.rows(self.outputs):
            row = checked_record(raw_row, PackedRow)
            if row["ordinal"] != expected or len(row["tokens"]) != self.plan.packing.length:
                raise ValueError("packing output does not cover sequence ordinals")
            yield Sequence(expected, row["tokens"], row["spans"], compact_ranges(row["alignment"]))
            expected += 1
        if expected != len(self):
            raise ValueError("incomplete packed output")
