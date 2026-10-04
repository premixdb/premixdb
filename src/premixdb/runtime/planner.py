"""Validate and resolve public queries before running the Python engine."""

from __future__ import annotations

import math

from blake3 import blake3
from google.protobuf.message import Message

from premixdb.engine import plans as execution
from premixdb.engine.identity import identity_domain
from premixdb.runtime import environment as _runtime
from premixdb.schemas.messages import copy_fields, reject_unknown
from premixdb.schemas.protobuf import copy_message
from premixdb.v1 import query_pb2 as query
from premixdb.v1 import status_pb2 as status

FIELDS = {
    query.FIELD_TEXT_BYTES: "bytes",
    query.FIELD_TEXT_CHARACTERS: "characters",
    query.FIELD_OBJECT_URI: "object_uri",
}
COMPARISONS = {
    query.Comparison.OPERATOR_EQ: "eq",
    query.Comparison.OPERATOR_NE: "ne",
    query.Comparison.OPERATOR_LT: "lt",
    query.Comparison.OPERATOR_LE: "le",
    query.Comparison.OPERATOR_GT: "gt",
    query.Comparison.OPERATOR_GE: "ge",
}


def field(value: query.IntrinsicField) -> str:
    name = FIELDS.get(value)
    if name is None:
        raise ValueError(f"unsupported field: {value}")
    return name


def execution_steps(plan: query.Query | query.CreateQueryRequest) -> list[execution.Step]:
    """Validate the entire plan. Unsupported semantics never become no-ops."""
    reject_unknown(plan)
    steps = []
    for operation in plan.operations:
        kind = operation.WhichOneof("kind")
        if kind == "document_ids":
            steps.append(execution.filter_documents(operation.document_ids.ids))
        elif kind == "where":
            comparison = operation.where
            name = field(comparison.field)
            op = COMPARISONS.get(comparison.operator)
            if op is None:
                raise ValueError("invalid comparison operator")
            expected = "text" if name == "object_uri" else "count"
            if comparison.WhichOneof("value") != expected:
                raise ValueError("comparison value does not match the field type")
            steps.append(execution.filter(name, op, getattr(comparison, expected)))
        elif kind == "dedupe":
            policy = operation.dedupe
            units = {
                query.Dedupe.ALGORITHM_EXACT_DOCUMENT: "document",
                query.Dedupe.ALGORITHM_EXACT_LINE: "line",
            }
            if policy.algorithm not in units:
                raise ValueError("invalid exact comparison unit")
            orders = _orders(policy)
            separator = (
                policy.source_group_separator if policy.HasField("source_group_separator") else None
            )
            steps.append(execution.dedupe(orders, units[policy.algorithm], separator))
        elif kind == "field_where":
            selector = operation.field_where
            if len(selector.field_snapshot_id) != 32:
                raise ValueError("external field selector requires a resolved build ID")
            if selector.operator not in COMPARISONS or selector.WhichOneof("value") is None:
                raise ValueError("external comparison requires an operator and value")
            if selector.WhichOneof("value") == "number" and not math.isfinite(selector.number):
                raise ValueError("comparison must be finite")
            if selector.projection not in (
                query.FieldComparison.SCALAR,
                query.FieldComparison.IS_NULL,
                query.FieldComparison.TOP_CLASS,
                query.FieldComparison.CLASS_PROBABILITY,
                query.FieldComparison.VECTOR_COMPONENT,
            ):
                raise ValueError("unsupported field projection")
            steps.append(execution.external_filter(external_definition(operation)))
        elif kind == "indexed_dedupe":
            policy = operation.indexed_dedupe
            if len(policy.index_snapshot_id) != 32:
                raise ValueError("indexed dedupe requires a build ID")
            orders = _orders(policy)
            if policy.index_name == "dupekit.lsh":
                if (
                    not policy.HasField("threshold")
                    or not math.isfinite(policy.threshold)
                    or not 0 <= policy.threshold <= 1
                ):
                    raise ValueError("LSH dedupe requires a finite verification threshold in [0,1]")
                steps.append(
                    execution.policy(policy_definition(operation), ("similarity", orders, None))
                )
            else:
                if policy.HasField("threshold"):
                    raise ValueError("threshold applies only to approximate dedupe")
                steps.append(execution.external_dedupe(external_definition(operation), orders))
        elif kind == "similarity_dedupe":
            policy = operation.similarity_dedupe
            if (
                policy.algorithm
                not in (query.SimilarityDedupe.JACCARD, query.SimilarityDedupe.COSINE)
                or not math.isfinite(policy.threshold)
                or not 0 <= policy.threshold <= 1
            ):
                raise ValueError("similarity requires a valid algorithm and threshold in [0,1]")
            if policy.algorithm == query.SimilarityDedupe.JACCARD and not 1 <= policy.n <= 1024:
                raise ValueError("Jaccard n-gram size must be in [1,1024]")
            if (
                policy.algorithm == query.SimilarityDedupe.COSINE
                and policy.embedding.field != query.FIELD_EMBEDDING_HARRIER
            ):
                raise ValueError("cosine dedupe requires a pinned embedding field")
            orders = _orders(policy)
            steps.append(
                execution.policy(policy_definition(operation), ("similarity", orders, None))
            )
        else:
            raise NotImplementedError(f"unsupported query operation: {kind}")
    if plan.HasField("decontaminate"):
        policy = plan.decontaminate
        references = sorted(set(policy.snapshot_ids))
        if not references or any(len(id) != 32 for id in references):
            raise ValueError("decontamination requires immutable reference snapshots")
        if policy.algorithm not in (1, 2, 3) or policy.granularity not in (1, 2):
            raise ValueError("invalid decontamination algorithm or granularity")
        if policy.algorithm == 3 and not 1 <= policy.n <= 1024:
            raise ValueError("n-gram size must be in [1,1024]")
        if policy.algorithm != 3 and policy.n:
            raise ValueError("n is only valid for n-gram decontamination")
        steps.append(execution.policy(policy_definition(policy), ("decontaminate", policy, ())))
    if plan.HasField("sampling"):
        policy = plan.sampling
        budget = policy.WhichOneof("budget")
        if budget is None or not policy.HasField("seed"):
            raise ValueError("sampling requires a budget and explicit seed")
        if budget == "fraction" and (
            not math.isfinite(policy.fraction) or not 0 <= policy.fraction <= 1
        ):
            raise ValueError("fraction must be finite and in [0,1]")
        if policy.weights and (
            any(not math.isfinite(w) or w < 0 for w in policy.weights.values())
            or not math.isclose(math.fsum(policy.weights.values()), 1, abs_tol=1e-12)
        ):
            raise ValueError("sampling weights must be finite, nonnegative and sum to one")
        if policy.HasField("tokenizer_asset") and (
            budget != "tokens" or len(policy.tokenizer_asset.blake3_digest) != 32
        ):
            raise ValueError("tokenizer assets require a token budget and digest")
        if policy.tokenizer_json:
            if (
                not policy.HasField("tokenizer_asset")
                or blake3(policy.tokenizer_json).digest() != policy.tokenizer_asset.blake3_digest
            ):
                raise ValueError("sampling tokenizer JSON differs from its digest")
        steps.append(execution.policy(policy_definition(policy), ("sample", policy, None)))
    return steps


def compile_query(plan: query.CreateQueryRequest | query.Query) -> query.Query:
    """Resolve a public query and its identity without reading input documents.

    Scheduler, queue, retry count, and machine count do not affect resource identity.
    This stage consumes snapshot references; no documents enter the control plane.
    """
    reject_unknown(plan)
    resolved = query_request(plan)
    if not resolved.snapshot_ids or any(len(value) != 32 for value in resolved.snapshot_ids):
        raise ValueError("queries require 32-byte snapshot IDs")
    for name in ("snapshot_ids", "field_snapshot_ids", "index_snapshot_ids"):
        values = sorted(set(getattr(resolved, name)))
        if any(len(value) != 32 for value in values):
            raise ValueError("query inputs require 32-byte IDs")
        del getattr(resolved, name)[:]
        getattr(resolved, name).extend(values)
    if resolved.HasField("decontaminate"):
        policy = resolved.decontaminate
        references = sorted(set(policy.snapshot_ids))
        del policy.snapshot_ids[:]
        policy.snapshot_ids.extend(references)
        if not policy.granularity:
            policy.granularity = query.Decontaminate.DOCUMENT
    code = _runtime.resolve_code(resolved.git_commit)
    resolved.git_commit = bytes.fromhex(code.commit)
    from premixdb.runtime.catalog import resolve, validate_logical_selector

    resolve(resolved)
    projected = {}
    for selector in resolved.fields:
        if selector.operator or selector.WhichOneof("value") is not None:
            raise ValueError("query fields must be projections without comparison values")
        projected[selector.SerializeToString(deterministic=True)] = selector
    del resolved.fields[:]
    resolved.fields.extend(projected[key] for key in sorted(projected))
    for op in resolved.operations:
        if op.HasField("document_ids"):
            ids = sorted(set(op.document_ids.ids))
            if any(len(id) != 32 for id in ids):
                raise ValueError("document selection requires 32-byte IDs")
            del op.document_ids.ids[:]
            op.document_ids.ids.extend(ids)
        if op.HasField("field_where"):
            validate_logical_selector(op.field_where)
    from premixdb.runtime.catalog import selectors

    for selector in selectors(resolved):
        if selector.WhichOneof("value") is None:
            validate_projection(
                selector,
                vector=selector.field == query.FIELD_EMBEDDING_HARRIER
                and selector.projection == query.FieldComparison.SCALAR
                and (
                    selector in resolved.fields
                    or any(
                        op.HasField("similarity_dedupe")
                        and op.similarity_dedupe.embedding == selector
                        for op in resolved.operations
                    )
                ),
            )
    steps = execution_steps(resolved)
    resource_id = bytes.fromhex(
        execution.query_identity(
            [value.hex() for value in resolved.snapshot_ids],
            steps,
            code,
            field_definitions(resolved),
        )
    )
    result = query.Query(id=resource_id, status=status.STATUS_PENDING)
    copy_fields(resolved, result)
    return result


def field_definitions(plan: query.Query | query.CreateQueryRequest) -> tuple[bytes, ...]:
    return tuple(policy_definition(selector) for selector in plan.fields)


def query_request(resource: query.Query | query.CreateQueryRequest) -> query.CreateQueryRequest:
    return copy_fields(resource, query.CreateQueryRequest())


def external_definition(operation: query.Operation) -> bytes:
    return blake3(
        identity_domain("external-operation") + operation.SerializeToString(deterministic=True)
    ).digest()


def _orders(
    policy: query.Dedupe | query.IndexedDedupe | query.SimilarityDedupe,
) -> list[tuple[str, bool]]:
    orders = []
    for order in policy.order_by:
        if order.direction not in (query.OrderBy.DIRECTION_ASC, query.OrderBy.DIRECTION_DESC):
            raise ValueError("invalid ordering direction")
        orders.append((order_field(order), order.direction == query.OrderBy.DIRECTION_DESC))
    return orders


def order_field(order: query.OrderBy) -> str:
    if order.HasField("selector"):
        if order.field:
            raise ValueError("ordering requires one field selector")
        from premixdb.engine.curation import selector_key

        validate_projection(order.selector)
        return selector_key(order.selector)
    return field(order.field)


def policy_definition(message: Message) -> bytes:
    from premixdb.runtime.mixing import canonical_digest

    logical = copy_message(message)
    if isinstance(logical, query.QuerySampling) and logical.HasField("tokenizer_asset"):
        logical.ClearField("tokenizer_json")
        for name in ("uri", "size_bytes", "profile"):
            logical.tokenizer_asset.ClearField(name)
    return canonical_digest("query-policy-v1", logical)


def validate_projection(selector: query.FieldComparison, *, vector: bool = False) -> None:
    if selector.field in (1, 2, 3, 4):
        if (
            selector.projection != query.FieldComparison.SCALAR
            or selector.class_name
            or selector.HasField("component")
        ):
            raise ValueError("intrinsic strata require scalar projections")
        return
    if vector:
        if (
            selector.projection != query.FieldComparison.SCALAR
            or selector.class_name
            or selector.HasField("component")
        ):
            raise ValueError("cosine similarity requires the complete embedding")
        return
    from premixdb.fields import ContentType, Topic
    from premixdb.runtime.catalog import validate_logical_selector

    probe = copy_message(selector)
    probe.operator = query.Comparison.OPERATOR_EQ
    if probe.projection == query.FieldComparison.TOP_CLASS:
        probe.text = next(
            iter(Topic if probe.field == query.FIELD_WEBORGANIZER_TOPIC else ContentType)
        ).value
    elif probe.projection == query.FieldComparison.IS_NULL:
        probe.boolean = True
    elif probe.field in (query.FIELD_DATATROVE_LENGTH, query.FIELD_DATATROVE_N_WORDS):
        probe.integer = 0
    elif probe.field == query.FIELD_LANGUAGE_LABEL:
        probe.text = "en"
    else:
        probe.number = 0
    validate_logical_selector(probe)
