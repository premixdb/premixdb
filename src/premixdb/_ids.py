"""URL-safe public identities; storage and protobufs keep their original bytes."""

from __future__ import annotations

import base64
import binascii
import re

from ._typing import JSON
from .v1.dataset_pb2 import DatasetProfile


def _encode_id(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode_id(value: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError("ID must be base64url or hexadecimal")
    if len(value) in (32, 64) and re.fullmatch(r"[0-9a-fA-F]+", value):
        return bytes.fromhex(value)
    if not re.fullmatch(r"[A-Za-z0-9_-]+={0,2}", value):
        raise ValueError("ID must be base64url or hexadecimal")
    unpadded = value.rstrip("=")
    try:
        decoded = base64.b64decode(
            unpadded + "=" * (-len(unpadded) % 4), altchars=b"-_", validate=True
        )
    except (binascii.Error, ValueError) as exc:
        raise ValueError("ID must be base64url or hexadecimal") from exc
    if _encode_id(decoded) != unpadded:
        raise ValueError("ID must use canonical base64url encoding")
    return decoded


def _public_lineage(value: JSON, *, field: str | None = None) -> JSON:
    if isinstance(value, dict):
        return {
            _encode_id(bytes.fromhex(key))
            if re.fullmatch(r"[0-9a-f]{64}", key)
            else key: _public_lineage(item, field=key)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_public_lineage(item, field=field) for item in value]
    if (
        isinstance(value, str)
        and field
        in (
            "id",
            "corpus_id",
            "document_id",
            "snapshot_id",
            "query_id",
            "dataset_id",
            "snapshots",
            "reference_snapshot_ids",
            "kept",
        )
        and re.fullmatch(r"[0-9a-f]{32}|[0-9a-f]{64}", value)
    ):
        return _encode_id(bytes.fromhex(value))
    return value


def _public_dataset_profile(
    profile: DatasetProfile, *, corpus_strata: bool = False
) -> DatasetProfile:
    for name in (
        "source_tokens",
        *(("planned_stratum_tokens", "stratum_tokens") if corpus_strata else ()),
    ):
        values = getattr(profile, name)
        encoded = {_encode_id(_decode_id(key)): count for key, count in values.items()}
        values.clear()
        values.update(encoded)
    return profile
