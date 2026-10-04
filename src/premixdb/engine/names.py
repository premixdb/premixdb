"""Stable name identities shared by readers and execution."""

from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5


def corpus_id(name: str) -> bytes:
    return uuid5(NAMESPACE_URL, f"premixdb:corpus/v1:{name}").bytes
