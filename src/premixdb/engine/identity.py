"""Version-1 logical identity codec, independent of physical storage and protobuf."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Self

from blake3 import blake3

# Version-1 identities and snapshot headers use the project namespace.
IDENTITY_NAMESPACE = "premixdb"


def identity_domain(kind: str) -> bytes:
    return f"{IDENTITY_NAMESPACE}/{kind}/v1\0".encode()


def unsigned(value: object, bits: int = 64) -> int:
    if type(value) is not int or not 0 <= value < 2**bits:
        raise ValueError(f"expected an unsigned {bits}-bit integer")
    return value


def digest(value: str, size: int = 32) -> bytes:
    if not isinstance(value, str) or re.fullmatch(rf"[0-9a-fA-F]{{{size * 2}}}", value) is None:
        raise ValueError(f"expected {size * 2} hexadecimal digits")
    return bytes.fromhex(value)


class Canonical:
    def __init__(self, kind: str) -> None:
        self._hash = blake3(identity_domain(kind))

    def fixed(self, value: bytes) -> Self:
        self._hash.update(value)
        return self

    def u64(self, value: int) -> Self:
        return self.fixed(unsigned(value).to_bytes(8, "big"))

    def string(self, value: str) -> Self:
        data = value.encode("utf-8")
        return self.u64(len(data)).fixed(data)

    def finish(self) -> bytes:
        return self._hash.digest()


@dataclass(frozen=True)
class CodeVersion:
    repository: str
    commit: str
    environment: str

    def __post_init__(self) -> None:
        if (
            not self.repository.strip()
            or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.commit) is None
        ):
            raise ValueError("code requires a repository and full lowercase Git commit")
        object.__setattr__(self, "environment", digest(self.environment).hex())

    def canonical_digest(self) -> bytes:
        return (
            Canonical("code")
            .string(self.repository)
            .string(self.commit)
            .fixed(digest(self.environment))
            .finish()
        )

    def as_tuple(self) -> tuple[str, str, str]:
        return self.repository, self.commit, self.environment


def identity(domain: str, *parts: bytes) -> bytes:
    digest = blake3(domain.encode() + b"\0")
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.digest()
