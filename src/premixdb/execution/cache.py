"""One shared, conservative memory budget for disposable execution caches."""

from __future__ import annotations

import sys
from collections import OrderedDict
from collections.abc import Hashable, ItemsView, MutableMapping, ValuesView
from threading import RLock
from typing import Iterator, cast

from google.protobuf.message import Message


def retained_bytes(value: object, seen: set[int] | None = None) -> int:
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, Message):
        return size + int(value.ByteSize())
    if isinstance(value, dict):
        return size + sum(
            retained_bytes(k, seen) + retained_bytes(v, seen) for k, v in list(value.items())
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return size + sum(retained_bytes(v, seen) for v in value)
    if type(value).__module__.startswith("premixdb.") and hasattr(value, "__dict__"):
        return size + retained_bytes(vars(value), seen)
    return size


class MemoryCache:
    """LRU budget shared across namespaces, independent of resource publication.

    Shared graphs are charged conservatively to each entry. This bounds estimated
    cached retention, not memory held by active jobs, Python allocators or callers.
    """

    def __init__(self, max_bytes: int = 128 * 1024 * 1024) -> None:
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("cache_bytes must be a nonnegative integer")
        self.max_bytes = max_bytes
        self.used_bytes = 0
        self._values: OrderedDict[tuple[str, Hashable], tuple[object, int]] = OrderedDict()
        self._lock = RLock()

    def namespace[K: Hashable, V](self, name: str) -> _Namespace[K, V]:
        return _Namespace(self, name)

    def _set(self, key: tuple[str, Hashable], value: object) -> None:
        size = retained_bytes(key) + retained_bytes(value)
        with self._lock:
            self._delete(key)
            if size > self.max_bytes:
                return
            while self.used_bytes + size > self.max_bytes:
                _, (_, removed) = self._values.popitem(last=False)
                self.used_bytes -= removed
            self._values[key] = value, size
            self.used_bytes += size

    def _delete(self, key: tuple[str, Hashable]) -> None:
        old = self._values.pop(key, None)
        if old is not None:
            self.used_bytes -= old[1]


class _Namespace[K: Hashable, V](MutableMapping[K, V]):
    def __init__(self, cache: MemoryCache, name: str) -> None:
        self._cache, self._name = cache, name

    def __getitem__(self, key: K) -> V:
        with self._cache._lock:
            composite = self._name, key
            value, _ = self._cache._values[composite]
            self._cache._values.move_to_end(composite)
            return cast(V, value)  # Each namespace has one declared key/value contract.

    def __setitem__(self, key: K, value: V) -> None:
        self._cache._set((self._name, key), value)

    def __delitem__(self, key: K) -> None:
        with self._cache._lock:
            if (self._name, key) not in self._cache._values:
                raise KeyError(key)
            self._cache._delete((self._name, key))

    def __iter__(self) -> Iterator[K]:
        with self._cache._lock:
            return iter([cast(K, key) for name, key in self._cache._values if name == self._name])

    def __len__(self) -> int:
        with self._cache._lock:
            return sum(name == self._name for name, _ in self._cache._values)

    def _snapshot(self) -> dict[K, V]:
        # Views keep their membership while another namespace triggers eviction.
        with self._cache._lock:
            return {
                cast(K, key): cast(V, value)
                for (name, key), (value, _) in self._cache._values.items()
                if name == self._name
            }

    def items(self) -> ItemsView[K, V]:
        return self._snapshot().items()

    def values(self) -> ValuesView[V]:
        return self._snapshot().values()

    def refresh(self, key: K) -> None:
        try:
            value = self[key]
        except KeyError:
            return
        self[key] = value
