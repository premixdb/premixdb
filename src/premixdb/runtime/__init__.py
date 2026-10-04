"""Recipe execution; read-only sessions need only the catalog."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from premixdb.v1 import query_pb2 as query

if TYPE_CHECKING:
    from premixdb.runtime.coordinator import Coordinator as Coordinator
    from premixdb.runtime.planner import compile_query as compile_query


def __getattr__(name: str) -> type[Coordinator] | Callable[[query.CreateQueryRequest], query.Query]:
    if name == "Coordinator":
        from premixdb.runtime.coordinator import Coordinator

        return Coordinator
    if name == "compile_query":
        from premixdb.runtime.planner import compile_query

        return compile_query
    raise AttributeError(name)


__all__ = ["Coordinator", "compile_query"]
