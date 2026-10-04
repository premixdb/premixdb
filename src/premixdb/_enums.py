"""Public policy enums. String values preserve the version-1 SDK vocabulary."""

from enum import Enum

from ._field_catalog import IntrinsicField as IntrinsicField


class DedupeAlgorithm(str, Enum):
    EXACT_DOCUMENT = "exact_document"
    EXACT_LINE = "exact_line"


class RemovalUnit(str, Enum):
    DOCUMENT = "document"


class ExecutionStatus(str, Enum):
    UNSPECIFIED = "unspecified"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
