"""Shared entry points for snapshot, query and dataset execution."""

from __future__ import annotations

from collections.abc import Iterable

from .._inputs import Source as Source
from .._reader import Reader as Reader
from .datasets import Dataset as Dataset
from .datasets import HuggingFaceTokenizer as HuggingFaceTokenizer
from .datasets import Sequence as Sequence
from .identity import CodeVersion as CodeVersion
from .mixtures import MixturePool as MixturePool
from .plans import Step as Step
from .plans import dedupe as dedupe
from .plans import field_definition as field_definition
from .plans import filter as filter
from .plans import policy as policy
from .plans import query_identity as query_identity
from .queries import CorpusIndex as CorpusIndex
from .queries import Query as Query
from .queries import Row as Row
from .snapshots import Snapshot as Snapshot
from .snapshots import Store as Store


def execute(snapshots: Iterable[Snapshot], steps: Iterable[Step], code: CodeVersion) -> Query:
    return CorpusIndex(snapshots).execute(steps, code)
