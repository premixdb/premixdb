"""Shared entry points for snapshot, query and dataset execution."""

from __future__ import annotations

from collections.abc import Iterable

from premixdb.engine.datasets import Dataset as Dataset
from premixdb.engine.datasets import HuggingFaceTokenizer as HuggingFaceTokenizer
from premixdb.engine.datasets import Sequence as Sequence
from premixdb.engine.identity import CodeVersion as CodeVersion
from premixdb.engine.mixtures import MixturePool as MixturePool
from premixdb.engine.plans import Step as Step
from premixdb.engine.plans import dedupe as dedupe
from premixdb.engine.plans import field_definition as field_definition
from premixdb.engine.plans import filter as filter
from premixdb.engine.plans import policy as policy
from premixdb.engine.plans import query_identity as query_identity
from premixdb.engine.queries import CorpusIndex as CorpusIndex
from premixdb.engine.queries import Query as Query
from premixdb.engine.queries import Row as Row
from premixdb.engine.snapshots import Snapshot as Snapshot
from premixdb.engine.snapshots import Store as Store
from premixdb.engine.sources import Source as Source
from premixdb.training.reader import Reader as Reader


def execute(snapshots: Iterable[Snapshot], steps: Iterable[Step], code: CodeVersion) -> Query:
    return CorpusIndex(snapshots).execute(steps, code)
