"""Partitionable field and dedupe-evidence producers for workers.

Importing this module does not load third-party processing or model runtimes.
The query planner derives built-in fields automatically when queries need them.
"""

from premixdb.enrichment.classification import probabilities, top_class
from premixdb.enrichment.datatrove import DataTroveFields, from_datatrove
from premixdb.enrichment.dupekit import DupekitIndex
from premixdb.enrichment.language import LanguageScores
from premixdb.enrichment.models import Embeddings, QuRating, WebOrganizer
from premixdb.enrichment.types import Document, ModelPin

__all__ = [
    "DataTroveFields",
    "Document",
    "DupekitIndex",
    "LanguageScores",
    "Embeddings",
    "ModelPin",
    "QuRating",
    "WebOrganizer",
    "from_datatrove",
    "probabilities",
    "top_class",
]
