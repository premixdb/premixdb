"""Partitionable field and dedupe-evidence producers for workers.

Importing this module does not load third-party processing or model runtimes.
The query planner derives built-in fields automatically when queries need them.
"""

from .classification import probabilities, top_class
from .datatrove import DataTroveFields, from_datatrove
from .dupekit import DupekitIndex
from .language import LanguageScores
from .models import Embeddings, QuRating, WebOrganizer
from .types import Document, ModelPin

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
