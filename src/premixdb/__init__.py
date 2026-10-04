"""Fluent data resources backed by protobuf requests and Python execution.

Open local storage, compose recipes, and train from saved data.
Raw request builders and generated premixdb.v1 messages describe saved recipes.
"""

from ._curation import decontaminate, sample, similarity_dedupe
from ._enums import DedupeAlgorithm, ExecutionStatus, IntrinsicField, RemovalUnit
from ._inputs import HuggingFaceSource, Source
from ._mixing import Bounds, RegMix, Tokens
from ._policies import DecontaminateDefault, SamplerDefault
from ._profiles import DistributionSummary, HistogramBucket, QuantileRange
from ._requests import (
    SourceGroup,
    byte_tokenizer,
    concat,
    corpus,
    dedupe,
    document_id,
    gpt2_tokenizer,
    hugging_face_tokenizer,
    indexed_dedupe,
    mix,
    object,
    query,
    snapshot,
    source,
    text,
    where,
)
from ._requests import byte_tokenizer as ByteTokenizer
from ._requests import concat as Concat
from ._requests import gpt2_tokenizer as GPT2Tokenizer
from ._resources import (
    Corpus,
    DataMixture,
    Dataset,
    ExecutionError,
    PremixDB,
    Query,
    Snapshot,
    Topology,
)
from ._storage import RangeReader
from ._types import (
    Changes,
    Checkpoint,
    CorpusListing,
    DocumentListing,
    ExecutionRecord,
    SnapshotListing,
)
from ._version import __version__
from .fields import (
    ContentType,
    DataTroveMetric,
    DedupeIndex,
    EmbeddingModel,
    Language,
    Quality,
    Topic,
    content_type,
    datatrove,
    embedding,
    language,
    quality,
    topic,
)
from .v1.data_mixture_pb2 import (
    DatasetProfile,
    Domains,
    MixPreview,
    MixProfile,
    Sampling,
    Tokenizer,
)
from .v1.field_pb2 import Field, FieldSnapshot
from .v1.index_pb2 import Index, IndexSnapshot
from .v1.profile_pb2 import (
    DocumentEstimate,
    FieldDistribution,
    FieldProfile,
    NumericSummary,
    QueryEstimate,
)
from .v1.query_pb2 import QueryProfile, QueryStepProfile
from .v1.snapshot_pb2 import SnapshotProfile
from .v1.storage_pb2 import (
    FileSource,
    FileSources,
    HuggingFaceDataset,
    MemorySource,
    MemorySources,
    ObjectProfile,
    ObjectRef,
    SourceManifest,
    SpanProfile,
    SpanRef,
)
from .v1.storage_pb2 import Source as SourceSpec

__all__ = [
    "DecontaminateDefault",
    "SamplerDefault",
    "__version__",
    "DistributionSummary",
    "HistogramBucket",
    "QuantileRange",
    "NumericSummary",
    "HuggingFaceSource",
    "decontaminate",
    "sample",
    "similarity_dedupe",
    "ContentType",
    "DataTroveMetric",
    "DedupeIndex",
    "EmbeddingModel",
    "Language",
    "Quality",
    "Topic",
    "content_type",
    "datatrove",
    "embedding",
    "language",
    "quality",
    "topic",
    "indexed_dedupe",
    "Field",
    "FieldSnapshot",
    "Index",
    "IndexSnapshot",
    "RangeReader",
    "SnapshotProfile",
    "QueryProfile",
    "QueryStepProfile",
    "QueryEstimate",
    "DocumentEstimate",
    "FieldProfile",
    "FieldDistribution",
    "DatasetProfile",
    "ObjectProfile",
    "SpanProfile",
    "SpanRef",
    "Bounds",
    "RegMix",
    "Tokens",
    "DataMixture",
    "MixProfile",
    "MixPreview",
    "mix",
    "Changes",
    "Checkpoint",
    "CorpusListing",
    "DocumentListing",
    "ExecutionRecord",
    "SnapshotListing",
    "DedupeAlgorithm",
    "RemovalUnit",
    "ExecutionStatus",
    "IntrinsicField",
    "PremixDB",
    "ByteTokenizer",
    "GPT2Tokenizer",
    "gpt2_tokenizer",
    "Concat",
    "ExecutionError",
    "Corpus",
    "Dataset",
    "FileSource",
    "FileSources",
    "HuggingFaceDataset",
    "ObjectRef",
    "Query",
    "MemorySource",
    "MemorySources",
    "Snapshot",
    "Source",
    "SourceSpec",
    "SourceGroup",
    "SourceManifest",
    "Tokenizer",
    "Sampling",
    "Domains",
    "Topology",
    "byte_tokenizer",
    "concat",
    "corpus",
    "dedupe",
    "hugging_face_tokenizer",
    "query",
    "snapshot",
    "object",
    "source",
    "text",
    "where",
    "document_id",
]
