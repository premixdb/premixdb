"""Fluent data resources backed by protobuf requests and Python execution.

Open local storage, compose recipes, and train from saved data.
Raw request builders and generated premixdb.v1 messages describe saved recipes.
"""

from premixdb.api import (
    Corpus,
    DataMixture,
    Dataset,
    DatasetSplit,
    PremixDB,
    Query,
    Snapshot,
)
from premixdb.api.curation import decontaminate, sample, similarity_dedupe
from premixdb.api.profiles import DistributionSummary, HistogramBucket, QuantileRange
from premixdb.contracts import (
    Changes,
    Checkpoint,
    CorpusListing,
    DocumentListing,
    ExecutionError,
    ExecutionRecord,
    SnapshotListing,
)
from premixdb.engine.mixing import Bounds, RegMix, Tokens
from premixdb.engine.policies import DecontaminateDefault, SamplerDefault
from premixdb.engine.sources import HuggingFaceSource, Source
from premixdb.fields import (
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
from premixdb.schemas.enums import DedupeAlgorithm, ExecutionStatus, IntrinsicField, RemovalUnit
from premixdb.schemas.requests import (
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
from premixdb.schemas.requests import byte_tokenizer as ByteTokenizer
from premixdb.schemas.requests import concat as Concat
from premixdb.schemas.requests import gpt2_tokenizer as GPT2Tokenizer
from premixdb.storage.ranges import RangeReader
from premixdb.training.reader import Topology
from premixdb.v1.data_mixture_pb2 import (
    DatasetProfile,
    Domains,
    MixPreview,
    MixProfile,
    Sampling,
    Splits,
    Tokenizer,
)
from premixdb.v1.field_pb2 import Field, FieldSnapshot
from premixdb.v1.index_pb2 import Index, IndexSnapshot
from premixdb.v1.profile_pb2 import (
    DocumentEstimate,
    FieldDistribution,
    FieldProfile,
    NumericSummary,
    QueryEstimate,
)
from premixdb.v1.query_pb2 import QueryProfile, QueryStepProfile
from premixdb.v1.snapshot_pb2 import SnapshotProfile
from premixdb.v1.storage_pb2 import (
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
from premixdb.v1.storage_pb2 import Source as SourceSpec
from premixdb.version import __version__

__all__ = [
    "DatasetSplit",
    "Splits",
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
