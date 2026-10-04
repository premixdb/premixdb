"""Marin's native hash/MinHash kernels as mergeable Arrow evidence shards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from premixdb.contracts import Metadata
from premixdb.enrichment.types import Document

if TYPE_CHECKING:
    from pyarrow import RecordBatch

from blake3 import blake3

from premixdb.enrichment.types import check_documents, package_versions, positive
from premixdb.v1 import index_pb2 as ix


@dataclass(frozen=True)
class DupekitIndex:
    """All-member evidence, never a keep/drop mask or a preselected winner.

    Group exact candidates by hash, then verify original text equality. Fuzzy
    candidates join on (band ordinal, bucket); signatures estimate similarity.
    Short documents have no fuzzy evidence rather than matching empty shingles.
    """

    num_perms: int = 128
    num_bands: int = 16
    ngram_size: int = 5
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("num_perms", "num_bands", "ngram_size"):
            positive(getattr(self, name), name)
        if self.num_perms % self.num_bands:
            raise ValueError("num_perms must be divisible by num_bands")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be uint64")

    @property
    def definition(self) -> dict[str, Metadata]:
        """Describe the pinned assets and settings that determine output identity."""
        return {
            "provider": "marin-dupekit",
            "version": 1,
            "packages": package_versions("marin-dupekit", "pyarrow"),
            "num_perms": self.num_perms,
            "num_bands": self.num_bands,
            "ngram_size": self.ngram_size,
            "seed": self.seed,
            "normalization": "none",
            "exact_hash": "blake3",
            "short_documents": "no_fuzzy_evidence",
            "approximate": True,
        }

    @property
    def indexes(self) -> tuple[ix.Index, ...]:
        """Return the deduplication evidence definitions produced for each document."""
        result = []
        for name, approximate in (("dupekit.exact_candidates", False), ("dupekit.lsh", True)):
            index = ix.Index(name=name, version=1, kind=ix.INDEX_LOOKUP, approximate=approximate)
            index.id = blake3(index.SerializeToString(deterministic=True)).digest()
            result.append(index)
        return tuple(result)

    cache_scope = "document"

    def compute(self, documents: Sequence[Document]) -> RecordBatch:
        """Compute one result per document, preserving input order and empty documents."""
        import pyarrow as pa
        from dupekit import HashAlgorithm, Transformation, transform

        check_documents(documents)
        batch = pa.record_batch(
            {
                "id": pa.array([doc.id for doc in documents], type=pa.string()),
                "text": pa.array([doc.text for doc in documents], type=pa.string()),
            }
        )
        if not documents:
            return pa.record_batch(
                {
                    "id": pa.array([], type=pa.string()),
                    "exact_hash": pa.array([], type=pa.binary(32)),
                    "minhash": pa.array([], type=pa.list_(pa.uint64())),
                    "lsh_buckets": pa.array([], type=pa.list_(pa.uint64())),
                }
            )
        hashes = transform(batch, [Transformation.Hash("text", "exact_hash", HashAlgorithm.Blake3)])
        # Unicode scalar length matches Dupekit's character ngrams.
        eligible = [i for i, doc in enumerate(documents) if len(doc.text) >= self.ngram_size]
        signatures: list[list[int] | None] = [None] * len(documents)
        buckets: list[list[int] | None] = [None] * len(documents)
        if eligible:
            fuzzy = transform(
                batch.take(pa.array(eligible)),
                [
                    Transformation.MinHash(
                        "text", "minhash", self.num_perms, self.ngram_size, self.seed
                    ),
                    Transformation.MinHashLSH("minhash", "lsh_buckets", self.num_bands),
                ],
            )
            for i, signature, bucket in zip(
                eligible,
                fuzzy.column("minhash").to_pylist(),
                fuzzy.column("lsh_buckets").to_pylist(),
            ):
                signatures[i], buckets[i] = signature, bucket
        return pa.record_batch(
            {
                "id": batch.column("id"),
                "exact_hash": pa.array(
                    [bytes.fromhex(value) for value in hashes.column("exact_hash").to_pylist()],
                    type=pa.binary(32),
                ),
                "minhash": pa.array(signatures, type=pa.list_(pa.uint64())),
                "lsh_buckets": pa.array(buckets, type=pa.list_(pa.uint64())),
            }
        )
