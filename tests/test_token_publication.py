"""Publication retains provenance for many short documents in one sequence."""

from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest

import premixdb as p
from premixdb import RangeReader
from premixdb.engine.datasets import Sequence
from premixdb.engine.identity import CodeVersion
from premixdb.engine.queries import CorpusIndex
from premixdb.engine.snapshots import Snapshot
from premixdb.storage.objects import ObjectStore
from premixdb.storage.tokens import publish
from premixdb.training.sequences import read_page
from premixdb.v1 import data_mixture_pb2 as d
from premixdb.v1.storage_pb2 import COMPRESSION_UNSPECIFIED, COMPRESSION_ZSTANDARD


def test_many_short_documents_keep_full_and_preview_provenance(tmp_path: Path) -> None:
    code = CodeVersion("local://publication", "a" * 40, "09" * 32)
    snapshot = Snapshot("01" * 16, [(f"{i:04}", "é") for i in range(512)], code)
    query = CorpusIndex([snapshot]).execute([], code)
    native = query.dataset(1024, None, None)
    with ObjectStore(tmp_path) as store, RangeReader(local_root=tmp_path) as reader:
        with (
            patch.object(
                Sequence,
                "tokens",
                new_callable=PropertyMock,
                side_effect=AssertionError("expanded all token values"),
            ),
            patch.object(
                Sequence,
                "mask",
                new_callable=PropertyMock,
                side_effect=AssertionError("expanded all mask values"),
            ),
        ):
            token_spans, pages, preview = publish(store, native)
        resource = d.Dataset(sequence_length=1024, sequences=pages, tokens=token_spans)
        resource.profile.sequences = 1
        sequence = read_page(resource, reader, 0)[0]
        assert sequence.tokens == list("é".encode()) * 512
        assert sequence.mask == [True] * 1024
        regions = sequence.spans
        assert len(regions) == 512
        for ordinal, region in enumerate(regions):
            assert region.kind == d.TokenRegion.KIND_CONTENT
            assert (region.start, region.end) == (ordinal * 2, ordinal * 2 + 2)
            assert region.query_ordinal == ordinal
            assert region.document_id.hex() == native.occurrence_document(ordinal)
            assert region.document_token_start == 0
            assert list(region.source_ranges) == [
                d.TokenByteRange(token=ordinal * 2, token_end=ordinal * 2 + 2, start=0, end=2)
            ]
        example = preview.sequences[0]
        assert example.text == "é" * 128
        assert example.truncated
        assert len(example.tokens) == len(example.loss_mask) == len(example.attention_mask) == 256
        assert list(example.regions) == regions[:128]


def test_oversized_sequence_pages_publish_and_reopen_in_all_readers(tmp_path: Path) -> None:
    from premixdb.training.torch import StreamingDataset, TorchDataset

    # Exercise the real publisher at a smaller IO limit, including the final raw page.
    with (
        patch("premixdb.storage.tokens.MAX_RANGE_BYTES", 65536),
        patch("premixdb.storage.ranges.MAX_RANGE_BYTES", 65536),
    ):
        with p.PremixDB(storage=tmp_path, progress=False) as db:
            dataset = (
                db.Corpus("pages", [p.Source("a", "abcd" * 129)])
                .query()
                .mix(
                    tokenizer=p.ByteTokenizer(), sequence_length=4, packing=p.Concat(separator=None)
                )[0]
                .wait()
            )
            resource, identity = dataset._proto, dataset.id
            assert resource.profile.sequences == 129
            assert resource.sequences[0].profile.content_bytes > 65536
            assert resource.sequences[0].end <= 65536
            assert resource.sequences[0].compression == COMPRESSION_ZSTANDARD
            assert resource.sequences[1].compression == COMPRESSION_UNSPECIFIED
        with p.PremixDB(storage=tmp_path, read_only=True) as reopened:
            assert reopened._dataset(identity)._proto == resource
            reader = reopened._object_reader
            for ordinal in (0, 127, 128):
                seq = read_page(resource, reader, ordinal, 1)[0]
                assert seq.ordinal == ordinal
                assert seq.tokens == list(b"abcd")
                assert seq.mask == [True] * 4
                assert seq.spans[0].document_token_start == ordinal * 4
            indexed = TorchDataset(resource, reader).__getitems__([128, 0, 127, 1])
            streamed = list(StreamingDataset(resource, reader))
            assert len(indexed) == 4 and len(streamed) == 129
            for example in [*indexed, *streamed]:
                assert example["input_ids"].tolist() == list(b"abcd")


@pytest.mark.parametrize("limit,kind", [(1, "encoded"), (256, "decoded")])
def test_unreadable_sequence_pages_are_rejected_during_publication(
    tmp_path: Path, limit: int, kind: str
) -> None:
    constant = "MAX_RANGE_BYTES" if kind == "encoded" else "MAX_INDEX_PAGE_BYTES"
    with p.PremixDB(storage=tmp_path, progress=False) as db:
        dataset = (
            db.Corpus("bounded", [p.Source("a", "abcd")])
            .query()
            .mix(tokenizer=p.ByteTokenizer(), sequence_length=4)[0]
        )
        with patch(f"premixdb.storage.tokens.{constant}", limit):
            with pytest.raises(ValueError, match=f"oversized {kind} sequence index"):
                dataset.wait()
        failed = db._dataset(dataset.id)
        assert failed.status is p.ExecutionStatus.ERROR
        assert not failed._proto.sequences
