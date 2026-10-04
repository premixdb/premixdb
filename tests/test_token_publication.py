"""Publication retains provenance for many short documents in one sequence."""

from pathlib import Path
from unittest.mock import PropertyMock, patch

from premixdb import RangeReader
from premixdb.engine.datasets import Sequence
from premixdb.engine.identity import CodeVersion
from premixdb.engine.queries import CorpusIndex
from premixdb.engine.snapshots import Snapshot
from premixdb.storage.objects import ObjectStore
from premixdb.storage.tokens import publish
from premixdb.training.sequences import read_page
from premixdb.v1 import data_mixture_pb2 as d


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
