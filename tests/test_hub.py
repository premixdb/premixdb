"""Only the optional Hub-authentication advisory is quiet during our downloads."""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from premixdb._hub import quiet_auth_advisory
from premixdb.enrichment.models import _sequence_model
from premixdb.enrichment.types import ModelPin
from premixdb.execution.sources import capture
from premixdb.v1 import storage_pb2 as s

ADVISORY = "Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN."


@pytest.fixture
def hub_output() -> Iterator[tuple[logging.Logger, io.StringIO]]:
    logger = logging.getLogger("huggingface_hub.utils._http")
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    before = list(logger.filters)
    logger.addHandler(handler)
    try:
        yield logger, output
    finally:
        logger.removeHandler(handler)
        assert logger.filters == before


def test_advisory_is_scoped_and_retries_errors_and_exceptions_survive(
    hub_output: tuple[logging.Logger, io.StringIO],
) -> None:
    logger, output = hub_output
    with pytest.raises(OSError, match="download failed"):
        with quiet_auth_advisory():
            with quiet_auth_advisory():
                logger.warning(ADVISORY)
            logger.warning(ADVISORY)
            logger.warning("HTTP Error 429: retrying")
            logger.error("HTTP Error 403: access denied")
            raise OSError("download failed")
    assert "unauthenticated" not in output.getvalue()
    assert "retrying" in output.getvalue() and "access denied" in output.getvalue()
    logger.warning(ADVISORY)
    assert "unauthenticated" in output.getvalue()


def test_overlapping_downloads_keep_the_filter_until_both_finish(
    hub_output: tuple[logging.Logger, io.StringIO],
) -> None:
    logger, output = hub_output
    entered, release = Event(), Event()

    def other_download() -> None:
        with quiet_auth_advisory():
            entered.set()
            release.wait(timeout=3)

    worker = Thread(target=other_download)
    try:
        with quiet_auth_advisory():
            worker.start()
            assert entered.wait(timeout=3)
        logger.warning(ADVISORY)
        assert not output.getvalue()
    finally:
        release.set()
        worker.join(timeout=3)
    logger.warning(ADVISORY)
    assert "unauthenticated" in output.getvalue()


def test_classifier_loading_quiets_the_advisory_without_overriding_auth(
    hub_output: tuple[logging.Logger, io.StringIO],
) -> None:
    logger, output = hub_output
    model = SimpleNamespace(config=SimpleNamespace(), to=lambda device: model, eval=lambda: model)

    def load(repository: str, **options: str | int | bool) -> SimpleNamespace:
        assert "token" not in options
        logger.warning(ADVISORY)
        return SimpleNamespace(pad_token_id=1)

    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load),
        AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=lambda *a, **k: model),
    )
    with patch.dict("sys.modules", {"transformers": transformers}):
        _, result = _sequence_model(ModelPin("org/model", "a" * 40), "cpu")
        assert result is model
    assert not output.getvalue()


def test_dataset_streaming_quiets_the_advisory_and_restores_on_close(
    hub_output: tuple[logging.Logger, io.StringIO],
) -> None:
    logger, output = hub_output

    def rows() -> Iterator[dict[str, str]]:
        logger.warning(ADVISORY)
        yield {"text": "hello"}
        logger.warning("download retry")
        yield {"text": "world"}

    source = s.Source(hugging_face=s.HuggingFaceDataset(repository="org/data", split="train"))
    with patch.dict(
        "sys.modules", {"datasets": SimpleNamespace(load_dataset=lambda *a, **k: rows())}
    ):
        iterator = capture(source)
        assert next(iterator)[1] == "hello"
        assert next(iterator)[1] == "world"
        assert output.getvalue() == "download retry\n"
        iterator.close()
    logger.warning(ADVISORY)
    assert "unauthenticated" in output.getvalue()
