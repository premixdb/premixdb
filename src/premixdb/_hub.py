"""Scope suppression of the Hub's optional-authentication advisory to our downloads."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from threading import Lock
from typing import Iterator


class _AuthAdvisory(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (
            record.levelno == logging.WARNING
            and record.getMessage()
            .removeprefix("Warning: ")
            .startswith("You are sending unauthenticated requests to the HF Hub.")
        )


_FILTER = _AuthAdvisory()
_LOCK = Lock()
_DOWNLOADS = 0


@contextmanager
def quiet_auth_advisory() -> Iterator[None]:
    """Keep retries/errors visible and leave normal Hub token discovery intact."""
    global _DOWNLOADS
    logger = logging.getLogger("huggingface_hub.utils._http")
    with _LOCK:
        if not _DOWNLOADS:
            logger.addFilter(_FILTER)
        _DOWNLOADS += 1
    try:
        yield
    finally:
        with _LOCK:
            _DOWNLOADS -= 1
            if not _DOWNLOADS:
                logger.removeFilter(_FILTER)
