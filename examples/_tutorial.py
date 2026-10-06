"""Input helpers for the numbered lessons; edit settings in each lesson."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from premixdb import PremixDB, Snapshot, Source

ROOT = Path(__file__).resolve().parents[1]
TINY = Path(str(files("premixdb").joinpath("data/tiny_shakespeare_excerpt.txt")))
DATA = ROOT / "examples/data"
C4 = DATA / "c4.jsonl"
PAPERS = DATA / "s2orc-validation.jsonl"
TRAIN_PAPERS = DATA / "s2orc-train.jsonl"
DEFAULT_STORAGE = ROOT / ".cache/tutorials"


def check_inputs(*paths: Path, limit: int) -> None:
    """Fail before capture when a lesson's edited settings are invalid."""
    if limit <= 0:
        raise ValueError("LIMIT must be positive. Start with a few documents.")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. See examples/README.md for the checked-in sample paths."
            )


def tiny_sources(path: Path = TINY, limit: int = 100) -> list[Source]:
    """Read each dialogue block as a document."""
    check_inputs(path, limit=limit)
    blocks = path.read_text(encoding="utf-8").strip().split("\n\n")
    return [Source(f"speech/{i:04d}", block + "\n\n") for i, block in enumerate(blocks[:limit])]


def mixture_sources(
    db: PremixDB,
    *,
    web: Path = C4,
    papers: Path = TRAIN_PAPERS,
    literature: Path = TINY,
    limit: int = 100,
) -> tuple[Snapshot, Snapshot, Snapshot]:
    """Capture web pages and training papers plus the Shakespeare excerpt."""
    check_inputs(web, papers, literature, limit=limit)
    web_snapshot = db.Corpus("tutorial/c4", Source.read_jsonl(web, limit=limit))
    science = db.Corpus(
        "tutorial/pes2o-train", Source.read_jsonl(papers, key_column="id", limit=limit)
    )
    literature_snapshot = db.Corpus("tutorial/tiny-shakespeare", tiny_sources(literature, limit))
    return web_snapshot, science, literature_snapshot
