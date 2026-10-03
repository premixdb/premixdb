"""Stream a small peS2o v2 full-paper demo derived from S2ORC.

Source: https://huggingface.co/datasets/allenai/peS2o (ODC-BY).
Reads only the requested prefix of a pinned shard, not the full corpus.
The default validation prefix is for inspection. Use --split train for the
mixture tutorials; it writes a separate train.jsonl. Neither is representative.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tempfile
from pathlib import Path
from urllib.request import urlopen

REVISION = "636a503e44a3ca1b58e01fb61eab0825cd574de0"
SHARD = "data/v2/validation-00001-of-00002.json.gz"
URL = f"https://huggingface.co/datasets/allenai/peS2o/resolve/{REVISION}/{SHARD}"
OUTPUT = Path(__file__).resolve().parents[1] / ".cache/s2orc/papers.jsonl"
TRAIN_SHARD = "data/v2/train-00010-of-00020.json.gz"
TRAIN_URL = f"https://huggingface.co/datasets/allenai/peS2o/resolve/{REVISION}/{TRAIN_SHARD}"
TRAIN_OUTPUT = OUTPUT.with_name("train.jsonl")
MAX_RECORD_BYTES = 64 * 1024 * 1024


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(output: Path | None = None, *, limit: int = 1000, split: str = "validation") -> Path:
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be a positive document count")
    if split not in ("train", "validation"):
        raise ValueError("split must be train or validation")
    url = TRAIN_URL if split == "train" else URL
    output = Path(output) if output is not None else TRAIN_OUTPUT if split == "train" else OUTPUT
    receipt = output.with_suffix(".receipt.json")
    expected = dict(url=url, documents=limit)
    if output.is_file() and receipt.is_file():
        saved = json.loads(receipt.read_text())
        if all(saved.get(key) == value for key, value in expected.items()) and saved.get(
            "sha256"
        ) == digest(output):
            print(f"Verified cached S2ORC-derived demo: {output}")
            return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as target:
            temporary = Path(target.name)
            seen = set()
            with urlopen(url, timeout=60) as response, gzip.GzipFile(fileobj=response) as stream:
                for ordinal in range(limit):
                    line = stream.readline(MAX_RECORD_BYTES + 1)
                    if not line:
                        raise ValueError("shard ended before the requested document count")
                    if len(line) > MAX_RECORD_BYTES:
                        raise ValueError("demo record exceeds the document size limit")
                    row = json.loads(line)
                    identity, text = row["id"], row["text"]
                    if (
                        not isinstance(identity, str)
                        or not identity
                        or identity in seen
                        or not isinstance(text, str)
                        or row.get("source")
                        != ("s2orc/train" if split == "train" else "s2orc/valid")
                    ):
                        raise ValueError(f"expected distinct full-text S2ORC {split} papers")
                    seen.add(identity)
                    target.write((json.dumps(dict(id=identity, text=text)) + "\n").encode())
                    if (ordinal + 1) % 100 == 0:
                        print(f"Prepared {ordinal + 1:,} papers", flush=True)
        checksum = digest(temporary)
        temporary.replace(output)
        receipt.write_text(json.dumps(dict(**expected, sha256=checksum), indent=2) + "\n")
        print(f"Prepared {limit:,} S2ORC-derived papers → {output}")
        return output
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="override the split-specific output path")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    prepare(args.output, limit=args.limit, split=args.split)


if __name__ == "__main__":
    main()
