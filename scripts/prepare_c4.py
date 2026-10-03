"""Download one pinned English C4 training shard (319 MB compressed).

Dataset: https://huggingface.co/datasets/allenai/c4
License: ODC-BY; Common Crawl content terms also apply (see the dataset card).
The gzip JSONL is kept compressed. Existing downloads are verified and reused.
"""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path
from urllib.request import urlopen

REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
SHARD = "en/c4-train.00000-of-01024.json.gz"
URL = f"https://huggingface.co/datasets/allenai/c4/resolve/{REVISION}/{SHARD}"
SHA256 = "8ef8d75b0e045dec4aa5123a671b4564466b0707086a7ed1ba8721626dfffbc9"
OUTPUT = Path(__file__).resolve().parents[1] / ".cache/c4" / Path(SHARD).name


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(output: Path = OUTPUT) -> Path:
    output = Path(output)
    if output.is_file() and digest(output) == SHA256:
        print(f"Verified cached C4 shard: {output}")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as target:
            temporary = Path(target.name)
            checksum = hashlib.sha256()
            size = 0
            with urlopen(URL, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
                    checksum.update(chunk)
                    size += len(chunk)
                    if size % (32 * 1024 * 1024) == 0:
                        print(f"Downloaded {size / 1_000_000:.0f} MB", flush=True)
        if checksum.hexdigest() != SHA256:
            raise ValueError("C4 source checksum mismatch; download was not published")
        temporary.replace(output)
        print(f"Verified C4 shard: {size:,} compressed bytes → {output}")
        return output
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    prepare(args.output)


if __name__ == "__main__":
    main()
