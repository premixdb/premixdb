"""Regenerate or verify ignored local bindings with the pinned development toolchain."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from _premixdb_build import generate_protos

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if local bindings are stale")
    args = parser.parse_args()
    generate_protos(root / "src", check=args.check)


if __name__ == "__main__":
    main()
