"""Apply a prepared web recipe to the checked-in C4 sample.

Run from the repository root: uv run python -m examples.recipes.apply
For LLaMA/GPT-3, use their sources() functions with your own classifiers as
shown in README.md; this runner covers the recipes with supplied heuristics.
"""

from __future__ import annotations

from pathlib import Path

import premixdb as p

from . import falcon, gopher, t5

INPUT = Path(__file__).resolve().parents[1] / "data/c4.jsonl"
STORAGE = Path(__file__).resolve().parents[2] / ".cache/tutorials/model-recipes"
MODEL = "falcon"  # falcon, t5, or gopher
LIMIT = 100  # None reads the entire file; begin with a bounded sample.


def main() -> None:
    if MODEL not in ("falcon", "t5", "gopher"):
        raise ValueError("MODEL must be falcon, t5, or gopher")
    if LIMIT is not None and (type(LIMIT) is not int or LIMIT <= 0):
        raise ValueError("LIMIT must be positive or None")
    if not INPUT.is_file():
        raise FileNotFoundError(
            f"Missing {INPUT}. Supply extracted text JSONL; see recipes/README.md."
        )
    recipe = {"falcon": falcon, "t5": t5, "gopher": gopher}[MODEL]
    with p.PremixDB(storage=STORAGE) as db:
        prepared = db.Corpus(
            f"recipes/{MODEL}/web", recipe.sources(p.Source.read_jsonl(INPUT, limit=LIMIT))
        )
        selected = recipe.query(prepared)
        print("Recipe adaptation:", MODEL)
        print("After preprocessing:", prepared.profile().documents)
        print("After query:", selected.profile().output_documents)
        print(selected.profile())
        print(selected.preview())


if __name__ == "__main__":
    main()
