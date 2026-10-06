"""Mix prepared source categories with a published model's fractions.

Edit the settings and run: uv run python -m examples.recipes.mix
Checked-in toy categories demonstrate allocation, not original training corpora.
"""

from __future__ import annotations

from pathlib import Path

import premixdb as p

from . import falcon, gopher, gpt3, llama, source_weights, t5

DATA = Path(__file__).resolve().parents[1] / "data"
C4 = DATA / "c4.jsonl"
TRAIN_PAPERS = DATA / "s2orc-train.jsonl"

MODEL = "llama"  # llama means LLaMA 1; also falcon, gopher, gpt3, t5
# Local stand-ins for each category; see ../data/README.md for provenance.
INPUTS = {
    "common_crawl": C4,
    "c4": C4,
    "code": DATA / "code.jsonl",
    "wiki": DATA / "reference.jsonl",
    "books": DATA / "literature.jsonl",
    "arxiv": TRAIN_PAPERS,
    "stack_exchange": DATA / "questions.jsonl",
}
STORAGE = Path(__file__).resolve().parents[2] / ".cache/tutorials/model-recipes"
LIMIT = 100
TOKENS = 256
SEQUENCE_LENGTH = 64
TOKENIZER = p.GPT2Tokenizer()  # Replace with the target model's tokenizer asset.
REPLACEMENT = False  # Falcon avoids upsampling; inspect capacity before permitting repeats.


def main() -> None:
    recipes = {"falcon": falcon, "gopher": gopher, "gpt3": gpt3, "llama": llama, "t5": t5}
    if MODEL not in recipes:
        raise ValueError(f"MODEL must be one of {sorted(recipes)}")
    recipe = recipes[MODEL]
    if set(INPUTS) != set(recipe.WEIGHTS):
        raise ValueError(f"INPUTS must contain these categories: {sorted(recipe.WEIGHTS)}")
    if LIMIT is not None and (type(LIMIT) is not int or LIMIT <= 0):
        raise ValueError("LIMIT must be positive or None")
    for input_file in INPUTS.values():
        if not input_file.is_file():
            raise FileNotFoundError(f"Missing {input_file}; see examples/recipes/README.md")
    with p.PremixDB(storage=STORAGE) as db:
        snapshots = {
            name: db.Corpus(f"recipes/{MODEL}/{name}", p.Source.read_jsonl(input_file, limit=LIMIT))
            for name, input_file in INPUTS.items()
        }
        first, *rest = snapshots.values()
        query = first.union(*rest).query(steps=[p.where(p.text.characters > 0)])
        mixture = query.mix(
            weights=source_weights(recipe.WEIGHTS, snapshots),
            tokens=TOKENS,
            tokenizer=TOKENIZER,
            sequence_length=SEQUENCE_LENGTH,
            replacement=REPLACEMENT,
        )
        print("Model recipe:", MODEL)
        print("Requested token fractions:", recipe.WEIGHTS)
        print(mixture[0].profile())
        print(mixture[0].preview())


if __name__ == "__main__":
    main()
