# Curation

Run from the repository root with an open `db` and `import premixdb as p`.
Capture the checked-in training and evaluation samples:

```python
from pathlib import Path

training = db.Corpus("training", p.Source.read_jsonl(Path("examples/data/c4.jsonl")))
evaluation = db.Corpus(
    "evaluation",
    p.Source.read_jsonl(Path("examples/data/s2orc-validation.jsonl"), key_column="id"),
)
query = training.query(
    steps=[
        p.where(p.text.characters > 0),
        p.dedupe(order_by=[p.object.uri.asc()]),
    ],
    decontaminate=p.Decontaminate(evaluation, algorithm="document"),
    sampling=p.sample(seed=42, documents=100),
)
query.preview()
query.profile()
```

Steps run in order, followed by decontamination and sampling. Pass decontamination
separately from `steps`. Use separate `where()` calls for each condition.

## Dedupe

```python
p.dedupe(order_by=[p.quality.educational_value.desc()])
p.similarity_dedupe(n=5, threshold=0.8)
p.similarity_dedupe(embedding=p.embedding.harrier, threshold=0.9)
p.indexed_dedupe(p.DedupeIndex.MINHASH_LSH, threshold=0.8)
```

Ordering chooses which copy survives. Similarity dedupe compares against earlier
survivors. LSH retrieves candidates and can miss matching pairs.

## Sample

```python
p.sample(seed=42, fraction=0.1)
p.sample(seed=42, tokens=100_000)
p.sample(seed=42, documents=100, domains=p.topic.label)
```

Choose one budget: documents, fraction, bytes, characters, or tokens. Size budgets
keep whole documents and can overshoot. `replacement=True` permits repeats.

## Mix

```python
mixtures = query.mix(
    tokens=100_000,
    weights=p.RegMix(seed=42),
    bounds=p.Bounds(lower={training.corpus_id: 0.1}),
    sequence_length=2048,
)
mixtures.weights
mixtures.profile()
dataset = mixtures[0]
```

Domains default to source corpora. Use `domains=p.Topic` to mix by topic.
The budget counts content tokens; separators and padding are reported separately.
Replacement defaults to false; set `replacement=True` to permit repeated passes.
Train and evaluate candidates to choose one.

## Tokenizers and packing

```python
from importlib.resources import files
from pathlib import Path

from blake3 import blake3

asset = Path(str(files("premixdb").joinpath("data/gpt2-tokenizer.json")))
tokenizer = p.hugging_face_tokenizer(asset, digest=blake3(asset.read_bytes()).hexdigest())
dataset = query.mix(tokenizer=tokenizer, sequence_length=2048)[0]
```

Use the model's tokenizer asset. `p.ByteTokenizer()` is useful for inspecting
UTF-8 bytes. `p.Concat(separator=..., pad_token=..., drop_remainder=...)` controls
boundaries and the final incomplete sequence. Padding labels are `-100`.
