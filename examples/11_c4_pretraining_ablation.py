"""Search topic or content-type weights with RegMix and select by held-out loss.

Uses the checked-in C4 sample. Classifiers download on first use.
The training loop comes first; the small causal transformer is defined below.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import cast

import torch
from _tutorial import C4, DEFAULT_STORAGE, check_inputs
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

import premixdb as p


@dataclass(frozen=True)
class DataConfig:
    input: Path = C4
    storage: Path = DEFAULT_STORAGE / "pretraining-ablation"
    limit: int = 64


@dataclass(frozen=True)
class ModelConfig:
    sequence_length: int = 64
    width: int = 64
    layers: int = 2
    heads: int = 2


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 40
    batch_size: int = 8
    learning_rate: float = 3e-3
    seed: int = 0  # Paired model initialization; data APIs already default to seed 0.


@dataclass(frozen=True)
class MixConfig:
    domains: type[p.Topic] | type[p.ContentType] = p.Topic
    candidates: int = 6


@dataclass(frozen=True)
class EvaluationConfig:
    batches: int = 4


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cpu"
    threads: int = 2


DATA = DataConfig()
MODEL = ModelConfig()
TRAINING = TrainingConfig()
MIX = MixConfig()  # Use MixConfig(domains=p.ContentType) for content-type proportions.
EVALUATION = EvaluationConfig()
RUNTIME = RuntimeConfig()


def train(dataset: p.Dataset) -> TrainingRun:
    """Fresh model, same initialization and update budget for every ablation."""
    torch.manual_seed(TRAINING.seed)
    model = LanguageModel(MODEL).to(RUNTIME.device)
    optimizer = AdamW(model.parameters(), lr=TRAINING.learning_rate)
    batches = DataLoader(dataset.train.torch(), batch_size=TRAINING.batch_size, num_workers=0)
    if len(batches) != TRAINING.steps:
        raise ValueError("Training data must fill the fixed update budget.")
    model.train()
    for raw in batches:
        batch = cast(dict[str, Tensor], raw)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["input_ids"].to(RUNTIME.device))
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, 256), batch["labels"][:, 1:].to(RUNTIME.device).reshape(-1)
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return TrainingRun(model, optimizer, loss.item())


def main() -> None:
    check_inputs(DATA.input, limit=DATA.limit)
    if (
        min(
            TRAINING.steps,
            TRAINING.batch_size,
            EVALUATION.batches,
            MIX.candidates,
            MODEL.width,
            MODEL.layers,
            MODEL.heads,
            RUNTIME.threads,
        )
        <= 0
        or MODEL.sequence_length < 2
        or MODEL.width % MODEL.heads
    ):
        raise ValueError(
            "Use positive budgets, sequence_length >= 2, and width divisible by heads."
        )
    torch.set_num_threads(RUNTIME.threads)
    with p.PremixDB(storage=DATA.storage) as db:
        snapshot = db.Corpus(
            "tutorial/c4-ablation", p.Source.read_jsonl(DATA.input, limit=DATA.limit)
        )
        population = snapshot.query(steps=[p.where(p.text.bytes > 0), p.dedupe()])
        candidates = population.mix(
            domains=MIX.domains,
            weights=p.RegMix(),
            n_candidates=MIX.candidates,
            tokens=TRAINING.steps * TRAINING.batch_size * MODEL.sequence_length,
            replacement=True,
            tokenizer=p.ByteTokenizer(),
            sequence_length=MODEL.sequence_length,
            packing=p.Concat(),
        )
        weights = candidates.weights
        if len(weights[0]) < 2:
            raise ValueError(
                "Search needs at least two training labels. Increase DATA.limit or change MIX.domains."
            )
        # The first candidate is the comparison reference, with no extra training run.
        # Mixture weights affect training only; held-out membership stays fixed.
        reference = candidates[0]
        validation = evaluation_batches(reference.validation)
        results: list[RunResult] = []
        models: dict[str, TrainingRun] = {}
        for index, dataset in enumerate(candidates):
            name = f"regmix-{index}"
            run = train(dataset)
            validation_loss = evaluate(run.model, validation)
            delta = validation_loss - results[0].validation_loss if results else 0.0
            print(f"{name:18s} validation={validation_loss:.4f} delta={delta:+.4f} nats/byte")
            print("  weights:", weights[index])
            profile = dataset.train.profile()
            results.append(
                RunResult(
                    name=name,
                    weights=weights[index],
                    query_id=population.id,
                    mixture_id=candidates.id,
                    candidate_index=index,
                    dataset_id=dataset.id,
                    train_documents=profile.source_documents,
                    content_tokens=profile.content_tokens,
                    domain_tokens=dict(profile.stratum_tokens),
                    steps=TRAINING.steps,
                    train_loss=run.loss,
                    validation_loss=validation_loss,
                    delta_from_first=delta,
                )
            )
            models[name] = run

        winner = min(results, key=lambda result: result.validation_loss)
        test = evaluation_batches(reference.test)  # Select before looking at test loss.
        test_losses = {
            name: evaluate(models[name].model, test)
            for name in dict.fromkeys((results[0].name, winner.name))
        }
        output = DATA.storage / "results"
        output.mkdir(parents=True, exist_ok=True)
        report = SearchReport(
            snapshot_id=snapshot.id,
            reference_dataset_id=reference.id,
            domains=MIX.domains.__name__,
            model=MODEL,
            training=TRAINING,
            evaluation=EVALUATION,
            runs=results,
            selected=winner.name,
            selected_weights=winner.weights,
            selected_dataset_id=winner.dataset_id,
            test_loss=test_losses,
        )
        (output / "report.json").write_text(
            json.dumps(asdict(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        run = models[winner.name]
        checkpoint = WinnerCheckpoint(
            model_state_dict=dict(run.model.state_dict()),
            optimizer_state_dict=run.optimizer.state_dict(),
            model=MODEL,
            training=TRAINING,
            dataset_id=winner.dataset_id,
            split="train",
            selected=winner.name,
            weights=winner.weights,
        )
        torch.save(asdict(checkpoint), output / "winner.pt")
        print(f"Selected: {winner.name}. Test loss: {test_losses[winner.name]:.4f} nats/byte.")
        print("Best measured weights:", winner.weights)
        print("Report and checkpoint:", output)


# Model and evaluation helpers; replace these with your own trainer if needed.
class LanguageModel(nn.Module):
    """PyTorch transformer blocks with causal attention and a byte LM head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.tokens = nn.Embedding(256, config.width)
        self.positions = nn.Embedding(config.sequence_length, config.width)
        block = nn.TransformerEncoderLayer(
            d_model=config.width,
            nhead=config.heads,
            dim_feedforward=4 * config.width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            block, num_layers=config.layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, 256, bias=False)
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.normal_(parameter, std=0.02)

    def forward(self, ids: Tensor) -> Tensor:
        length = ids.shape[1]
        positions = torch.arange(length, device=ids.device)
        hidden = self.tokens.forward(ids) + self.positions.forward(positions)
        mask = torch.ones(length, length, dtype=torch.bool, device=ids.device).triu(1)
        hidden = self.transformer.forward(hidden, mask=mask)
        return self.head.forward(self.norm.forward(hidden))


def evaluation_batches(dataset: p.Dataset | p.DatasetSplit) -> list[dict[str, Tensor]]:
    """Cache a bounded reference once so every model sees the same held-out bytes."""
    loader = DataLoader(dataset.torch(), batch_size=TRAINING.batch_size, num_workers=0)
    batches = [cast(dict[str, Tensor], batch) for batch in islice(loader, EVALUATION.batches)]
    if not batches:
        raise ValueError(
            "Empty held-out split. Increase DATA.limit or reduce MODEL.sequence_length."
        )
    return batches


def evaluate(model: LanguageModel, batches: list[dict[str, Tensor]]) -> float:
    model.eval()
    total, targets = 0.0, 0
    with torch.no_grad():
        for batch in batches:
            logits = model(batch["input_ids"].to(RUNTIME.device))
            labels = batch["labels"][:, 1:].to(RUNTIME.device).reshape(-1)
            total += F.cross_entropy(
                logits[:, :-1].reshape(-1, 256), labels, reduction="sum"
            ).item()
            targets += int((labels != -100).sum().item())
    return total / targets


@dataclass
class TrainingRun:
    model: LanguageModel
    optimizer: AdamW
    loss: float


@dataclass(frozen=True)
class RunResult:
    name: str
    weights: dict[str, float]
    query_id: str
    mixture_id: str
    candidate_index: int
    dataset_id: str
    train_documents: int
    content_tokens: int
    domain_tokens: dict[str, int]
    steps: int
    train_loss: float
    validation_loss: float
    delta_from_first: float


@dataclass(frozen=True)
class SearchReport:
    snapshot_id: str
    reference_dataset_id: str
    domains: str
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    runs: list[RunResult]
    selected: str
    selected_weights: dict[str, float]
    selected_dataset_id: str
    test_loss: dict[str, float]


@dataclass(frozen=True)
class WinnerCheckpoint:
    model_state_dict: dict[str, Tensor]
    optimizer_state_dict: dict[str, object]
    model: ModelConfig
    training: TrainingConfig
    dataset_id: str
    split: str
    selected: str
    weights: dict[str, float]


if __name__ == "__main__":
    main()
