"""Offline semantics plus installed-dependency integration tests (no model downloads)."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Unpack, cast
from unittest.mock import patch

from _type_support import (
    EncodeOptions,
    Vectors,
    numeric,
)

if TYPE_CHECKING:
    from torch import Tensor

from premixdb.enrichment import (
    DataTroveFields,
    Document,
    DupekitIndex,
    Embeddings,
    ModelPin,
    QuRating,
    WebOrganizer,
    from_datatrove,
    inference_device,
    probabilities,
    top_class,
)
from premixdb.enrichment.interfaces import ModelTokenizer, SequenceModel
from premixdb.enrichment.models import QUALITY_DIMENSIONS
from premixdb.enrichment.types import field
from premixdb.runtime.enrichment import numeric_vector
from premixdb.v1 import field_pb2 as fields

REVISION = "a" * 40


class ClassificationTests(unittest.TestCase):
    def test_softmax_preserves_label_order_and_handles_large_logits(self) -> None:
        spec = field("topic", width=3, classes=("Z", "A", "B"))
        result = probabilities(spec, [10000, 10001, -10000])
        self.assertEqual(list(result), ["Z", "A", "B"])
        self.assertAlmostEqual(sum(result.values()), 1)
        self.assertEqual(top_class(spec, [10000, 10001, -10000])[0], "A")
        spec.classification.temperature = 2
        self.assertLess(probabilities(spec, [10000, 10001, -10000])["A"], result["A"])

    def test_regression_is_not_classification_and_invalid_values_fail(self) -> None:
        with self.assertRaises(ValueError):
            probabilities(field("quality.educational_value"), [2])
        spec = field("type", width=2, classes=("one", "two"))
        for values in ([1], [math.nan, 0], [0, math.inf]):
            with self.assertRaises(ValueError):
                probabilities(spec, values)
        for temperature in (0, -1, math.nan):
            spec.classification.temperature = temperature
            with self.assertRaises(ValueError):
                probabilities(spec, [0, 1])

    def test_sigmoid_is_independent_and_has_no_top_class(self) -> None:
        spec = field("tags", width=2, classes=("a", "b"))
        spec.classification.transform = fields.PROBABILITY_TRANSFORM_SIGMOID
        self.assertEqual(probabilities(spec, [1000, 1000]), {"a": 1, "b": 1})
        with self.assertRaises(ValueError):
            top_class(spec, [1000, 1000])


def classifier_tokenizer() -> ModelTokenizer:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"[UNK]": 0, "1": 1, "[PAD]": 2, "3": 3, "[BOS]": 4}))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single="[BOS] $A", special_tokens=[("[BOS]", 4)])
    return cast(
        ModelTokenizer,
        PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", bos_token="[BOS]"
        ),
    )


class ModelTests(unittest.TestCase):
    def test_automatic_device_is_pinned_in_builtin_model_recipes(self) -> None:
        from premixdb.runtime.catalog import plan, recipe
        from premixdb.runtime.enrichment import producer

        pins = set()
        for cuda, mps, expected in (
            (True, True, "cuda"),
            (False, True, "mps"),
            (False, False, "cpu"),
        ):
            with (
                self.subTest(device=expected),
                patch("torch.cuda.is_available", return_value=cuda),
                patch("torch.backends.mps.is_available", return_value=mps),
            ):
                self.assertEqual(inference_device(), expected)
                for name in ("weborganizer.topic", "quality.writing_style", "embedding.harrier"):
                    policy = recipe(name)
                    self.assertEqual(policy.model.device, expected)
                    worker = producer(policy)
                    assert isinstance(worker, (WebOrganizer, QuRating, Embeddings))
                    self.assertEqual(worker.device, expected)
                pins.add(plan([b"snapshot"], "weborganizer.topic", b"commit").id)
        self.assertEqual(len(pins), 3)

    def test_weborganizer_attention_mask_excludes_padding(self) -> None:
        import torch

        from premixdb.enrichment.models import _weborganizer_attention_mask

        model = torch.nn.Linear(2, 2).double()
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        bias = _weborganizer_attention_mask(model, mask, (2, 3))
        self.assertEqual(bias.shape, (2, 1, 1, 3))
        self.assertEqual(bias.dtype, torch.float64)
        probabilities = torch.softmax(torch.ones(2, 2, 3, 3) + bias, dim=-1)
        torch.testing.assert_close(probabilities[0, :, :, 2], torch.zeros(2, 3).double())
        torch.testing.assert_close(probabilities[1, :, :, 0], torch.ones(2, 3).double())
        expanded = mask[:, None, :].expand(-1, 3, -1)
        torch.testing.assert_close(
            _weborganizer_attention_mask(model, expanded, (2, 3)), bias.expand(-1, 1, 3, -1)
        )

    def test_weborganizer_load_restores_constructor_buffers_and_preserves_weights(self) -> None:
        import torch
        from transformers import BertConfig, BertForSequenceClassification

        from premixdb.enrichment.models import _sequence_model

        class ConstructorEmbeddings(torch.nn.Module):
            def __init__(self, config: BertConfig) -> None:
                super().__init__()
                self.word_embeddings = torch.nn.Embedding(5, 8)
                self.register_buffer("position_ids", torch.arange(16), persistent=False)
                self.rotary_emb = torch.nn.Module()
                frequencies = 1 / (10000 ** (torch.arange(0, 8, 2).float() / 8))
                angles = torch.outer(torch.arange(16).float(), frequencies)
                self.rotary_emb.register_buffer("inv_freq", frequencies, persistent=False)
                self.rotary_emb.register_buffer("cos_cached", angles.cos(), persistent=False)
                self.rotary_emb.register_buffer("sin_cached", angles.sin(), persistent=False)

        config = BertConfig(
            vocab_size=5,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
        config.num_labels = 24
        model = BertForSequenceClassification(config)
        model.add_module("new", torch.nn.Module())
        embeddings = ConstructorEmbeddings(config)
        model.get_submodule("new").add_module("embeddings", embeddings)
        expected = {name: value.clone() for name, value in embeddings.named_buffers()}
        weights = {name: value.clone() for name, value in model.state_dict().items()}
        for buffer in embeddings.buffers():
            buffer.fill_(-3959422977)
        rng = torch.get_rng_state().clone()
        with (
            patch(
                "transformers.AutoTokenizer.from_pretrained", return_value=classifier_tokenizer()
            ),
            patch(
                "transformers.AutoModelForSequenceClassification.from_pretrained",
                return_value=model,
            ),
        ):
            _, loaded = _sequence_model(
                ModelPin("WebOrganizer/TopicClassifier-NoURL", REVISION), "cpu", web=True
            )
        self.assertIs(loaded, model)
        self.assertTrue(callable(model.get_submodule("new").get_extended_attention_mask))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, buffer in embeddings.named_buffers():
            torch.testing.assert_close(buffer, expected[name])
        for name, weight in model.state_dict().items():
            torch.testing.assert_close(weight, weights[name])
        self.assertFalse(any("position_ids" in name or "rotary_emb" in name for name in weights))

    def test_builtin_metadata_and_empty_batches_do_not_load_models(self) -> None:
        from premixdb.enrichment.language import LanguageScores
        from premixdb.runtime.catalog import recipe
        from premixdb.runtime.enrichment import producer

        with patch(
            "premixdb.enrichment.models._sequence_model", side_effect=AssertionError("model load")
        ):
            for name in (
                "quality.educational_value",
                "weborganizer.topic",
                "weborganizer.content_type",
                "embedding.harrier",
                "language.en",
            ):
                worker = producer(recipe(name))
                assert not isinstance(worker, DupekitIndex)
                assert worker.fields
                assert worker.definition
                assert worker.compute([]) == []
            quality = producer(recipe("quality.educational_value"))
            assert quality.compute([Document("empty", " ")])[0]["quality.educational_value"] is None
            language = producer(recipe("language.en"))
            with patch.object(LanguageScores, "_lid", new_callable=property):
                assert language.compute([Document("empty", "")])[0]["language.en"] is None

    def test_quality_covers_tail_and_uses_content_weighted_windows(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(num_labels=4))
        windows_seen = []
        tokenizer = classifier_tokenizer()

        def scores(
            model: SequenceModel, windows: list[dict[str, list[int]]], rows: int, width: int
        ) -> list[list[float]]:
            windows_seen.extend(windows)
            return [[float(window["input_ids"][1])] * 4 for window in windows]

        with (
            patch("premixdb.enrichment.models._sequence_model", return_value=(tokenizer, model)),
            patch.object(tokenizer, "pad", side_effect=lambda windows, **kwargs: windows),
            patch("premixdb.enrichment.models._logits", side_effect=scores),
        ):
            provider = QuRating(ModelPin("princeton-nlp/QuRater-1.3B", REVISION), batch_size=2)
            docs = [Document("a", " ".join(["1"] * 511 + ["3"] * 2)), Document("empty", "")]
            rows = provider.compute(docs)
            for name in QUALITY_DIMENSIONS:
                self.assertAlmostEqual(numeric(rows[0][f"quality.{name}"]), 517 / 513)
                self.assertIsNone(rows[1][f"quality.{name}"])
            self.assertEqual([len(w["input_ids"]) for w in windows_seen], [512, 3])
            self.assertTrue(all(not f.HasField("classification") for f in provider.fields))
            self.assertEqual(rows, provider.compute(docs[:1]) + provider.compute(docs[1:]))

    def test_quality_runs_with_installed_transformers_without_model_downloads(self) -> None:
        from transformers.utils import is_torch_available

        self.assertTrue(
            is_torch_available(), "installed Transformers must support installed PyTorch"
        )
        from transformers import BertConfig, BertForSequenceClassification

        config = BertConfig(
            vocab_size=5,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            pad_token_id=2,
        )
        config.num_labels = 4
        model = BertForSequenceClassification(config).eval()
        provider = QuRating(ModelPin("princeton-nlp/QuRater-1.3B", REVISION))
        provider._model_pair = (classifier_tokenizer(), model)
        rows = provider.compute([Document("short", "1 3"), Document("long", "1 " * 513)])
        self.assertEqual([row["id"] for row in rows], ["short", "long"])
        for row in rows:
            self.assertTrue(
                all(math.isfinite(numeric(row[f"quality.{name}"])) for name in QUALITY_DIMENSIONS)
            )

    def test_weborganizer_numeric_label_order_url_and_raw_logits(self) -> None:
        labels = {str(i): f"class {i}" for i in reversed(range(24))}
        model = SimpleNamespace(config=SimpleNamespace(id2label=labels))
        calls = []

        def tokenize(texts: list[str], **kwargs: str | bool | int) -> dict[str, Tensor]:
            calls.extend(texts)
            return {}

        with (
            patch("premixdb.enrichment.models._sequence_model", return_value=(tokenize, model)),
            patch("premixdb.enrichment.models._logits", return_value=[list(range(24))]),
        ):
            provider = WebOrganizer(
                ModelPin("WebOrganizer/TopicClassifier", REVISION), task="topic"
            )
            spec = provider.fields[0]
            self.assertEqual(list(spec.classification.classes), [f"class {i}" for i in range(24)])
            result = provider.compute([Document("a", "web text", "https://example.org")])
            self.assertEqual(calls, ["https://example.org\n\nweb text"])
            self.assertEqual(result[0][spec.name], list(range(24)))
            self.assertEqual(top_class(spec, numeric_vector(result[0][spec.name]))[0], "class 23")
            with self.assertRaises(ValueError):
                provider.compute([Document("b", "missing URL")])
            no_url = WebOrganizer(
                ModelPin("WebOrganizer/FormatClassifier-NoURL", REVISION), task="content_type"
            )
            no_url.compute([Document("a", "plain text")])
            self.assertEqual(calls[-1], "plain text")

    def test_content_type_query_runs_with_installed_pytorch(self) -> None:
        import torch
        from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

        import premixdb as p
        from premixdb.enrichment.models import _sequence_model

        classes = tuple(label.value for label in p.ContentType)
        config = BertConfig(
            vocab_size=5,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            pad_token_id=2,
            id2label=dict(enumerate(classes)),
            label2id={label: i for i, label in enumerate(classes)},
        )
        model = BertForSequenceClassification(config).eval()
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.zero_()
            model.classifier.bias[classes.index(p.ContentType.LISTICLE.value)] = 10
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model.save_pretrained(root / "model", safe_serialization=True)
            cast(PreTrainedTokenizerFast, classifier_tokenizer()).save_pretrained(root / "model")
            pair = _sequence_model(ModelPin(str(root / "model"), REVISION), "cpu")
            with (
                patch("premixdb.enrichment.models._sequence_model", return_value=pair),
                p.PremixDB(storage=root / "db") as db,
            ):
                db.Corpus("demo", [p.Source("a", "1 3"), p.Source("b", "3 1")])
                query = db.Corpus("demo").query(steps=[p.where(p.content_type.listicle >= 0.5)])
                self.assertEqual(query.profile().output_documents, 2)
                self.assertEqual({row["source_key"] for row in query.preview()}, {"a", "b"})

    def test_embedding_document_and_query_prompts(self) -> None:
        calls = []

        class Encoder:
            def __init__(self, *args: str, **kwargs: str) -> None:
                self.max_seq_length = None

            def get_embedding_dimension(self) -> int:
                return 2

            def encode(self, texts: list[str], **kwargs: Unpack[EncodeOptions]) -> Vectors:
                calls.append(kwargs)
                return cast(Vectors, SimpleNamespace(tolist=lambda: [[1, 0] for _ in texts]))

        with patch.dict(
            "sys.modules", {"sentence_transformers": SimpleNamespace(SentenceTransformer=Encoder)}
        ):
            for family, repository, prompt in (
                ("harrier", "microsoft/harrier-oss-v1-0.6b", "web_search_query"),
            ):
                provider = Embeddings(ModelPin(repository, REVISION), family=family)
                self.assertEqual(provider.compute([]), [])
                provider.compute([Document("a", "text")])
                provider.encode_queries(["query"])
                self.assertEqual(calls[-2]["prompt"], "")
                self.assertEqual(calls[-1]["prompt_name"], prompt)
                self.assertTrue(calls[-1]["normalize_embeddings"])

    def test_input_validation(self) -> None:
        with self.assertRaises(ValueError):
            ModelPin("repo", "main")
        with self.assertRaises(ValueError):
            DupekitIndex(num_perms=10, num_bands=3)
        with self.assertRaises(ValueError):
            DupekitIndex(seed=-1)


class DupekitIntegrationTests(unittest.TestCase):
    def test_partition_equivalence_complete_membership_and_empty_schema(self) -> None:
        from blake3 import blake3

        provider = DupekitIndex(num_perms=16, num_bands=4)
        docs = [
            Document("a", "a useful repeated document"),
            Document("b", "a useful repeated document"),
            Document("c", "different useful text"),
            Document("short", "é"),
            Document("empty", ""),
        ]
        batch = provider.compute(docs)
        rows = batch.to_pylist()
        self.assertEqual(batch.schema, provider.compute([]).schema)
        self.assertEqual(
            rows, provider.compute(docs[:2]).to_pylist() + provider.compute(docs[2:]).to_pylist()
        )
        self.assertEqual(len(rows), len(docs))
        self.assertEqual(rows[0]["exact_hash"], blake3(docs[0].text.encode()).digest())
        self.assertEqual(len(rows[0]["exact_hash"]), 32)
        self.assertEqual(rows[0]["minhash"], rows[1]["minhash"])
        self.assertEqual(rows[0]["lsh_buckets"], rows[1]["lsh_buckets"])
        self.assertNotEqual(rows[0]["minhash"], rows[2]["minhash"])
        self.assertIsNone(rows[3]["minhash"])
        self.assertIsNone(rows[4]["lsh_buckets"])
        with self.assertRaises(ValueError):
            provider.compute([docs[0], docs[0]])


class DataTroveIntegrationTests(unittest.TestCase):
    def test_real_fields_preserve_empty_documents(self) -> None:
        provider = DataTroveFields()
        docs = [Document("a", "the cat is here."), Document("empty", ""), Document("space", " ")]
        rows = provider.compute(docs)
        self.assertEqual(rows[0]["datatrove.length"], 16)
        self.assertGreater(numeric(rows[0]["datatrove.n_words"]), 0)
        self.assertGreater(numeric(rows[0]["datatrove.stop_word_ratio"]), 0)
        self.assertEqual(rows[1]["datatrove.n_words"], 0)
        self.assertIsNone(rows[1]["datatrove.avg_word_length"])
        self.assertEqual(rows[2]["datatrove.white_space_ratio"], 1)
        self.assertEqual(rows, [row for doc in docs for row in provider.compute([doc])])
        self.assertEqual(set(rows[0]) - {"id"}, {f.name for f in provider.fields})

    def test_datatrove_reader_bridge(self) -> None:
        from datatrove.data import Document as DTDocument

        source = DTDocument(id="42", text="Hello", metadata={"url": "https://example.org"})
        result = list(from_datatrove([source], namespace="crawl"))[0]
        self.assertEqual(result.url, "https://example.org")
        self.assertNotEqual(result.id, list(from_datatrove([source], namespace="other"))[0].id)


if __name__ == "__main__":
    unittest.main()
