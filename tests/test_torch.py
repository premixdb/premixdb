"""Real PyTorch map-style loading, including spawned workers and local ranges."""

from __future__ import annotations

import pickle
import struct
import tempfile
import unittest

import pytest
from blake3 import blake3

import premixdb


class TorchTests(unittest.TestCase):
    def test_dataloader_batch_supports_a_training_step(self) -> None:
        import torch
        from torch.utils.data import DataLoader

        with tempfile.TemporaryDirectory() as directory:
            with premixdb.PremixDB(storage=directory) as client:
                dataset = (
                    client.Corpus("train", [premixdb.Source("a", "hello world")])
                    .query()
                    .mix(tokenizer=premixdb.ByteTokenizer(), sequence_length=8)[0]
                )
                data = dataset.torch()
            batch = next(iter(DataLoader(data, batch_size=2)))
            model = torch.nn.Sequential(torch.nn.Embedding(258, 8), torch.nn.Linear(8, 258))
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            logits = model(batch["input_ids"])
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, 258), batch["labels"][:, 1:].reshape(-1)
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue((batch["labels"] == -100).any())
            before = model[1].weight.detach().clone()
            loss.backward()
            optimizer.step()
            self.assertFalse(torch.equal(before, model[1].weight))
            self.assertTrue(
                all(torch.isfinite(parameter).all() for parameter in model.parameters())
            )
            data.reader.close()

    def test_attention_and_loss_masks_are_independent(self) -> None:
        from premixdb.storage.objects import ObjectStore
        from premixdb.training.torch import TorchDataset
        from premixdb.v1 import data_mixture_pb2 as d

        with tempfile.TemporaryDirectory() as directory:
            with ObjectStore(directory) as store:

                def save(data: bytes) -> premixdb.SpanRef:
                    return premixdb.SpanRef(
                        object=store.put("dataset", data),
                        end=len(data),
                        blake3_digest=blake3(data).digest(),
                    )

                sequence = d.Sequence(
                    ordinal=0,
                    tokens=save(struct.pack("<3I", 1, 2, 3)),
                    attention_mask=save(bytes([1, 1, 0])),
                    loss_mask=save(bytes([0, 1, 0])),
                )
                resource = d.Dataset(sequence_length=3)
                resource.profile.sequences = 1
                resource.sequences.append(
                    save(d.SequenceBatch(sequences=[sequence]).SerializeToString())
                )
                reader = premixdb.RangeReader(local_root=directory)
                try:
                    item = TorchDataset(resource, reader)[0]
                finally:
                    reader.close()
        self.assertEqual(item["attention_mask"].tolist(), [1, 1, 0])
        self.assertEqual(item["labels"].tolist(), [-100, 2, -100])

    @pytest.mark.integration
    def test_tensors_and_spawned_loader_do_not_need_live_client(self) -> None:
        import torch
        from torch.utils.data import DataLoader

        with tempfile.TemporaryDirectory() as directory:
            with premixdb.PremixDB(storage=directory) as client:
                query = client.Corpus("torch", [premixdb.Source("a", "abcdefgh")]).query()
                mix = query.mix(tokenizer=premixdb.ByteTokenizer(), sequence_length=4)
                datasets = [d.torch() for d in mix]
                self.assertEqual(len(datasets), 1)
                data = datasets[0]
                item = data[-1]
                self.assertEqual(item["input_ids"].tolist(), [256, 257, 257, 257])
                self.assertEqual(item["attention_mask"].tolist(), [1, 0, 0, 0])
                self.assertEqual(item["labels"].tolist(), [256, -100, -100, -100])
                self.assertEqual(item["input_ids"].dtype, torch.long)
                restored = pickle.loads(pickle.dumps(data))
            loader = DataLoader(
                restored, batch_size=2, num_workers=2, multiprocessing_context="spawn"
            )
            actual = [row.tolist() for batch in loader for row in batch["input_ids"]]
            self.assertEqual(actual, [data[i]["input_ids"].tolist() for i in range(len(data))])


class NumPyCompatibilityTests(unittest.TestCase):
    @pytest.mark.integration
    def test_numpy_bridge_in_a_fresh_process_has_no_abi_warning(self) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import numpy as np, torch; a = np.array([1, 2]); assert torch.from_numpy(a).numpy().tolist() == [1, 2]",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("Failed to initialize NumPy", result.stderr)
        self.assertNotIn("compiled using NumPy 1.x", result.stderr)
