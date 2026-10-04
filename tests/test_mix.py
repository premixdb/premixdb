"""New Mix resources: registered recipes, deterministic proposals and lazy packing."""

from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from _type_support import coordinator

import premixdb
from premixdb._ids import _decode_id, _encode_id, _public_dataset_profile
from premixdb._resources import DomainInput
from premixdb.engine.mixtures import MixturePool
from premixdb.v1 import dataset_pb2 as pb
from premixdb.v1 import query_pb2 as query_pb


class MixTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.client = premixdb.PremixDB(storage=self.root)
        self.addCleanup(self.client.close)
        self.query = self.client.corpus(
            "mix",
            [
                premixdb.Source("a", "abcd"),
                premixdb.Source("b", "XYZ"),
                premixdb.Source("empty", ""),
            ],
        ).query()

    def mix(
        self,
        *,
        domains: DomainInput = premixdb.object.uri,
        tokens: int = 9,
        sequence_length: int = 4,
        bounds: premixdb.Bounds | None = None,
        seed: int = 42,
        n_candidates: int = 1,
    ) -> premixdb.Mix:
        return self.query.mix(
            tokenizer=premixdb.ByteTokenizer(),
            domains=domains,
            tokens=tokens,
            sequence_length=sequence_length,
            bounds=premixdb.Bounds(max_epochs=4) if bounds is None else bounds,
            seed=seed,
            n_candidates=n_candidates,
        )

    def test_registered_recipes_profiles_and_restarts_are_lazy(self) -> None:
        mix = self.mix(n_candidates=3)
        self.assertEqual(len(mix), 3)
        self.assertEqual(len(mix._proto.dataset_ids), 3)
        weights = mix.weights
        self.assertFalse(coordinator(self.client)._storage.list("dataset", pb.Dataset))
        self.assertEqual(len(coordinator(self.client)._mix_pools), 1)
        self.assertIs(mix[0], mix[:1][-1])
        self.assertEqual(len(mix[:2]), 2)
        weights[0].clear()
        self.assertTrue(mix.weights[0])
        recipe = mix._configs[0]
        self.assertIsInstance(recipe, pb.CreateDatasetRequest)
        recipe.Clear()
        self.assertTrue(mix._configs[0].query_id)
        planned = mix.profile(0)
        self.assertEqual(sum(planned.planned_stratum_tokens.values()), 9)
        with premixdb.PremixDB(storage=self.root) as reopened:
            again = reopened._mix(mix.id)
            self.assertEqual(again.weights, mix.weights)
            self.assertFalse(hasattr(coordinator(reopened), "_dataset_handles"))
            dataset = again[0]
            self.assertIs(dataset.status, premixdb.ExecutionStatus.PENDING)
            list(dataset)
            self.assertEqual(dataset.profile(), planned)
            self.assertIs(dataset.status, premixdb.ExecutionStatus.COMPLETED)
        self.assertEqual(self.client._dataset(mix[0].id).status, premixdb.ExecutionStatus.COMPLETED)

    def test_each_candidate_prepares_its_profile_draws_once(self) -> None:
        with patch.object(
            MixturePool, "_draw", autospec=True, side_effect=MixturePool._draw
        ) as draw:
            mix = self.mix(n_candidates=3)
            self.assertEqual(draw.call_count, 3)
            planned = mix.profile(0)
            self.assertEqual(draw.call_count, 3)
            candidate = mix[0].wait()
            self.assertEqual(draw.call_count, 4)
            self.assertEqual(candidate.profile(), planned)
            self.assertEqual(len(candidate), planned.sequences)

    def test_invalid_mixture_options_do_not_run_pending_queries(self) -> None:
        invalid = (
            lambda: self.mix(tokens=0),
            lambda: self.mix(sequence_length=0),
            lambda: self.mix(n_candidates=0),
            lambda: self.mix(domains={"invalid-id": "letters"}),
            lambda: self.query.mix(size=premixdb.Tokens(1), tokens=1),
            lambda: self.query.mix(sampler=premixdb.RegMixSampler(minimum_weight=float("nan"))),
        )
        with patch.object(self.query, "wait", side_effect=AssertionError("query ran")):
            for index, build in enumerate(invalid):
                with self.subTest(case=index), self.assertRaises(ValueError):
                    build()
        self.assertIs(self.query.status, premixdb.ExecutionStatus.PENDING)

    def test_mixture_planning_requires_an_open_writable_session(self) -> None:
        self.query.wait()
        with premixdb.PremixDB(storage=self.root, read_only=True) as db:
            query = db._query(self.query.id)
            with (
                patch("premixdb._resources._requests.mix", side_effect=AssertionError("planned")),
                self.assertRaisesRegex(PermissionError, "read-only session.*plan mixtures"),
            ):
                query.mix()
        self.client.close()
        with (
            patch("premixdb._resources._requests.mix", side_effect=AssertionError("planned")),
            self.assertRaisesRegex(ValueError, "PremixDB is closed"),
        ):
            self.query.mix()

    def test_proposals_bounds_identity_and_retries(self) -> None:
        mix = self.mix(
            n_candidates=6,
            tokens=4,
            bounds=premixdb.Bounds(lower={"a": 0.1}, upper={"a": 0.9}, max_epochs=2),
        )
        self.assertEqual(len({tuple(sorted(w.items())) for w in mix.weights}), 6)
        for weights in mix.weights:
            self.assertAlmostEqual(sum(weights.values()), 1)
            self.assertTrue(0.1 <= weights["a"] <= 0.9)
            self.assertEqual(weights["empty"], 0)
        service = coordinator(self.client)
        request = mix._request
        assert request is not None
        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(lambda _: service.CreateMix(request).id, range(4)))
        self.assertEqual(set(ids), {_decode_id(mix.id)})
        changed = mix._request
        assert changed is not None
        changed.seed += 1
        self.assertNotEqual(service.CreateMix(changed).id, _decode_id(mix.id))
        direct = service.CreateDataset(mix[0]._recipe)
        self.assertEqual(_encode_id(direct.id), mix[0].id)

    def test_exact_sampling_provenance_and_tail_profiles(self) -> None:
        sampling = pb.Sampling(
            domains=pb.Domains(field=query_pb.FIELD_OBJECT_URI),
            weights={"a": 2 / 3, "b": 1 / 3, "empty": 0},
            tokens=9,
            seed=42,
            replacement=True,
            max_epochs=2,
        )
        for length in (1, 4, 5, 20):
            for packing in (
                premixdb.Concat(separator=256),
                premixdb.Concat(separator=256, drop_remainder=False, pad_token=257),
            ):
                request = premixdb.dataset(
                    self.query.id,
                    tokenizer=premixdb.ByteTokenizer(),
                    sampling=sampling,
                    sequence_length=length,
                    packing=packing,
                )
                planned = _public_dataset_profile(
                    coordinator(self.client)._profile_dataset(request)
                )
                dataset = self.client._dataset(
                    coordinator(self.client)
                    .CreateDataset(
                        premixdb.dataset(
                            self.query.id,
                            tokenizer=premixdb.ByteTokenizer(),
                            sampling=sampling,
                            sequence_length=length,
                            packing=packing,
                        )
                    )
                    .id
                )
                self.assertEqual(planned, dataset.profile())
                self.assertEqual(dataset.profile().document_occurrences, 3)
                ids = {
                    r.id: r.source_key
                    for r in coordinator(self.client)
                    ._query_handles[_decode_id(self.query.id)]
                    .rows()
                }
                counts = dict.fromkeys(("a", "b", "empty"), 0)
                for sequence in dataset:
                    for span in sequence.spans:
                        if span.kind == pb.TokenRegion.KIND_CONTENT:
                            counts[ids[span.document_id.hex()]] += span.end - span.start
                self.assertEqual(counts, dict(planned.stratum_tokens))
                ordinals = []
                for rank in range(3):
                    topology = premixdb.Topology(rank=rank, world_size=3)
                    reader = dataset.reader(topology=topology)
                    first = next(reader, None)
                    if first:
                        ordinals.append(first.ordinal)
                    state = json.loads(json.dumps(reader.checkpoint()))
                    ordinals.extend(
                        seq.ordinal for seq in dataset.reader(topology=topology, checkpoint=state)
                    )
                self.assertEqual(sorted(ordinals), list(range(len(dataset))))

    def test_defaults_raw_requests_and_invalid_policies(self) -> None:
        service = coordinator(self.client)
        response = service.CreateMix(pb.CreateMixRequest(query_id=_decode_id(self.query.id)))
        mix = service.GetMix(pb.GetMixRequest(id=response.id)).mix
        self.assertEqual(mix.tokens, 4)
        self.assertEqual(mix.n_candidates, 3)
        self.assertEqual(len(set(mix.dataset_ids)), 3)
        self.assertTrue(mix.HasField("seed"))
        for mutate in (
            lambda r: setattr(r.domains, "field", 999),
            lambda r: setattr(r.bounds, "max_epochs", 0),
            lambda r: setattr(r, "n_candidates", 10001),
            lambda r: r.bounds.lower.update({"a": 0.9, "b": 0.9}),
            lambda r: setattr(r, "git_commit", b"a" * 20),
        ):
            request = self.mix()._request
            assert request is not None
            mutate(request)
            with self.assertRaises((ValueError, NotImplementedError)):
                service.CreateMix(request)
        with self.assertRaises(ValueError):
            service.CreateDataset(
                premixdb.dataset(
                    self.query.id, sampling=pb.Sampling(weights={"a": float("nan")}, tokens=1)
                )
            )
        with self.assertRaises(ValueError):
            premixdb.mix(self.query.id, tokens=0)
        with self.assertRaises(ValueError):
            premixdb.mix(self.query.id, size=premixdb.Tokens(1), tokens=1)

    def test_default_mix_creates_three_reproducible_datasets_for_one_domain(self) -> None:
        mixture = self.query.mix(tokenizer=premixdb.ByteTokenizer(), seed=2**64 - 1)
        self.assertEqual(len(mixture), 3)
        self.assertEqual(len({dataset.id for dataset in mixture}), 3)
        self.assertEqual([dataset._recipe.sampling.seed for dataset in mixture], [2**64 - 1, 0, 1])
        self.assertTrue(all(weights == mixture.weights[0] for weights in mixture.weights))
        self.assertTrue(
            all(dataset.status is premixdb.ExecutionStatus.PENDING for dataset in mixture)
        )
        self.assertEqual(
            self.query.mix(tokenizer=premixdb.ByteTokenizer(), seed=2**64 - 1).id,
            mixture.id,
        )
        self.assertEqual(len(self.query.mix(n_candidates=1)), 1)
        self.assertEqual(premixdb.mix(self.query.id).n_candidates, 3)

    def test_default_mix_proposes_three_distinct_weights_for_multiple_domains(self) -> None:
        mixture = self.query.mix(domains=premixdb.object.uri, tokenizer=premixdb.ByteTokenizer())
        self.assertEqual(len(mixture), 3)
        self.assertEqual(len({tuple(sorted(weights.items())) for weights in mixture.weights}), 3)
        self.assertEqual(len({dataset.id for dataset in mixture}), 3)
        self.assertEqual({dataset._recipe.sampling.seed for dataset in mixture}, {0})

    def test_assignments_no_replacement_and_pool_reuse(self) -> None:
        self.query.wait()
        handle = coordinator(self.client)._query_handles[_decode_id(self.query.id)]
        labels = {r.id: "letters" if r.source_key == "a" else "other" for r in handle.rows()}
        first = self.mix(domains=labels)
        second = self.mix(domains=labels, seed=43)
        self.assertNotEqual(first[0].id, second[0].id)
        self.assertEqual(len(coordinator(self.client)._mix_pools), 1)
        with self.assertRaises(Exception):
            self.mix(domains={})
        mixture = self.query.mix(tokenizer=premixdb.ByteTokenizer(), replacement=False)
        dataset = mixture[0].wait()
        self.assertEqual(dataset.profile().content_tokens, 7)
        self.assertEqual(dataset.profile().document_occurrences, 2)
