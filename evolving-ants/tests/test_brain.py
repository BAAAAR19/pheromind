import tempfile
import unittest
from pathlib import Path

import numpy as np

from antmind.brain import (
    GENOME_SIZE,
    N_ACTIONS,
    N_HIDDEN,
    N_SENSES,
    Brain,
    load_genome,
    random_genome,
    save_genome,
)


class TestBrain(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.genome = random_genome(self.rng)

    def test_genome_size_matches_the_architecture(self):
        expected = N_SENSES * N_HIDDEN + N_HIDDEN + N_HIDDEN * N_ACTIONS + N_ACTIONS
        self.assertEqual(GENOME_SIZE, expected)
        self.assertEqual(self.genome.shape, (GENOME_SIZE,))

    def test_unpacking_uses_every_gene_exactly_once(self):
        brain = Brain.from_genome(self.genome)
        total = brain.w1.size + brain.b1.size + brain.w2.size + brain.b2.size
        self.assertEqual(total, GENOME_SIZE)

    def test_wrong_sized_genome_is_rejected(self):
        with self.assertRaises(ValueError):
            Brain.from_genome(np.zeros(7, dtype=np.float32))

    def test_think_returns_legal_turns_and_bounded_lay(self):
        brain = Brain.from_genome(self.genome)
        senses = self.rng.normal(size=(32, N_SENSES)).astype(np.float32)
        turn, lay = brain.think(senses)

        self.assertEqual(turn.shape, (32,))
        self.assertTrue(set(np.unique(turn)).issubset({-1, 0, 1}))
        self.assertTrue(np.all((lay >= 0.0) & (lay <= 1.0)))

    def test_identical_senses_give_identical_decisions(self):
        brain = Brain.from_genome(self.genome)
        senses = np.tile(self.rng.normal(size=(1, N_SENSES)), (4, 1)).astype(np.float32)
        turn, lay = brain.think(senses)
        self.assertEqual(len(set(turn.tolist())), 1)
        self.assertEqual(len(set(np.round(lay, 6).tolist())), 1)

    def test_genome_survives_a_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "champ.json"
            save_genome(path, self.genome, {"generations": 3})
            restored, meta = load_genome(path)

        self.assertEqual(meta["generations"], 3)
        np.testing.assert_allclose(restored, self.genome, atol=1e-5)

    def test_loading_a_mismatched_genome_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "short.json"
            save_genome(path, self.genome[:20])
            with self.assertRaises(ValueError):
                load_genome(path)


if __name__ == "__main__":
    unittest.main()
