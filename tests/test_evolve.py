import unittest
from dataclasses import replace

import numpy as np

from pheromind.brain import GENOME_SIZE, random_genome
from pheromind.colony import ColonyConfig
from pheromind.evolve import (
    EvolutionConfig,
    crossover,
    evolve,
    mutate,
    tournament_select,
)
from pheromind.world import WorldConfig

FAST_COLONY = ColonyConfig(n_ants=10, steps=45)
FAST_WORLD = WorldConfig(width=30, height=18, food_piles=3, min_pile_distance=6.0)


class TestOperators(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_tournament_favours_high_scores(self):
        scores = np.arange(20, dtype=np.float32)
        picks = [tournament_select(scores, 5, self.rng) for _ in range(400)]
        self.assertGreater(np.mean(picks), 12.0)

    def test_tournament_of_one_is_uniform(self):
        scores = np.arange(10, dtype=np.float32)
        picks = {tournament_select(scores, 1, self.rng) for _ in range(300)}
        self.assertEqual(len(picks), 10)

    def test_crossover_keeps_shape_and_stays_near_the_parents(self):
        a = np.full(GENOME_SIZE, 1.0, dtype=np.float32)
        b = np.full(GENOME_SIZE, 3.0, dtype=np.float32)
        child = crossover(a, b, self.rng)
        self.assertEqual(child.shape, (GENOME_SIZE,))
        # Blend crossover overshoots a little by design, but not wildly.
        self.assertTrue(np.all((child >= 0.4) & (child <= 3.6)))

    def test_crossover_of_identical_parents_changes_nothing(self):
        a = random_genome(self.rng)
        np.testing.assert_allclose(crossover(a, a.copy(), self.rng), a, atol=1e-5)

    def test_mutation_touches_some_genes_but_not_all(self):
        cfg = EvolutionConfig(mutation_rate=0.2)
        parent = random_genome(self.rng)
        child = mutate(parent, cfg, self.rng)
        changed = int(np.sum(child != parent))
        self.assertGreater(changed, 0)
        self.assertLess(changed, GENOME_SIZE)

    def test_mutation_does_not_modify_the_parent(self):
        parent = random_genome(self.rng)
        before = parent.copy()
        mutate(parent, EvolutionConfig(mutation_rate=1.0), self.rng)
        np.testing.assert_array_equal(parent, before)

    def test_mutation_stays_inside_the_weight_bounds(self):
        genome = np.full(GENOME_SIZE, 5.9, dtype=np.float32)
        cfg = EvolutionConfig(mutation_rate=1.0, mutation_scale=5.0)
        for _ in range(20):
            genome = mutate(genome, cfg, self.rng)
        self.assertLessEqual(float(np.abs(genome).max()), 6.0)


class TestEvolve(unittest.TestCase):
    def _run(self, **overrides):
        # Serial by default: these populations are far too small to repay the
        # cost of spawning a pool, and one test below covers the parallel path.
        cfg = EvolutionConfig(
            population=6,
            generations=3,
            worlds_per_genome=1,
            seed=1,
            **{"workers": 1, **overrides},
        )
        return evolve(cfg, FAST_COLONY, FAST_WORLD)

    def test_returns_a_usable_genome_and_full_history(self):
        champion, history = self._run()
        self.assertEqual(champion.shape, (GENOME_SIZE,))
        self.assertEqual(len(history), 3)
        self.assertEqual([h.generation for h in history], [0, 1, 2])

    def test_reruns_with_the_same_seed_match(self):
        a, _ = self._run()
        b, _ = self._run()
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_diverge(self):
        a, _ = self._run()
        cfg = EvolutionConfig(
            population=6, generations=3, worlds_per_genome=1, seed=99, workers=1
        )
        b, _ = evolve(cfg, FAST_COLONY, FAST_WORLD)
        self.assertFalse(np.array_equal(a, b))

    def test_elites_never_let_the_best_score_collapse(self):
        cfg = EvolutionConfig(
            population=8, generations=4, worlds_per_genome=1, seed=2, elites=2, workers=1
        )
        _, history = evolve(cfg, FAST_COLONY, FAST_WORLD)
        bests = [h.best for h in history]
        # Worlds change each generation, so demand no collapse rather than
        # strict monotonicity.
        self.assertGreaterEqual(max(bests[1:]), 0.5 * bests[0])

    def test_report_line_is_printable(self):
        _, history = self._run()
        self.assertIn("gen", history[0].line())

    def test_parallel_scoring_matches_serial_exactly(self):
        # The pool exists to save wall clock, not to change answers. map()
        # preserves order, so results must be bit-for-bit identical.
        cfg = EvolutionConfig(population=6, generations=2, worlds_per_genome=2, seed=4)
        serial, _ = evolve(replace(cfg, workers=1), FAST_COLONY, FAST_WORLD)
        parallel, _ = evolve(replace(cfg, workers=3), FAST_COLONY, FAST_WORLD)
        np.testing.assert_array_equal(serial, parallel)

    def test_worker_count_resolves_sensibly(self):
        self.assertEqual(EvolutionConfig(workers=5).resolved_workers(), 5)
        # Auto never asks for more workers than there are jobs to hand out.
        tiny = EvolutionConfig(workers=0, population=2, worlds_per_genome=1)
        self.assertLessEqual(tiny.resolved_workers(), 2)
        self.assertGreaterEqual(tiny.resolved_workers(), 1)


if __name__ == "__main__":
    unittest.main()
