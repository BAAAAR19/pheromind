import unittest

import numpy as np

from pheromind.brain import N_SENSES, Brain, random_genome
from pheromind.colony import Colony, ColonyConfig, evaluate, run_episode
from pheromind.world import World, WorldConfig


def make_colony(n_ants=12, seed=0, **world_kwargs):
    rng = np.random.default_rng(seed)
    cfg = WorldConfig(width=36, height=20, food_piles=3, **world_kwargs)
    world = World.generate(cfg, rng)
    brain = Brain.from_genome(random_genome(rng))
    return Colony.spawn(world, brain, ColonyConfig(n_ants=n_ants, steps=60), rng)


class TestColony(unittest.TestCase):
    def test_ants_start_at_the_nest(self):
        colony = make_colony()
        nx, ny = colony.world.nest
        self.assertTrue(np.all(colony.x == nx))
        self.assertTrue(np.all(colony.y == ny))
        self.assertFalse(colony.carrying.any())

    def test_senses_have_the_shape_the_brain_expects(self):
        colony = make_colony()
        senses = colony.senses()
        self.assertEqual(senses.shape, (len(colony.x), N_SENSES))
        self.assertTrue(np.all(np.isfinite(senses)))

    def test_ants_never_leave_the_grid(self):
        colony = make_colony(n_ants=40)
        h, w = colony.world.shape
        for _ in range(200):
            colony.step()
            self.assertTrue(np.all((colony.x >= 0) & (colony.x < w)))
            self.assertTrue(np.all((colony.y >= 0) & (colony.y < h)))

    def test_ants_never_stand_inside_rock(self):
        rng = np.random.default_rng(6)
        cfg = WorldConfig(width=40, height=24, food_piles=3,
                          wall_style="blocks", wall_blocks=12)
        world = World.generate(cfg, rng)
        colony = Colony.spawn(world, Brain.from_genome(random_genome(rng)),
                              ColonyConfig(n_ants=30), rng)
        for _ in range(200):
            colony.step()
            self.assertFalse(world.walls[colony.y, colony.x].any())

    def test_food_is_conserved(self):
        colony = make_colony(n_ants=30)
        colony.run(150)
        world = colony.world
        accounted = world.food_remaining() + colony.delivered + int(colony.carrying.sum())
        self.assertEqual(accounted, world.food_at_start)

    def test_carrying_ants_lay_the_food_channel(self):
        from pheromind.world import FOOD, HOME

        colony = make_colony(n_ants=8)
        colony.carrying[:] = True
        colony.world.pheromone[:] = 0.0
        # Park the ants away from the nest so nobody delivers and drops the flag.
        colony.x[:] = 2
        colony.y[:] = 2
        colony.step()
        self.assertGreater(float(colony.world.pheromone[FOOD].sum()), 0.0)
        self.assertEqual(float(colony.world.pheromone[HOME].sum()), 0.0)

    def test_delivering_food_scores_and_frees_the_ant(self):
        colony = make_colony(n_ants=4)
        colony.carrying[:] = True
        nx, ny = colony.world.nest
        colony.x[:] = nx
        colony.y[:] = ny
        colony.step()
        self.assertEqual(colony.delivered, 4)
        self.assertFalse(colony.carrying.any())

    def test_blinding_zeroes_exactly_the_named_senses(self):
        from pheromind.brain import SENSE_GROUPS

        colony = make_colony(n_ants=10)
        colony.run(20)
        seeing = colony.senses()

        colony.cfg = ColonyConfig(n_ants=10, blind=SENSE_GROUPS["home_trail"])
        blinded = colony.senses()

        for i in SENSE_GROUPS["home_trail"]:
            self.assertTrue(np.all(blinded[:, i] == 0.0))
        # Untouched senses must survive intact, or ablation proves nothing.
        for i in SENSE_GROUPS["nest_bearing"]:
            np.testing.assert_allclose(blinded[:, i], seeing[:, i])

    def test_sense_groups_cover_every_input_exactly_once(self):
        from pheromind.brain import N_SENSES, SENSE_GROUPS

        covered = sorted(i for idx in SENSE_GROUPS.values() for i in idx)
        self.assertEqual(covered, list(range(N_SENSES)))

    def test_run_episode_hands_back_a_finished_colony(self):
        genome = random_genome(np.random.default_rng(2))
        colony = run_episode(genome, seed=1, colony_cfg=ColonyConfig(n_ants=8, steps=30))
        self.assertEqual(colony.ticks, 30)
        self.assertGreaterEqual(colony.delivered, 0)

    def test_evaluate_is_deterministic_for_a_seed(self):
        genome = random_genome(np.random.default_rng(3))
        cfg = ColonyConfig(n_ants=10, steps=40)
        a = evaluate(genome, seed=5, colony_cfg=cfg)
        b = evaluate(genome, seed=5, colony_cfg=cfg)
        self.assertEqual(a, b)

    def test_evaluate_is_never_negative(self):
        cfg = ColonyConfig(n_ants=10, steps=40)
        rng = np.random.default_rng(9)
        for _ in range(5):
            self.assertGreaterEqual(evaluate(random_genome(rng), seed=1, colony_cfg=cfg), 0.0)


if __name__ == "__main__":
    unittest.main()
