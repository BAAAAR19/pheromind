import unittest

import numpy as np

from antmind.world import FOOD, HOME, World, WorldConfig


class TestWorld(unittest.TestCase):
    def setUp(self):
        self.cfg = WorldConfig(width=40, height=24, food_piles=4)
        self.world = World.generate(self.cfg, np.random.default_rng(0))

    def test_generates_food_away_from_the_nest(self):
        self.assertGreater(self.world.food_at_start, 0)
        nx, ny = self.world.nest
        self.assertEqual(self.world.food[ny, nx], 0)

    def test_sampling_off_grid_reads_as_empty(self):
        far = np.array([-5, self.cfg.width + 5])
        self.assertTrue(np.all(self.world.sample(FOOD, far, far) == 0.0))
        self.assertTrue(np.all(self.world.food_at(far, far) == 0.0))

    def test_deposits_from_several_ants_on_one_cell_accumulate(self):
        x, y = np.array([5, 5, 5]), np.array([6, 6, 6])
        self.world.deposit(np.array([HOME, HOME, HOME]), x, y, np.array([0.1, 0.1, 0.1]))
        self.assertAlmostEqual(float(self.world.pheromone[HOME, 6, 5]), 0.3, places=5)

    def test_pheromone_is_capped(self):
        x, y = np.array([3]), np.array([3])
        for _ in range(50):
            self.world.deposit(np.array([FOOD]), x, y, np.array([0.5]))
        self.assertLessEqual(float(self.world.pheromone.max()), self.cfg.max_pheromone)

    def test_trails_fade_to_nothing(self):
        self.world.deposit(np.array([FOOD]), np.array([10]), np.array([10]), np.array([1.0]))
        for _ in range(400):
            self.world.diffuse()
        self.assertEqual(float(self.world.pheromone.sum()), 0.0)

    def test_diffusion_spreads_to_neighbours(self):
        self.world.deposit(np.array([FOOD]), np.array([10]), np.array([10]), np.array([1.0]))
        self.world.diffuse()
        self.assertGreater(float(self.world.pheromone[FOOD, 10, 11]), 0.0)
        self.assertLess(float(self.world.pheromone[FOOD, 10, 10]), 1.0)

    def test_two_ants_cannot_take_the_same_last_crumb(self):
        self.world.food[:] = 0
        self.world.food[4, 4] = 1
        took = self.world.take_food(np.array([4, 4]), np.array([4, 4]))
        self.assertEqual(list(took), [True, False])
        self.assertEqual(self.world.food_remaining(), 0)

    def test_nest_radius(self):
        nx, ny = self.world.nest
        self.assertTrue(self.world.at_nest(np.array([nx]), np.array([ny]))[0])
        self.assertFalse(self.world.at_nest(np.array([nx + 9]), np.array([ny]))[0])


if __name__ == "__main__":
    unittest.main()
