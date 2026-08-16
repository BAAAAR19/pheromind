import unittest

import numpy as np

from pheromind.world import FOOD, HOME, World, WorldConfig


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

    def test_open_ground_by_default(self):
        self.assertFalse(self.world.walls.any())

    def test_block_terrain_leaves_the_nest_reachable(self):
        cfg = WorldConfig(width=40, height=24, wall_style="blocks", wall_blocks=12)
        world = World.generate(cfg, np.random.default_rng(2))
        self.assertTrue(world.walls.any())
        nx, ny = world.nest
        self.assertFalse(world.walls[ny, nx])
        self.assertTrue(world.passable(np.array([nx]), np.array([ny]))[0])

    def test_ring_terrain_always_has_a_way_out(self):
        cfg = WorldConfig(width=44, height=28, wall_style="ring", ring_gaps=3)
        world = World.generate(cfg, np.random.default_rng(5))
        nx, ny = world.nest
        ring = np.abs(np.hypot(*np.meshgrid(np.arange(cfg.width) - nx,
                                            np.arange(cfg.height) - ny)) - cfg.ring_radius) < 0.7
        # The doorways are cells on the ring's line that are not solid.
        self.assertTrue((ring & ~world.walls).any())

    def test_food_never_spawns_inside_rock(self):
        cfg = WorldConfig(width=40, height=24, wall_style="blocks", wall_blocks=14)
        world = World.generate(cfg, np.random.default_rng(3))
        self.assertEqual(int(world.food[world.walls].sum()), 0)

    def test_rock_holds_no_scent(self):
        cfg = WorldConfig(width=30, height=20, wall_style="blocks", wall_blocks=6)
        world = World.generate(cfg, np.random.default_rng(1))
        world.pheromone[:] = 1.0
        world.diffuse()
        self.assertEqual(float(world.pheromone[:, world.walls].sum()), 0.0)

    def test_unknown_wall_style_is_rejected(self):
        with self.assertRaises(ValueError):
            World.generate(WorldConfig(wall_style="swamp"), np.random.default_rng(0))

    def test_out_of_bounds_counts_as_solid(self):
        far = np.array([-3, self.cfg.width + 2])
        self.assertTrue(np.all(self.world.is_wall(far, far)))
        self.assertFalse(np.any(self.world.passable(far, far)))

    def test_nest_radius(self):
        nx, ny = self.world.nest
        self.assertTrue(self.world.at_nest(np.array([nx]), np.array([ny]))[0])
        self.assertFalse(self.world.at_nest(np.array([nx + 9]), np.array([ny]))[0])


if __name__ == "__main__":
    unittest.main()
