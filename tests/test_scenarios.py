import unittest

import numpy as np

from pheromind import scenarios
from pheromind.brain import Brain, random_genome
from pheromind.colony import Colony
from pheromind.world import World


class TestScenarios(unittest.TestCase):
    def test_default_scenario_exists(self):
        self.assertIn(scenarios.DEFAULT_SCENARIO, scenarios.SCENARIOS)

    def test_every_scenario_is_named_after_its_key(self):
        for key, scenario in scenarios.SCENARIOS.items():
            self.assertEqual(key, scenario.name)

    def test_unknown_scenario_lists_the_real_ones(self):
        with self.assertRaises(KeyError) as caught:
            scenarios.get("atlantis")
        self.assertIn("classic", str(caught.exception))

    def test_describe_mentions_every_scenario(self):
        text = scenarios.describe()
        for name in scenarios.SCENARIOS:
            self.assertIn(name, text)

    def test_every_scenario_builds_a_runnable_world(self):
        # A scenario whose piles are unreachable or whose nest is sealed would
        # be a silent dud, so actually run each one for a few ticks.
        for name, scenario in scenarios.SCENARIOS.items():
            with self.subTest(scenario=name):
                rng = np.random.default_rng(0)
                world = World.generate(scenario.world, rng)
                self.assertGreater(world.food_at_start, 0, f"{name} has no food")

                nx, ny = world.nest
                self.assertFalse(world.walls[ny, nx], f"{name} sealed the nest")

                colony = Colony.spawn(
                    world, Brain.from_genome(random_genome(rng)), scenario.colony, rng
                )
                colony.run(40)
                self.assertFalse(world.walls[colony.y, colony.x].any())

    def test_scenarios_actually_differ(self):
        shapes = {
            (s.world.width, s.world.height, s.world.food_piles,
             s.world.wall_style, s.colony.n_ants, s.colony.steps)
            for s in scenarios.SCENARIOS.values()
        }
        self.assertEqual(len(shapes), len(scenarios.SCENARIOS))


if __name__ == "__main__":
    unittest.main()
