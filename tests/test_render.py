import re
import unittest

import numpy as np

from pheromind.brain import Brain, random_genome
from pheromind.colony import Colony, ColonyConfig
from pheromind.render import duel_frame, frame, grid_lines, grid_width, legend, status
from pheromind.world import World, WorldConfig

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    return ANSI.sub("", text)


def make_colony(seed=0, **world_kwargs):
    rng = np.random.default_rng(seed)
    cfg = WorldConfig(width=32, height=16, food_piles=3, **world_kwargs)
    world = World.generate(cfg, rng)
    colony = Colony.spawn(world, Brain.from_genome(random_genome(rng)),
                          ColonyConfig(n_ants=12), rng)
    colony.run(40)
    return colony


class TestRender(unittest.TestCase):
    def test_every_grid_line_has_the_same_visual_width(self):
        # Side-by-side rendering depends on this exactly holding.
        colony = make_colony()
        widths = {len(plain(line)) for line in grid_lines(colony)}
        self.assertEqual(widths, {grid_width(colony)})

    def test_grid_has_a_border_row_top_and_bottom(self):
        colony = make_colony()
        lines = grid_lines(colony)
        self.assertEqual(len(lines), colony.world.cfg.height + 2)
        self.assertTrue(plain(lines[0]).startswith("+"))
        self.assertTrue(plain(lines[-1]).startswith("+"))

    def test_walls_are_drawn(self):
        colony = make_colony(wall_style="blocks", wall_blocks=8)
        self.assertIn("▓", plain("\n".join(grid_lines(colony))))

    def test_nest_is_drawn(self):
        self.assertIn("@", plain("\n".join(grid_lines(make_colony()))))

    def test_frame_carries_the_status_line(self):
        colony = make_colony()
        self.assertIn("delivered", plain(frame(colony)))
        self.assertIn("tick", plain(status(colony)))

    def test_duel_frame_shows_both_sides(self):
        left, right = make_colony(seed=1), make_colony(seed=1)
        text = plain(duel_frame(left, right, "champion", "untrained"))
        self.assertIn("champion", text)
        self.assertIn("untrained", text)
        # Both maps on one row means two borders on the same line.
        border_row = [ln for ln in text.splitlines() if ln.strip().startswith("+")][0]
        self.assertEqual(border_row.count("+"), 4)

    def test_legend_is_plain_enough_to_read(self):
        self.assertIn("nest", plain(legend()))


if __name__ == "__main__":
    unittest.main()
