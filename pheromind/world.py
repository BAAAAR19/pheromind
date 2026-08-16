"""The ground the ants walk on.

The world is a grid holding food piles, one nest, and two pheromone fields:

* ``HOME`` — dropped by ants that are *searching*, so it points back to the nest.
* ``FOOD`` — dropped by ants that are *carrying*, so it points back to a pile.

Neither field is designed; ants only get an output that says "lay some now", and
which field it lands in depends on whether the ant is loaded. What the trails
end up meaning is entirely evolution's problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HOME = 0
FOOD = 1

# Eight compass headings, indexed 0..7 going counter-clockwise from east.
DIRECTIONS = np.array(
    [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)],
    dtype=np.int32,
)


@dataclass
class WorldConfig:
    width: int = 56
    height: int = 30
    food_piles: int = 5
    pile_radius: float = 2.2
    food_per_cell: int = 6
    nest_radius: float = 2.0
    # Fraction of pheromone surviving each tick, and how much of what survives
    # bleeds into the four neighbouring cells.
    evaporation: float = 0.96
    diffusion: float = 0.18
    max_pheromone: float = 1.0
    # Piles are kept in an annulus around the nest so food is never trivially
    # underfoot but never unreachable either.
    min_pile_distance: float = 9.0
    max_pile_distance: float = 0.42  # as a fraction of the map diagonal
    # Terrain. "none" is open ground; "blocks" scatters rectangular boulders;
    # "ring" walls the nest in behind a broken circle the colony must find a
    # way through. Walls block movement *and* scent, so a trail has to route
    # around them the same way the ants do.
    wall_style: str = "none"
    wall_blocks: int = 7
    ring_radius: float = 8.0
    ring_gaps: int = 3
    ring_gap_width: float = 2.0


@dataclass
class World:
    """Mutable state of one foraging episode."""

    cfg: WorldConfig
    food: np.ndarray  # (h, w) int32, units of food left in each cell
    pheromone: np.ndarray  # (2, h, w) float32
    walls: np.ndarray  # (h, w) bool, True where nothing may pass
    nest: tuple[int, int]  # (x, y)
    food_at_start: int = 0
    delivered: int = 0

    # -- construction ----------------------------------------------------

    @classmethod
    def generate(cls, cfg: WorldConfig | None = None, rng: np.random.Generator | None = None) -> "World":
        cfg = cfg or WorldConfig()
        rng = rng or np.random.default_rng()

        food = np.zeros((cfg.height, cfg.width), dtype=np.int32)
        nest = (cfg.width // 2, cfg.height // 2)

        yy, xx = np.mgrid[0 : cfg.height, 0 : cfg.width]
        diagonal = float(np.hypot(cfg.width, cfg.height))
        far_limit = cfg.max_pile_distance * diagonal

        walls = _build_walls(cfg, nest, xx, yy, rng)

        for _ in range(cfg.food_piles):
            for _attempt in range(200):
                px = int(rng.integers(2, cfg.width - 2))
                py = int(rng.integers(2, cfg.height - 2))
                distance = np.hypot(px - nest[0], py - nest[1])
                if cfg.min_pile_distance <= distance <= far_limit and not walls[py, px]:
                    break
            blob = np.hypot(xx - px, yy - py) <= cfg.pile_radius
            food[blob] = cfg.food_per_cell

        # Never bury the nest under a pile, and never bury food under rock.
        food[np.hypot(xx - nest[0], yy - nest[1]) <= cfg.nest_radius] = 0
        food[walls] = 0

        return cls(
            cfg=cfg,
            food=food,
            pheromone=np.zeros((2, cfg.height, cfg.width), dtype=np.float32),
            walls=walls,
            nest=nest,
            food_at_start=int(food.sum()),
        )

    # -- queries ---------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return self.cfg.height, self.cfg.width

    def in_bounds(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x >= 0) & (x < self.cfg.width) & (y >= 0) & (y < self.cfg.height)

    def is_wall(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """True where a cell is solid rock. Off-grid counts as solid."""
        inside = self.in_bounds(x, y)
        cx, cy = self.clip(x, y)
        return np.where(inside, self.walls[cy, cx], True)

    def passable(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.in_bounds(x, y) & ~self.is_wall(x, y)

    def clip(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.clip(x, 0, self.cfg.width - 1),
            np.clip(y, 0, self.cfg.height - 1),
        )

    def sample(self, channel: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Read a pheromone channel at integer coordinates; off-grid reads as 0."""
        inside = self.in_bounds(x, y)
        cx, cy = self.clip(x, y)
        return np.where(inside, self.pheromone[channel, cy, cx], 0.0).astype(np.float32)

    def food_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        inside = self.in_bounds(x, y)
        cx, cy = self.clip(x, y)
        return np.where(inside, self.food[cy, cx], 0).astype(np.float32)

    def at_nest(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.hypot(x - self.nest[0], y - self.nest[1]) <= self.cfg.nest_radius

    # -- mutation --------------------------------------------------------

    def deposit(self, channel: np.ndarray, x: np.ndarray, y: np.ndarray, amount: np.ndarray) -> None:
        """Add pheromone. ``np.add.at`` so several ants on one cell all count."""
        cx, cy = self.clip(x, y)
        np.add.at(self.pheromone, (channel, cy, cx), amount)
        np.clip(self.pheromone, 0.0, self.cfg.max_pheromone, out=self.pheromone)

    def take_food(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Remove one unit of food per ant standing on a stocked cell.

        Returns a bool mask of which ants actually picked something up. Ants are
        served in array order so two ants cannot split the last crumb.
        """
        cx, cy = self.clip(x, y)
        took = np.zeros(len(cx), dtype=bool)
        for i in range(len(cx)):
            if self.food[cy[i], cx[i]] > 0:
                self.food[cy[i], cx[i]] -= 1
                took[i] = True
        return took

    def diffuse(self) -> None:
        """Evaporate, then let a little of each cell bleed into its neighbours."""
        p = self.pheromone
        p *= self.cfg.evaporation

        share = self.cfg.diffusion
        if share > 0.0:
            spread = np.zeros_like(p)
            spread[:, 1:, :] += p[:, :-1, :]
            spread[:, :-1, :] += p[:, 1:, :]
            spread[:, :, 1:] += p[:, :, :-1]
            spread[:, :, :-1] += p[:, :, 1:]
            p *= 1.0 - share
            p += (share / 4.0) * spread

        # Rock does not hold scent, so trails bend around obstacles instead of
        # tunnelling through them.
        p[:, self.walls] = 0.0

        # Anything below this is a rounding artefact, not a trail.
        p[p < 1e-4] = 0.0

    def food_remaining(self) -> int:
        return int(self.food.sum())


def _build_walls(
    cfg: WorldConfig,
    nest: tuple[int, int],
    xx: np.ndarray,
    yy: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Lay out terrain for the configured wall style."""
    walls = np.zeros((cfg.height, cfg.width), dtype=bool)

    if cfg.wall_style == "blocks":
        for _ in range(cfg.wall_blocks):
            bw = int(rng.integers(2, 8))
            bh = int(rng.integers(2, 5))
            bx = int(rng.integers(0, max(1, cfg.width - bw)))
            by = int(rng.integers(0, max(1, cfg.height - bh)))
            walls[by : by + bh, bx : bx + bw] = True

    elif cfg.wall_style == "ring":
        distance = np.hypot(xx - nest[0], yy - nest[1])
        walls |= np.abs(distance - cfg.ring_radius) < 0.7
        # Punch a few doorways, otherwise the colony is simply sealed in.
        for angle in rng.uniform(0.0, 2.0 * np.pi, size=cfg.ring_gaps):
            gx = nest[0] + cfg.ring_radius * np.cos(angle)
            gy = nest[1] + cfg.ring_radius * np.sin(angle)
            walls[np.hypot(xx - gx, yy - gy) <= cfg.ring_gap_width] = False

    elif cfg.wall_style != "none":
        raise ValueError(f"unknown wall_style {cfg.wall_style!r}")

    # Whatever the style, the ants must be able to leave home.
    walls[np.hypot(xx - nest[0], yy - nest[1]) <= cfg.nest_radius + 1.0] = False
    return walls
