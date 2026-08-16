"""A swarm of ants sharing one brain, stepped as arrays.

Every ant in a colony runs identical weights. The only reason they behave
differently is that they stand in different places and smell different things —
which is exactly the constraint that forces evolution to invent trail-following
instead of hard-coding a route.

One tick, for every ant at once:

1. read three antennae (front-left, front, front-right)
2. think, and turn by -1 / 0 / +1 eighths of a circle
3. lay pheromone in the channel implied by whether it's carrying
4. step one cell forward, bouncing off the walls
5. pick up food if standing on some, or drop it if home
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .brain import Brain, N_SENSES
from .world import DIRECTIONS, FOOD, HOME, World, WorldConfig

ANTENNA_RANGE = 2  # how many cells ahead each antenna reaches


@dataclass
class ColonyConfig:
    n_ants: int = 45
    steps: int = 420
    lay_rate: float = 0.35  # pheromone laid at full output
    lay_threshold: float = 0.15  # below this the ant lays nothing at all
    wall_bounce: int = 4  # eighths of a circle to turn when hitting an edge


@dataclass
class Colony:
    world: World
    brain: Brain
    cfg: ColonyConfig
    x: np.ndarray  # (n,) int32
    y: np.ndarray  # (n,) int32
    heading: np.ndarray  # (n,) int32 in 0..7
    carrying: np.ndarray  # (n,) bool
    delivered: int = 0
    picked_up: int = 0
    ticks: int = 0

    @classmethod
    def spawn(
        cls,
        world: World,
        brain: Brain,
        cfg: ColonyConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> "Colony":
        cfg = cfg or ColonyConfig()
        rng = rng or np.random.default_rng()
        n = cfg.n_ants
        nx, ny = world.nest
        return cls(
            world=world,
            brain=brain,
            cfg=cfg,
            x=np.full(n, nx, dtype=np.int32),
            y=np.full(n, ny, dtype=np.int32),
            heading=rng.integers(0, 8, size=n).astype(np.int32),
            carrying=np.zeros(n, dtype=bool),
        )

    # -- perception ------------------------------------------------------

    def antenna_positions(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Cells under the left, centre and right antennae of every ant."""
        spots = []
        for offset in (-1, 0, 1):
            d = DIRECTIONS[(self.heading + offset) % 8]
            spots.append((self.x + d[:, 0] * ANTENNA_RANGE, self.y + d[:, 1] * ANTENNA_RANGE))
        return spots

    def senses(self) -> np.ndarray:
        w = self.world
        spots = self.antenna_positions()

        columns = [self.carrying.astype(np.float32)]
        columns += [w.sample(FOOD, sx, sy) for sx, sy in spots]
        columns += [w.sample(HOME, sx, sy) for sx, sy in spots]
        # Squash food counts so a big pile doesn't saturate the layer.
        columns += [np.tanh(w.food_at(sx, sy) / 3.0) for sx, sy in spots]

        # Ants keep a rough bearing home the way real ones do, by path
        # integration; without it a fresh random colony would never once
        # stumble back to the nest and evolution would have no signal at all.
        dx = w.nest[0] - self.x
        dy = w.nest[1] - self.y
        distance = np.hypot(dx, dy)
        bearing = np.arctan2(dy, dx) - self.heading * (np.pi / 4.0)
        diagonal = float(np.hypot(w.cfg.width, w.cfg.height))
        columns += [np.sin(bearing), np.cos(bearing), distance / diagonal]

        senses = np.stack([np.asarray(c, dtype=np.float32) for c in columns], axis=1)
        assert senses.shape[1] == N_SENSES, senses.shape
        return senses

    # -- one tick --------------------------------------------------------

    def step(self) -> None:
        turn, lay = self.brain.think(self.senses())
        self.heading = (self.heading + turn) % 8

        # Searching ants scent the way home; loaded ants scent the way to food.
        wants_to_lay = lay > self.cfg.lay_threshold
        if wants_to_lay.any():
            channel = np.where(self.carrying, FOOD, HOME).astype(np.int32)
            self.world.deposit(
                channel[wants_to_lay],
                self.x[wants_to_lay],
                self.y[wants_to_lay],
                lay[wants_to_lay] * self.cfg.lay_rate,
            )

        step = DIRECTIONS[self.heading]
        nx = self.x + step[:, 0]
        ny = self.y + step[:, 1]

        # An ant that walks into a wall turns hard instead of leaving the map.
        blocked = ~self.world.in_bounds(nx, ny)
        if blocked.any():
            self.heading[blocked] = (self.heading[blocked] + self.cfg.wall_bounce) % 8
            nx[blocked] = self.x[blocked]
            ny[blocked] = self.y[blocked]

        self.x, self.y = nx.astype(np.int32), ny.astype(np.int32)

        self._handle_food()
        self.world.diffuse()
        self.ticks += 1

    def _handle_food(self) -> None:
        searching = ~self.carrying
        if searching.any():
            idx = np.flatnonzero(searching)
            took = self.world.take_food(self.x[idx], self.y[idx])
            self.carrying[idx[took]] = True
            self.picked_up += int(took.sum())

        home = self.carrying & self.world.at_nest(self.x, self.y)
        if home.any():
            n = int(home.sum())
            self.carrying[home] = False
            self.delivered += n
            self.world.delivered += n
            # Turn deliverers around so they head back out, not into the nest.
            self.heading[home] = (self.heading[home] + 4) % 8

    def run(self, steps: int | None = None) -> int:
        for _ in range(steps if steps is not None else self.cfg.steps):
            self.step()
        return self.delivered


def evaluate(
    genome: np.ndarray,
    seed: int,
    colony_cfg: ColonyConfig | None = None,
    world_cfg: WorldConfig | None = None,
) -> float:
    """Score one genome on one world. Higher is better.

    Delivery is what actually matters, so it dominates. A crumb of credit for
    merely *finding* food keeps the very first generations from being an
    undifferentiated wall of zeros with nothing for selection to grip.
    """
    rng = np.random.default_rng(seed)
    world = World.generate(world_cfg, rng)
    colony = Colony.spawn(world, Brain.from_genome(genome), colony_cfg, rng)
    colony.run()
    return float(colony.delivered) + 0.05 * float(colony.picked_up)
