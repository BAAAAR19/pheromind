"""Named worlds worth evolving against.

Each scenario is a different argument about what makes foraging hard, and they
select for visibly different colonies. Breed against ``famine`` and you get
wide-ranging scouts; breed against ``feast`` and you get a colony that barely
bothers with trails, because it never needs one.

    python -m pheromind list
    python -m pheromind train --scenario gauntlet
    python -m pheromind watch --scenario gauntlet --genome champions/gauntlet.json
"""

from __future__ import annotations

from dataclasses import dataclass

from .colony import ColonyConfig
from .world import WorldConfig


@dataclass(frozen=True)
class Scenario:
    name: str
    blurb: str
    world: WorldConfig
    colony: ColonyConfig


SCENARIOS: dict[str, Scenario] = {
    "classic": Scenario(
        name="classic",
        blurb="Open ground, five piles at a fair distance. The baseline.",
        world=WorldConfig(),
        colony=ColonyConfig(),
    ),
    "famine": Scenario(
        name="famine",
        blurb="Two small piles, pushed out to the edges. Rewards scouts.",
        world=WorldConfig(
            food_piles=2,
            pile_radius=1.6,
            food_per_cell=4,
            min_pile_distance=16.0,
            max_pile_distance=0.55,
        ),
        colony=ColonyConfig(steps=520),
    ),
    "feast": Scenario(
        name="feast",
        blurb="Food everywhere and close by. Trails barely earn their keep.",
        world=WorldConfig(
            food_piles=9,
            pile_radius=3.0,
            food_per_cell=8,
            min_pile_distance=5.0,
        ),
        colony=ColonyConfig(),
    ),
    "boulders": Scenario(
        name="boulders",
        blurb="Open plan, but strewn with rock the colony has to route around.",
        world=WorldConfig(wall_style="blocks", wall_blocks=9),
        colony=ColonyConfig(),
    ),
    "gauntlet": Scenario(
        name="gauntlet",
        blurb="The nest is walled in. Three doorways, and all the food outside.",
        world=WorldConfig(
            wall_style="ring",
            ring_radius=8.0,
            ring_gaps=3,
            ring_gap_width=1.8,
            min_pile_distance=13.0,
            max_pile_distance=0.55,
        ),
        colony=ColonyConfig(steps=520),
    ),
    "keyhole": Scenario(
        name="keyhole",
        blurb="One single gap in the wall. Brutal. Trails or nothing.",
        world=WorldConfig(
            wall_style="ring",
            ring_radius=7.0,
            ring_gaps=1,
            ring_gap_width=1.8,
            food_piles=4,
            min_pile_distance=12.0,
            max_pile_distance=0.55,
        ),
        colony=ColonyConfig(n_ants=60, steps=600),
    ),
    "sprawl": Scenario(
        name="sprawl",
        blurb="A big map and a short clock. Every wasted step costs.",
        world=WorldConfig(width=78, height=36, food_piles=7, max_pile_distance=0.5),
        colony=ColonyConfig(n_ants=60, steps=600),
    ),
}

DEFAULT_SCENARIO = "classic"


def get(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError:
        known = ", ".join(SCENARIOS)
        raise KeyError(f"unknown scenario {name!r}; try one of: {known}") from None


def describe() -> str:
    width = max(len(n) for n in SCENARIOS)
    lines = []
    for scenario in SCENARIOS.values():
        world, colony = scenario.world, scenario.colony
        terrain = world.wall_style if world.wall_style != "none" else "open"
        lines.append(
            f"  {scenario.name:<{width}}  {scenario.blurb}\n"
            f"  {'':<{width}}  {world.width}x{world.height}, {world.food_piles} piles, "
            f"{colony.n_ants} ants, {colony.steps} ticks, terrain: {terrain}"
        )
    return "\n".join(lines)
