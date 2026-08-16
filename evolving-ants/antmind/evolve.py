"""The part that does the learning — without ever computing a gradient.

Standard generational GA: score every genome, keep a couple of elites intact,
then fill the next population from tournament winners via blend crossover and
Gaussian mutation.

Two details matter more than the rest:

* **Fresh worlds every generation.** Scoring on fixed maps breeds colonies that
  memorise where the food was. Re-rolling the seeds each generation means the
  only thing that survives is a strategy.
* **Several worlds per genome.** One map is mostly luck. Averaging a handful
  keeps a fluke from winning the gene pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from .brain import GENOME_SIZE, random_genome
from .colony import ColonyConfig, evaluate
from .world import WorldConfig


@dataclass
class EvolutionConfig:
    population: int = 32
    generations: int = 25
    elites: int = 2
    tournament: int = 3
    mutation_rate: float = 0.18  # fraction of genes touched
    mutation_scale: float = 0.28  # size of the nudge
    crossover_rate: float = 0.7
    worlds_per_genome: int = 2
    seed: int = 0


@dataclass
class GenerationReport:
    generation: int
    best: float
    mean: float
    median: float
    best_genome: np.ndarray = field(repr=False)

    def line(self) -> str:
        return (
            f"gen {self.generation:>3} | best {self.best:7.2f} | "
            f"mean {self.mean:7.2f} | median {self.median:7.2f}"
        )


def tournament_select(scores: np.ndarray, k: int, rng: np.random.Generator) -> int:
    contenders = rng.integers(0, len(scores), size=k)
    return int(contenders[np.argmax(scores[contenders])])


def crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Blend crossover: each gene lands somewhere on the line between parents.

    Uniform swapping tends to chop a weight matrix into two halves that were
    never meant to meet. Interpolating keeps a child near both parents.
    """
    mix = rng.uniform(-0.25, 1.25, size=a.shape).astype(np.float32)
    return (mix * a + (1.0 - mix) * b).astype(np.float32)


def mutate(genome: np.ndarray, cfg: EvolutionConfig, rng: np.random.Generator) -> np.ndarray:
    child = genome.copy()
    hits = rng.random(child.shape) < cfg.mutation_rate
    child[hits] += rng.normal(0.0, cfg.mutation_scale, size=int(hits.sum())).astype(np.float32)
    return np.clip(child, -6.0, 6.0)


def score_population(
    population: Iterable[np.ndarray],
    generation: int,
    cfg: EvolutionConfig,
    colony_cfg: ColonyConfig | None,
    world_cfg: WorldConfig | None,
) -> np.ndarray:
    # Every genome in a generation faces the identical set of worlds, so the
    # comparison between them is fair, while the set itself moves on.
    seeds = [cfg.seed + generation * 1000 + w for w in range(cfg.worlds_per_genome)]
    return np.array(
        [np.mean([evaluate(g, s, colony_cfg, world_cfg) for s in seeds]) for g in population],
        dtype=np.float32,
    )


def evolve(
    cfg: EvolutionConfig | None = None,
    colony_cfg: ColonyConfig | None = None,
    world_cfg: WorldConfig | None = None,
    on_generation: Callable[[GenerationReport], None] | None = None,
) -> tuple[np.ndarray, list[GenerationReport]]:
    """Run the GA. Returns the best genome ever seen and the per-generation log."""
    cfg = cfg or EvolutionConfig()
    rng = np.random.default_rng(cfg.seed)

    population = [random_genome(rng) for _ in range(cfg.population)]
    history: list[GenerationReport] = []
    champion = population[0]
    champion_score = -np.inf

    for generation in range(cfg.generations):
        scores = score_population(population, generation, cfg, colony_cfg, world_cfg)
        order = np.argsort(scores)[::-1]

        if scores[order[0]] > champion_score:
            champion_score = float(scores[order[0]])
            champion = population[order[0]].copy()

        report = GenerationReport(
            generation=generation,
            best=float(scores[order[0]]),
            mean=float(scores.mean()),
            median=float(np.median(scores)),
            best_genome=population[order[0]].copy(),
        )
        history.append(report)
        if on_generation:
            on_generation(report)

        if generation == cfg.generations - 1:
            break

        nxt = [population[i].copy() for i in order[: cfg.elites]]
        while len(nxt) < cfg.population:
            parent = population[tournament_select(scores, cfg.tournament, rng)]
            if rng.random() < cfg.crossover_rate:
                mate = population[tournament_select(scores, cfg.tournament, rng)]
                child = crossover(parent, mate, rng)
            else:
                child = parent.copy()
            nxt.append(mutate(child, cfg, rng))
        population = nxt

    assert champion.shape == (GENOME_SIZE,)
    return champion, history
