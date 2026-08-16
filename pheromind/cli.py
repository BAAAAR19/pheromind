"""Command line: breed colonies, watch one forage, or measure how good it is.

    python -m pheromind list
    python -m pheromind train --scenario gauntlet
    python -m pheromind watch --scenario gauntlet
    python -m pheromind bench --scenario gauntlet --trials 25
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import scenarios
from .brain import Brain, load_genome, random_genome, save_genome
from .colony import Colony, ColonyConfig, evaluate
from .evolve import EvolutionConfig, GenerationReport, evolve
from .render import begin_animation, draw, end_animation
from .world import World, WorldConfig

CHAMPION_DIR = Path(__file__).resolve().parent.parent / "champions"


# -- turning CLI flags into configs --------------------------------------
#
# Every world/colony flag defaults to None rather than a number, so "not
# passed" is distinguishable from "passed the same value the scenario uses".
# The scenario supplies the baseline; explicit flags override it.


def _world_cfg(args: argparse.Namespace) -> WorldConfig:
    cfg = scenarios.get(args.scenario).world
    changes = {}
    if args.width is not None:
        changes["width"] = args.width
    if args.height is not None:
        changes["height"] = args.height
    if args.piles is not None:
        changes["food_piles"] = args.piles
    if args.walls is not None:
        changes["wall_style"] = args.walls
    return replace(cfg, **changes)


def _colony_cfg(args: argparse.Namespace) -> ColonyConfig:
    cfg = scenarios.get(args.scenario).colony
    changes = {}
    if args.ants is not None:
        changes["n_ants"] = args.ants
    if args.steps is not None:
        changes["steps"] = args.steps
    return replace(cfg, **changes)


def _genome_path(args: argparse.Namespace, attr: str) -> Path:
    """Champions are named after their scenario unless told otherwise."""
    given = getattr(args, attr, None)
    return Path(given) if given else CHAMPION_DIR / f"{args.scenario}.json"


def _load(args: argparse.Namespace, attr: str = "genome") -> np.ndarray:
    path = _genome_path(args, attr)
    if not path.exists():
        print(
            f"no genome at {path}\n"
            f"  breed one:  python -m pheromind train --scenario {args.scenario}\n"
            f"  or watch untrained weights:  python -m pheromind watch --random",
            file=sys.stderr,
        )
        raise SystemExit(2)
    genome, meta = load_genome(path)
    if meta:
        trained_on = meta.get("scenario")
        if trained_on and trained_on != args.scenario:
            print(f"note: this genome was bred on '{trained_on}', "
                  f"now running on '{args.scenario}'")
    return genome


# -- commands ------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    print("scenarios:\n")
    print(scenarios.describe())
    print(f"\nchampions live in {CHAMPION_DIR}")
    for path in sorted(CHAMPION_DIR.glob("*.json")):
        print(f"  {path.stem}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    evo_cfg = EvolutionConfig(
        population=args.population,
        generations=args.generations,
        worlds_per_genome=args.worlds,
        seed=args.seed,
        workers=args.workers,
    )
    colony_cfg = _colony_cfg(args)
    world_cfg = _world_cfg(args)
    out = _genome_path(args, "out")

    episodes = evo_cfg.population * evo_cfg.generations * evo_cfg.worlds_per_genome
    print(f"scenario '{args.scenario}' — {scenarios.get(args.scenario).blurb}")
    print(f"breeding {evo_cfg.population} colonies of {colony_cfg.n_ants} ants "
          f"for {evo_cfg.generations} generations")
    print(f"{episodes} episodes across {evo_cfg.resolved_workers()} worker(s)\n")

    started = time.time()

    def report(r: GenerationReport) -> None:
        print(f"{r.line()}  {'#' * min(int(r.mean), 60)}")

    champion, history = evolve(evo_cfg, colony_cfg, world_cfg, on_generation=report)
    elapsed = time.time() - started

    first, last = history[0], history[-1]
    print(
        f"\ndone in {elapsed:.1f}s — mean colony went {first.mean:.1f} -> {last.mean:.1f}, "
        f"best generation scored {max(h.best for h in history):.1f}"
    )

    save_genome(
        out,
        champion,
        {
            "scenario": args.scenario,
            "generations": evo_cfg.generations,
            "population": evo_cfg.population,
            "ants": colony_cfg.n_ants,
            "steps": colony_cfg.steps,
            "seed": evo_cfg.seed,
            "final_mean": round(last.mean, 2),
            "best_score": round(max(h.best for h in history), 2),
        },
    )
    print(f"champion written to {out}")
    print(f"watch it:  python -m pheromind watch --scenario {args.scenario}")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    colony_cfg = _colony_cfg(args)

    if args.random:
        genome = random_genome(rng)
        title = f"{args.scenario}: untrained colony (random weights)"
    else:
        genome = _load(args)
        title = f"{args.scenario}: evolved colony"

    world = World.generate(_world_cfg(args), rng)
    colony = Colony.spawn(world, Brain.from_genome(genome), colony_cfg, rng)

    delay = 1.0 / args.fps if args.fps > 0 else 0.0
    begin_animation()
    try:
        for _ in range(colony_cfg.steps):
            colony.step()
            if colony.ticks % args.every == 0:
                draw(colony, title)
                if delay:
                    time.sleep(delay)
            if world.food_remaining() == 0 and not colony.carrying.any():
                break
    except KeyboardInterrupt:
        pass
    finally:
        draw(colony, title)
        end_animation()

    print(f"\n{colony.delivered} food delivered in {colony.ticks} ticks.")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    genome = random_genome(np.random.default_rng(args.seed)) if args.random else _load(args)
    colony_cfg, world_cfg = _colony_cfg(args), _world_cfg(args)

    scores = []
    for trial in range(args.trials):
        scores.append(evaluate(genome, 10_000 + trial, colony_cfg, world_cfg))
        print(f"\rworld {trial + 1}/{args.trials}", end="", flush=True)

    spread = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    print(
        f"\n{args.scenario}: {args.trials} unseen worlds | "
        f"mean {statistics.mean(scores):.2f} | median {statistics.median(scores):.2f} "
        f"| sd {spread:.2f} | worst {min(scores):.2f} | best {max(scores):.2f}"
    )
    return 0


# -- parser --------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pheromind",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subs = parser.add_subparsers(dest="command", required=True)

    def add_world_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--scenario", default=scenarios.DEFAULT_SCENARIO,
                       choices=sorted(scenarios.SCENARIOS), help="named world preset")
        p.add_argument("--width", type=int, help="override scenario map width")
        p.add_argument("--height", type=int, help="override scenario map height")
        p.add_argument("--piles", type=int, help="override number of food piles")
        p.add_argument("--walls", choices=("none", "blocks", "ring"),
                       help="override terrain style")
        p.add_argument("--ants", type=int, help="override colony size")
        p.add_argument("--steps", type=int, help="override ticks per episode")
        p.add_argument("--seed", type=int, default=0)

    listing = subs.add_parser("list", help="show scenarios and available champions")
    listing.set_defaults(func=cmd_list)

    train = subs.add_parser("train", help="breed colonies with the genetic algorithm")
    add_world_args(train)
    train.add_argument("--population", type=int, default=32)
    train.add_argument("--generations", type=int, default=25)
    train.add_argument("--worlds", type=int, default=2, help="worlds averaged per genome")
    train.add_argument("--workers", type=int, default=0,
                       help="parallel processes; 0 = one per core, 1 = serial")
    train.add_argument("--out", help="where to write the champion "
                                     "(default: champions/<scenario>.json)")
    train.set_defaults(func=cmd_train)

    watch = subs.add_parser("watch", help="animate one colony foraging")
    add_world_args(watch)
    watch.add_argument("--genome", help="default: champions/<scenario>.json")
    watch.add_argument("--random", action="store_true", help="watch untrained weights instead")
    watch.add_argument("--fps", type=float, default=18.0, help="0 for no delay")
    watch.add_argument("--every", type=int, default=1, help="draw every Nth tick")
    watch.set_defaults(func=cmd_watch)

    bench = subs.add_parser("bench", help="score a genome on unseen worlds")
    add_world_args(bench)
    bench.add_argument("--genome", help="default: champions/<scenario>.json")
    bench.add_argument("--random", action="store_true")
    bench.add_argument("--trials", type=int, default=20)
    bench.set_defaults(func=cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
