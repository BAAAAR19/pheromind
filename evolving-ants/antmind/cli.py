"""Command line: breed colonies, watch one forage, or measure how good it is.

    python -m antmind train --generations 30 --out champions/best.json
    python -m antmind watch --genome champions/best.json
    python -m antmind bench --genome champions/best.json --trials 20
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from .brain import Brain, load_genome, random_genome, save_genome
from .colony import Colony, ColonyConfig, evaluate
from .evolve import EvolutionConfig, GenerationReport, evolve
from .render import begin_animation, draw, end_animation
from .world import World, WorldConfig

DEFAULT_CHAMPION = Path(__file__).resolve().parent.parent / "champions" / "best.json"


def _world_cfg(args: argparse.Namespace) -> WorldConfig:
    return WorldConfig(width=args.width, height=args.height, food_piles=args.piles)


def _colony_cfg(args: argparse.Namespace) -> ColonyConfig:
    return ColonyConfig(n_ants=args.ants, steps=args.steps)


def cmd_train(args: argparse.Namespace) -> int:
    evo_cfg = EvolutionConfig(
        population=args.population,
        generations=args.generations,
        worlds_per_genome=args.worlds,
        seed=args.seed,
    )
    colony_cfg = _colony_cfg(args)
    world_cfg = _world_cfg(args)

    print(f"breeding {evo_cfg.population} colonies of {colony_cfg.n_ants} ants "
          f"for {evo_cfg.generations} generations\n")

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
        args.out,
        champion,
        {
            "generations": evo_cfg.generations,
            "population": evo_cfg.population,
            "ants": colony_cfg.n_ants,
            "steps": colony_cfg.steps,
            "seed": evo_cfg.seed,
            "final_mean": round(last.mean, 2),
            "best_score": round(max(h.best for h in history), 2),
        },
    )
    print(f"champion written to {args.out}")
    return 0


def _load(args: argparse.Namespace) -> np.ndarray:
    path = Path(args.genome)
    if not path.exists():
        print(f"no genome at {path} — run `python -m antmind train` first, "
              f"or pass --random to watch an untrained colony.", file=sys.stderr)
        raise SystemExit(2)
    genome, meta = load_genome(path)
    if meta:
        print(f"loaded champion: {meta}")
    return genome


def cmd_watch(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    if args.random:
        genome = random_genome(rng)
        title = "untrained colony (random weights)"
    else:
        genome = _load(args)
        title = "evolved colony"

    world = World.generate(_world_cfg(args), rng)
    colony = Colony.spawn(world, Brain.from_genome(genome), _colony_cfg(args), rng)

    delay = 1.0 / args.fps if args.fps > 0 else 0.0
    begin_animation()
    try:
        for _ in range(args.steps):
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
        scores.append(evaluate(genome, seed=10_000 + trial, colony_cfg=colony_cfg, world_cfg=world_cfg))
        print(f"\rworld {trial + 1}/{args.trials}", end="", flush=True)

    spread = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    print(
        f"\n{args.trials} unseen worlds | mean {statistics.mean(scores):.2f} "
        f"| median {statistics.median(scores):.2f} | sd {spread:.2f} "
        f"| worst {min(scores):.2f} | best {max(scores):.2f}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="antmind", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest="command", required=True)

    def add_world_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--width", type=int, default=56)
        p.add_argument("--height", type=int, default=30)
        p.add_argument("--piles", type=int, default=5, help="food piles per world")
        p.add_argument("--ants", type=int, default=45)
        p.add_argument("--steps", type=int, default=420, help="ticks per episode")
        p.add_argument("--seed", type=int, default=0)

    train = subs.add_parser("train", help="breed colonies with the genetic algorithm")
    add_world_args(train)
    train.add_argument("--population", type=int, default=32)
    train.add_argument("--generations", type=int, default=25)
    train.add_argument("--worlds", type=int, default=2, help="worlds averaged per genome")
    train.add_argument("--out", default=str(DEFAULT_CHAMPION))
    train.set_defaults(func=cmd_train)

    watch = subs.add_parser("watch", help="animate one colony foraging")
    add_world_args(watch)
    watch.add_argument("--genome", default=str(DEFAULT_CHAMPION))
    watch.add_argument("--random", action="store_true", help="watch untrained weights instead")
    watch.add_argument("--fps", type=float, default=18.0, help="0 for no delay")
    watch.add_argument("--every", type=int, default=1, help="draw every Nth tick")
    watch.set_defaults(func=cmd_watch)

    bench = subs.add_parser("bench", help="score a genome on unseen worlds")
    add_world_args(bench)
    bench.add_argument("--genome", default=str(DEFAULT_CHAMPION))
    bench.add_argument("--random", action="store_true")
    bench.add_argument("--trials", type=int, default=20)
    bench.set_defaults(func=cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
