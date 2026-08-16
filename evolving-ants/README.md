# 🐜 antmind — evolving ant colonies from scratch

No ant is told what to do. A colony shares one tiny neural network, and a genetic
algorithm breeds better colonies over generations until they discover foraging:
wander, find food, run it home, and lay a pheromone trail so nestmates follow.

Pure NumPy. No PyTorch, no gym, no gradients — just mutation, selection, and a
chemical field that evaporates.

```
                        %%%
   ····:::··             %%%%
  ··:*###*:··       ·:·   %%
  ·:*#@#*:·   a    ·:*:·
   ··:*#*:·  A    ··:·
```

## Why it's interesting

The colony has no shared plan and no memory beyond the world itself. Everything
coordinated it does is **stigmergy** — ants change the ground, the ground changes
the ants. Evolution only ever sees one number: how much food came home.

## Layout

| file | what's in it |
| --- | --- |
| `antmind/world.py` | grid, food patches, two evaporating/diffusing pheromone fields |
| `antmind/brain.py` | 13→8→4 MLP encoded as a flat genome of 148 floats |
| `antmind/colony.py` | vectorised ant swarm: senses, turns, carries, lays trail |
| `antmind/evolve.py` | genetic algorithm — tournament selection, blend crossover, elitism |
| `antmind/render.py` | ANSI terminal renderer |
| `antmind/cli.py` | `train`, `watch`, `bench` |

## Usage

```bash
pip install -r requirements.txt

python -m antmind train --generations 30 --out champions/best.json
python -m antmind watch --genome champions/best.json
python -m antmind bench --genome champions/best.json
```

## Status

Work in progress — built one piece at a time.
