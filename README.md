# 🐜 pheromind

**Ant colonies that evolve their own foraging strategy, in your terminal.**

Nobody tells these ants what to do. A colony shares one small neural network —
172 numbers — and a genetic algorithm breeds better colonies until they work out
foraging on their own: wander, find food, carry it home, and lay a chemical
trail that pulls nestmates along behind them.

Pure NumPy. No PyTorch, no gym, no gradients. Just mutation, selection, and a
scent that evaporates.

```
evolved                                  untrained
+------------------------------------+   +------------------------------------+
|        ·a:···    a    aaa·*·       |   |                               ·:*·A|
|  a*:·a####**::····#···#:##·:       |   |                              ·:::*·|
|    % ·:·:*a:····  ··· ·:*· ·       |   |                              ·:::#·|
|   %% :····a#:·      ·  ·*···       |   |   %%% %%%                     ·a#· |
|  %%%a%·:···##:·         ····       |   |  %%%%%%%%%                     ··  |
|   %·%·::::::##··        ···:·      |   |   %%% %%%                          |
|    ·····:*:*##*·       ····::··    |   |    %   %                          ·|
|         ··:*::A*ooo    ····:::···  |   |                 ooo              ·#|
|              ··*o@oa· ·····:::::·a*|   |                 o@o            % ·A|
|                ·ooo:#A:·:*::::::::·|   |                 ooo           %%% ·|
|                  ···:*##*······::% |   |                              %%%%% |
|                   ·:·a:a:·······%· |   |                                %%  |
|                   ·:···#·:::::·%%#·|   |                                %%  |
|                    ·:··#:····::A%%*|   |                                %%% |
|                    ··a:#:::::··%%%%|   |·                              %%%%%|
|                   ····:*::::::···:·|   |A:                              %%% |
|                       ·:······  %  |   |A#·                              %  |
+------------------------------------+   +------------------------------------+

  evolved: 200 delivered      untrained: 0 delivered      tick 220
```

Same map, same starting positions, same everything — the only difference is 172
numbers. After 220 ticks the left colony has carried home 200 units and stripped
two thirds of the map's food, carving visible highways between the nest and the
piles. The right colony has delivered nothing.

You can run that exact race yourself in about thirty seconds.

---

## Quickstart

```bash
git clone https://github.com/BAAAAR19/pheromind.git
cd pheromind
pip install -r requirements.txt
```

Python 3.10 or newer.

The only dependency is NumPy. Trained champions ship with the repo, so there is
nothing to train before you can watch something happen.

**Watch an evolved colony forage:**

```bash
python -m pheromind watch
```

**Race it against untrained weights:**

```bash
python -m pheromind duel
```

**Breed your own from random noise** (about 2–3 minutes on 8 cores):

```bash
python -m pheromind train
```

**See what it actually learned to use:**

```bash
python -m pheromind ablate
```

---

## Reading the screen

| symbol | meaning |
| --- | --- |
| `@` | the nest |
| `%` | food (brighter = a deeper pile) |
| `a` | an ant, searching |
| `A` | an ant, carrying food home |
| `▓` | rock — blocks movement *and* scent |
| `·` `:` `*` `#` | pheromone, faint to strong |

Warm-coloured trails point toward food. Cool-coloured trails point back home.
Nothing in the code draws a path between the nest and a pile; those lines are
forty-five ants agreeing with each other.

---

## Commands

| command | what it does |
| --- | --- |
| `list` | show the scenarios and which champions you have |
| `watch` | animate one colony foraging |
| `duel` | race two colonies side by side on an identical map |
| `train` | breed a new champion with the genetic algorithm |
| `bench` | score a champion across worlds it never trained on |
| `ablate` | knock out senses one at a time to see which ones matter |

Every command takes `--scenario`, `--seed`, and overrides for `--width`,
`--height`, `--piles`, `--walls`, `--ants` and `--steps`, so you can starve a
colony, flood it, or wall it in and see what survives.

```bash
python -m pheromind watch --scenario gauntlet --fps 30
python -m pheromind duel --genome champions/classic.json --vs champions/gauntlet.json
python -m pheromind train --scenario keyhole --generations 60 --population 60
python -m pheromind bench --scenario boulders --trials 50
python -m pheromind watch --piles 1 --ants 120 --steps 900   # one pile, a mob
```

---

## Scenarios

Each one asks a different question, and they breed visibly different colonies.

```bash
python -m pheromind list
```

| scenario | the problem it poses | champion ships? |
| --- | --- | --- |
| `classic` | Open ground, five piles at a fair distance. The baseline. | ✅ |
| `famine` | Two small piles pushed out to the edges. Rewards scouts. | ✅ |
| `feast` | Food everywhere and close by. Trails barely earn their keep. | — |
| `boulders` | Open plan, strewn with rock the colony has to route around. | ✅ |
| `gauntlet` | The nest is walled in. Three doorways, all the food outside. | ✅ |
| `keyhole` | One single gap in the wall. Brutal. | ✅ |
| `sprawl` | A big map and a short clock. Every wasted step costs. | — |

The two without a champion are left as an exercise — `train --scenario feast`
takes a couple of minutes.

---

## Does it actually work?

Breeding on `classic` — 40 generations of 40 colonies, each genome averaged over
3 freshly generated worlds:

```
gen   0 | best   74.53 | mean    7.03 | median    0.69
gen  10 | best  195.20 | mean   68.25 | median   64.31
gen  20 | best  234.78 | mean  156.78 | median  165.89
gen  30 | best  276.83 | mean  205.12 | median  210.38
gen  39 | best  295.63 | mean  214.05 | median  223.83
```

Every shipped champion is bred on its own scenario, then scored on 20 worlds it
has never seen — counting only food actually carried home, not the fitness score:

| scenario | evolved | worst | best | random weights |
| --- | --- | --- | --- | --- |
| `classic` | **256.2** | 184 | 332 | 0.0 |
| `gauntlet` | **203.9** | 155 | 266 | 0.0 |
| `boulders` | **144.8** | 12 | 305 | 0.0 |
| `keyhole` | **139.6** | 74 | 223 | 0.0 |
| `famine` | **63.1** | 48 | 72 | 0.0 |

Untrained colonies deliver nothing at all — not a little, exactly zero, on every
world of every scenario. Everything in that table was found by selection.

The spread is as interesting as the mean. `famine` is remarkably consistent
(48–72): with two piles there is not much luck available. `boulders` swings from
12 to 305, because a bad roll can drop a boulder right across the way out and
the colony never recovers inside its 420 ticks.

Reproduce any row:

```bash
python -m pheromind bench --scenario gauntlet --trials 20
python -m pheromind bench --scenario gauntlet --trials 20 --random
```

---

## What did it actually learn?

This is the part most projects skip, and it is the most interesting part.

You cannot answer it by reading the genome — a weight matrix will happily hold
big numbers for a sense the colony ignores. So `ablate` cuts one signal at a
time and re-runs:

```bash
python -m pheromind ablate
```

```
  knocked out               delivered     cost
  blind to carrying              12.6      95%
  blind to nest_bearing          16.8      93%
  cannot lay pheromone          187.6      21%
  blind to food_sight           188.2      21%
  blind to home_trail           222.0       7%
  blind to food_trail           240.5      -1%
  blind to rock                 257.2      -8%
```

The honest reading: on open ground this colony leans overwhelmingly on two
things — knowing whether it is already carrying something, and knowing which way
home is. Pheromones matter, but far less than the pretty screenshot suggests.
Muting them costs 21%. Blinding the food-trail channel specifically costs
*nothing measurable*.

That is a less flattering story than "the colony coordinates through chemical
trails", and it is the one the numbers support.

**But run the same test on a walled map and it inverts.** Here is the same
ablation on `gauntlet`, where the nest sits behind a ring wall:

| knocked out | cost on `classic` | cost on `gauntlet` |
| --- | --- | --- |
| cannot lay pheromone | 21% | **43%** |
| blind to rock | −8% | **24%** |
| blind to food-trail | −1% | **11%** |
| blind to home-trail | 7% | 11% |
| blind to nest bearing | 93% | 100% |

When a straight line home is a straight line into rock, a compass bearing stops
being enough. The trails go from a mild optimisation to carrying nearly half the
colony's performance — and the rock sense goes from actively unhelpful to
essential.

Same code, same fitness function, different world. The strategy that evolves is
a property of the problem, not of the algorithm.

---

## How it works

**The world** is a grid holding food piles, one nest, optional rock, and two
pheromone fields. Every tick both fields evaporate and bleed into their
neighbours — which is what turns isolated deposits into followable lines, and
what makes an unused trail disappear. Rock holds no scent, so trails bend
around obstacles the same way ants do.

**The brain** is a 16 → 8 → 4 MLP stored as one flat vector of 172 floats.
Sixteen senses:

- am I carrying food?
- food-pheromone at my left, centre and right antenna
- home-pheromone at the same three
- visible food at the same three
- solid rock at the same three
- sine and cosine of the bearing to the nest, and how far away it is

Four outputs: three turn logits (left / straight / right, highest wins) and one
gate for how much pheromone to lay. Which channel that lands in is decided by
whether the ant is loaded — so *what a trail means* is emergent, not specified.

**The colony** is 45 ants running identical weights. They diverge only because
they stand in different places and smell different things. That constraint is
the whole point: a strategy cannot be split across specialists, so it has to be
one reactive policy that happens to look like teamwork.

**Evolution** is a plain generational GA — score everyone, keep two elites,
refill from tournament winners with blend crossover and Gaussian mutation. Two
details matter more than the rest:

- *Fresh worlds every generation.* Fixed maps breed colonies that memorise where
  the food was. Re-rolling means only a strategy survives.
- *Fitness is one number.* Food delivered. No reward for following a trail, no
  bonus for exploring. Those had to be worth inventing.

Training runs the population across every core — 4.3× faster than serial on an
8-core laptop, and bit-for-bit identical output, which there is a test for.

---

## Development

```bash
python -m unittest discover -s tests -t .
```

61 tests, standard library only, about 7 seconds. They mostly pin the invariants
a simulation breaks quietly: food is conserved across pickup and delivery, ants
never walk off the grid or stand inside rock, trails really do decay to zero,
two ants cannot both take the last crumb, mutation never modifies its parent,
ablation zeroes exactly the senses it claims to, and a seeded run reproduces
exactly.

CI runs the suite on Python 3.10 through 3.13 and then drives the CLI end to
end, because that is where a refactor breaks things for someone who just cloned
the repo.

```
pheromind/
  world.py       grid, food, rock, evaporating pheromone fields
  brain.py       the MLP, the flat genome, save/load
  colony.py      vectorised ant swarm — senses, turning, carrying, laying
  evolve.py      the genetic algorithm and the process pool
  render.py      ANSI terminal rendering, single and side-by-side
  scenarios.py   the seven named worlds
  cli.py         list / watch / duel / train / bench / ablate
champions/       evolved genomes, one JSON file per scenario
tests/
```

Genomes are JSON on purpose — a champion should diff like text, not like a blob.

## Things worth trying

- `python -m pheromind duel --genome champions/gauntlet.json --vs champions/classic.json --scenario gauntlet`
  — the specialist against the generalist, on the specialist's home turf.
- `python -m pheromind ablate --scenario boulders` — see whether rock-sensing
  matters as much when the rock is scattered rather than in a ring.
- `python -m pheromind watch --scenario classic --piles 1 --ants 150` — one pile,
  a mob, and a traffic jam.
- `python -m pheromind train --scenario feast --generations 60` then ablate it.
  Food is everywhere, so trails should matter even less than on `classic`.
- Change `evaporation` in `WorldConfig` from `0.96` to `0.99` and retrain. Trails
  that never fade should, in principle, be worse than trails that do.

## License

MIT. See [LICENSE](LICENSE).
