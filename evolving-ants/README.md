# 🐜 antmind — evolving ant colonies from scratch

No ant is told what to do. A colony shares one tiny neural network — 148 floats —
and a genetic algorithm breeds better colonies until they discover foraging on
their own: wander, find food, run it home, and lay a pheromone trail so
nestmates follow.

Pure NumPy. No PyTorch, no gym, no gradients — just mutation, selection, and a
chemical field that evaporates.

```
+----------------------------------------------------+
|                            a#     ·a               |
|                          aa   *  ···               |
|        %                        ·:··%%             |
|       %%%                       :···%           ··*|
|      %%%%%                    A····%            A: |
|       %%%                   ··:···                 |
|        %                   ·*a:······              |
|                           ·#*:·········            |
|                           ·a*::·······AA%          |
|                         ooo:**:······· %%%      %  |
|                         o@o*···a···a    %      %%% |
|                         ooo· ···       %      %%%%%|
|                                                %%% |
|                                                 %  |
|                                                    |
|                                      %             |
|                                    ···           a |
|                                    ·····     ···::a|
|                                                    |
|                                      %             |
+----------------------------------------------------+
tick  140   delivered  159   carrying   9/45   food left  222/390
```

`@` nest · `%` food · `a` searching ant · `A` carrying ant · `·:*#` pheromone,
faint to strong. That smear running east from the nest to the pile is not drawn
by anything in the code — it is 45 ants agreeing.

## Results

40 generations of 40 colonies, each genome scored on 3 freshly generated worlds:

```
gen   0 | best   99.12 | mean   13.12 | median    0.99
gen   5 | best  218.52 | mean   56.67 | median   21.54
gen  10 | best  234.97 | mean  105.97 | median  102.65
gen  20 | best  254.98 | mean  159.98 | median  175.18
gen  30 | best  304.17 | mean  195.26 | median  222.07
gen  39 | best  302.52 | mean  209.93 | median  227.70
```

The mean colony went from 13 to 210 in 22 minutes on a laptop CPU.

On 25 worlds it had never seen:

| colony | mean delivered | median | worst | best |
| --- | --- | --- | --- | --- |
| evolved | **273.4** | 271.2 | 193.5 | 365.4 |
| random weights | 0.0 | 0.0 | 0.0 | 0.0 |

Untrained colonies deliver *nothing* — not a little, exactly zero across all 25
maps. Everything above that line was found by selection alone.

### What it actually learned

Watching the champion, the trail chemistry shifts as a map gets stripped:

| tick | delivered | food-trail strength | home-trail strength |
| --- | --- | --- | --- |
| 100 | 89 | 32.8 | 9.5 |
| 200 | 174 | 24.1 | 22.5 |
| 420 | 317 | 12.0 | 32.2 |

Early on, with piles still full, the colony pours out *food* pheromone —
recruitment. As the piles empty, the balance flips toward the *home* channel and
the ants spread out to search again. Nobody wrote that schedule; the only thing
evolution was ever shown is how much food came home.

## Why it's interesting

The colony has no shared plan, no map, and no memory outside the world itself.
Everything coordinated it does is **stigmergy** — ants change the ground, the
ground changes the ants. Three constraints do the heavy lifting:

* **One brain per colony, not per ant.** Every ant runs identical weights, so
  differences in behaviour can only come from what an ant is standing on. A
  strategy cannot be split across specialists; it has to be a single reactive
  policy that happens to look like teamwork.
* **Fresh worlds every generation.** Fixed maps breed colonies that memorise
  where the food was. Re-rolling means only a strategy survives.
* **Fitness is one number.** Food delivered. No shaping toward "follow the
  trail", no reward for exploring — those had to be worth inventing.

## Layout

| file | what's in it |
| --- | --- |
| `antmind/world.py` | grid, food piles, two evaporating/diffusing pheromone fields |
| `antmind/brain.py` | 13→8→4 MLP encoded as a flat genome of 148 floats |
| `antmind/colony.py` | vectorised ant swarm: senses, turns, carries, lays trail |
| `antmind/evolve.py` | genetic algorithm — tournament selection, blend crossover, elitism |
| `antmind/render.py` | ANSI terminal renderer |
| `antmind/cli.py` | `train`, `watch`, `bench` |

### The 13 senses

`carrying?` · food pheromone at three antennae · home pheromone at three
antennae · visible food at three antennae · sin/cos of the bearing to the nest ·
distance to the nest.

The four outputs are three turn logits (left / straight / right, argmax wins)
and one gate for how much pheromone to lay. Which channel it lands in is decided
by whether the ant is loaded — so *what a trail means* is emergent, not
specified.

The nest bearing is the one concession to realism-over-purity: real ants
path-integrate, and without it a first-generation random colony never once finds
its way home, leaving selection with a completely flat landscape to climb.

## Usage

```bash
pip install -r requirements.txt
```

Watch the champion forage:

```bash
python -m antmind watch
```

See what untrained weights look like, for contrast:

```bash
python -m antmind watch --random
```

Breed your own:

```bash
python -m antmind train --generations 40 --population 40 --worlds 3 --out champions/mine.json
```

Score a genome on worlds it never trained on:

```bash
python -m antmind bench --genome champions/best.json --trials 25
```

Every command takes `--width`, `--height`, `--piles`, `--ants`, `--steps` and
`--seed`, so you can starve a colony or flood it and see what survives.

## Tests

```bash
python -m unittest discover -s tests -t .
```

35 tests, standard library only, ~1.6s. They mostly guard the invariants a
simulation quietly breaks: food is conserved across pickup and delivery, ants
never walk off the grid, trails really do fade to zero, two ants cannot both
take the last crumb, and a seeded evolution run reproduces exactly.
