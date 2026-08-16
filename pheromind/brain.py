"""The thing inside an ant's head.

A brain is a two-layer MLP, 13 senses in and 4 decisions out, stored as one flat
vector of 148 floats. Flat is the point: the genetic algorithm only ever needs to
average two vectors and add noise, so it never has to know the shape of anything.

Senses (in order)::

    0      carrying food?
    1..3   FOOD pheromone at the left / centre / right antenna
    4..6   HOME pheromone at the left / centre / right antenna
    7..9   visible food at the left / centre / right antenna
    10..12 solid rock at the left / centre / right antenna
    13     sin of the bearing to the nest, relative to the ant's heading
    14     cos of the same bearing
    15     distance to the nest, normalised

Decisions::

    0..2  turn left / go straight / turn right   (argmax wins)
    3     how much pheromone to lay right now    (squashed to 0..1)

Nothing here says an ant should follow a trail, or that the nest is where food
belongs. Those weights start as noise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_SENSES = 16
N_HIDDEN = 8
N_ACTIONS = 4

GENOME_SIZE = N_SENSES * N_HIDDEN + N_HIDDEN + N_HIDDEN * N_ACTIONS + N_ACTIONS


def random_genome(rng: np.random.Generator, scale: float = 0.8) -> np.ndarray:
    return rng.normal(0.0, scale, size=GENOME_SIZE).astype(np.float32)


@dataclass(frozen=True)
class Brain:
    """A genome unpacked into weight matrices, ready to run on a whole swarm."""

    w1: np.ndarray  # (N_SENSES, N_HIDDEN)
    b1: np.ndarray  # (N_HIDDEN,)
    w2: np.ndarray  # (N_HIDDEN, N_ACTIONS)
    b2: np.ndarray  # (N_ACTIONS,)

    @classmethod
    def from_genome(cls, genome: np.ndarray) -> "Brain":
        genome = np.asarray(genome, dtype=np.float32)
        if genome.shape != (GENOME_SIZE,):
            raise ValueError(f"genome must have shape ({GENOME_SIZE},), got {genome.shape}")
        return cls(*_unpack(genome))

    def think(self, senses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run the whole colony's senses through the shared brain at once.

        ``senses`` is (n_ants, N_SENSES). Returns the chosen turn per ant, as
        -1 / 0 / +1, and how strongly each ant wants to lay pheromone, in 0..1.
        """
        hidden = np.tanh(senses @ self.w1 + self.b1)
        out = hidden @ self.w2 + self.b2

        turn = np.argmax(out[:, :3], axis=1).astype(np.int32) - 1
        lay = 0.5 * (np.tanh(out[:, 3]) + 1.0)
        return turn, lay.astype(np.float32)


def _unpack(genome: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    i = 0
    w1 = genome[i : i + N_SENSES * N_HIDDEN].reshape(N_SENSES, N_HIDDEN)
    i += w1.size
    b1 = genome[i : i + N_HIDDEN]
    i += b1.size
    w2 = genome[i : i + N_HIDDEN * N_ACTIONS].reshape(N_HIDDEN, N_ACTIONS)
    i += w2.size
    b2 = genome[i : i + N_ACTIONS]
    return w1, b1, w2, b2


def save_genome(path: str | Path, genome: np.ndarray, meta: dict | None = None) -> None:
    """Write a genome as JSON, so a champion diffs like text in git."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "genome_size": int(len(genome)),
        "architecture": [N_SENSES, N_HIDDEN, N_ACTIONS],
        "meta": meta or {},
        "genome": [round(float(v), 6) for v in genome],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_genome(path: str | Path) -> tuple[np.ndarray, dict]:
    payload = json.loads(Path(path).read_text())
    genome = np.asarray(payload["genome"], dtype=np.float32)
    if genome.shape != (GENOME_SIZE,):
        raise ValueError(
            f"{path} holds a {genome.shape[0]}-gene genome, but this build expects {GENOME_SIZE}"
        )
    return genome, payload.get("meta", {})
