"""Draw a colony in the terminal with ANSI escapes.

Pheromone strength becomes a shade ramp; the two channels get two colours, and a
cell holding both is drawn in the stronger one. Ants are drawn last so they are
always visible on top of their own trail.
"""

from __future__ import annotations

import numpy as np

from .colony import Colony
from .world import FOOD, HOME

RESET = "\033[0m"
CLEAR = "\033[2J"
HOME_CURSOR = "\033[H"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"

# Faint to solid.
SHADES = " ·:*#"

BLUE = "\033[38;5;{}m"
HOME_RAMP = [24, 25, 32, 39]  # cool blues: the way back to the nest
FOOD_RAMP = [130, 172, 214, 220]  # warm ambers: the way to a pile

GREEN = "\033[38;5;70m"
BRIGHT_GREEN = "\033[38;5;120m"
WHITE = "\033[38;5;255m"
RED = "\033[38;5;203m"
DIM = "\033[38;5;240m"


def _shade(value: float, ramp: list[int]) -> str:
    """Map 0..1 pheromone onto a character plus a colour from ``ramp``."""
    level = min(int(value * len(SHADES)), len(SHADES) - 1)
    if level <= 0:
        return " "
    return BLUE.format(ramp[min(level - 1, len(ramp) - 1)]) + SHADES[level] + RESET


def frame(colony: Colony) -> str:
    """Render one full frame, without cursor control."""
    world = colony.world
    h, w = world.shape
    grid = [[" " for _ in range(w)] for _ in range(h)]

    home_p = world.pheromone[HOME]
    food_p = world.pheromone[FOOD]
    # Scale by the strongest trail on screen so faint early trails stay visible.
    peak = max(float(home_p.max()), float(food_p.max()), 1e-6)

    for y in range(h):
        for x in range(w):
            hv, fv = home_p[y, x] / peak, food_p[y, x] / peak
            if hv < 0.06 and fv < 0.06:
                continue
            if fv >= hv:
                grid[y][x] = _shade(fv, FOOD_RAMP)
            else:
                grid[y][x] = _shade(hv, HOME_RAMP)

    for y, x in zip(*np.nonzero(world.food)):
        heavy = world.food[y, x] >= 3
        grid[y][x] = (BRIGHT_GREEN if heavy else GREEN) + "%" + RESET

    nx, ny = world.nest
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            x, y = nx + dx, ny + dy
            if 0 <= x < w and 0 <= y < h:
                grid[y][x] = WHITE + ("@" if dx == dy == 0 else "o") + RESET

    for x, y, carrying in zip(colony.x, colony.y, colony.carrying):
        grid[int(y)][int(x)] = (RED + "A" if carrying else WHITE + "a") + RESET

    border = DIM + "+" + "-" * w + "+" + RESET
    body = "\n".join(DIM + "|" + RESET + "".join(row) + DIM + "|" + RESET for row in grid)
    return f"{border}\n{body}\n{border}\n{status(colony)}"


def status(colony: Colony) -> str:
    world = colony.world
    carried = int(colony.carrying.sum())
    return (
        f"{WHITE}tick{RESET} {colony.ticks:>4}   "
        f"{RED}delivered{RESET} {colony.delivered:>4}   "
        f"{GREEN}carrying{RESET} {carried:>3}/{len(colony.x)}   "
        f"{DIM}food left{RESET} {world.food_remaining():>4}/{world.food_at_start}"
    )


def legend() -> str:
    return (
        f"{WHITE}@{RESET} nest   {GREEN}%{RESET} food   "
        f"{WHITE}a{RESET} searching   {RED}A{RESET} carrying   "
        f"{BLUE.format(FOOD_RAMP[2])}#{RESET} trail to food   "
        f"{BLUE.format(HOME_RAMP[2])}#{RESET} trail home"
    )


def draw(colony: Colony, title: str = "") -> None:
    """Repaint the screen in place, for animation."""
    header = f"{WHITE}{title}{RESET}\n" if title else ""
    print(HOME_CURSOR + header + frame(colony) + "\n" + legend(), flush=True)


def begin_animation() -> None:
    print(CLEAR + HIDE_CURSOR, end="", flush=True)


def end_animation() -> None:
    print(SHOW_CURSOR + RESET, flush=True)
