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
STONE = "\033[38;5;242m"


def _shade(value: float, ramp: list[int]) -> str:
    """Map 0..1 pheromone onto a character plus a colour from ``ramp``."""
    level = min(int(value * len(SHADES)), len(SHADES) - 1)
    if level <= 0:
        return " "
    return BLUE.format(ramp[min(level - 1, len(ramp) - 1)]) + SHADES[level] + RESET


def grid_lines(colony: Colony) -> list[str]:
    """The bordered map as a list of lines, every one the same visual width.

    Uniform width is what lets two colonies be stitched together side by side;
    the status line is deliberately left out because it is a different length.
    """
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

    for y, x in zip(*np.nonzero(world.walls)):
        grid[y][x] = STONE + "▓" + RESET

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
    rows = [DIM + "|" + RESET + "".join(row) + DIM + "|" + RESET for row in grid]
    return [border, *rows, border]


def grid_width(colony: Colony) -> int:
    """Visual width of a grid line: the map plus its two border columns."""
    return colony.world.cfg.width + 2


def frame(colony: Colony) -> str:
    """Render one full frame, without cursor control."""
    return "\n".join(grid_lines(colony)) + "\n" + status(colony)


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
        f"{WHITE}a{RESET} searching   {RED}A{RESET} carrying   {STONE}▓{RESET} rock   "
        f"{BLUE.format(FOOD_RAMP[2])}#{RESET} trail to food   "
        f"{BLUE.format(HOME_RAMP[2])}#{RESET} trail home"
    )


GAP = "   "


def duel_frame(left: Colony, right: Colony, left_title: str, right_title: str) -> str:
    """Two colonies stitched side by side, running the same map."""
    header = (
        WHITE + left_title[: grid_width(left)].ljust(grid_width(left)) + RESET
        + GAP
        + WHITE + right_title[: grid_width(right)].ljust(grid_width(right)) + RESET
    )
    rows = [a + GAP + b for a, b in zip(grid_lines(left), grid_lines(right))]
    scores = (
        f"  {WHITE}{left_title}{RESET}: {RED}{left.delivered}{RESET} delivered"
        f"      {WHITE}{right_title}{RESET}: {RED}{right.delivered}{RESET} delivered"
        f"      {DIM}tick {left.ticks}{RESET}"
    )
    return "\n".join([header, *rows, "", scores])


def draw_duel(left: Colony, right: Colony, left_title: str, right_title: str) -> None:
    print(HOME_CURSOR + duel_frame(left, right, left_title, right_title) + "\n" + legend(),
          flush=True)


def draw(colony: Colony, title: str = "") -> None:
    """Repaint the screen in place, for animation."""
    header = f"{WHITE}{title}{RESET}\n" if title else ""
    print(HOME_CURSOR + header + frame(colony) + "\n" + legend(), flush=True)


def begin_animation() -> None:
    print(CLEAR + HIDE_CURSOR, end="", flush=True)


def end_animation() -> None:
    print(SHOW_CURSOR + RESET, flush=True)
