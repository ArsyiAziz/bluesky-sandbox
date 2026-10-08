"""A grid an action's values are put on: whole steps, the way air traffic
control gives them - flight levels, turns in 10 deg, speeds in 10 kt.

One :class:`Grid` per action, given as its ``grid``, on any action that takes
one value: altitude, speed, heading, a comm channel, a clearance's duration.
It is applied where the action's value is - after the normalizer, before the
action commands it - so any normalizer composes with it. It reads only what
every action has: its :meth:`~.base.ActionField.nominal` (what the value
counts from, 0 for an absolute action) and its :meth:`~.base.ActionField.reach`
(the values it can take now), in the action's own unit.

``on`` is what is put on the grid:

- ``"value"`` (the default) - the action's own value: a delta's change, so
  ``+1,000 ft`` from 23,344 ft is 24,344 ft; an absolute action's value.
- ``"target"`` - what it commands, ``nominal + value``: a delta's sum, so
  ``+1,000 ft`` from 23,344 ft is FL240. On an absolute action, the same as
  ``"value"``.

::

    actions.AltDeltaFt(grid=Grid(1000.0, on="target"))  # flight levels
    actions.HdgDeltaDeg(grid=Grid(10.0))                # turns in 10 deg
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["Grid", "GridOn", "on_grid"]

GridOn = Literal["value", "target"]


def on_grid(value: float, step: float, low: float, high: float) -> float:
    """``value`` within ``[low, high]``, on the nearest multiple of ``step``
    there; where there is none, the value in it nearest ``value``."""
    grid_low = math.ceil(low / step - 1e-9) * step
    grid_high = math.floor(high / step + 1e-9) * step
    if grid_low > grid_high:
        return min(max(float(value), float(low)), float(high))
    return min(max(round(float(value) / step) * step, grid_low), grid_high)


@dataclass(frozen=True)
class Grid:
    """Whole multiples of ``step`` (in the action's unit), for the action's
    value (``on="value"``) or for what it commands (``on="target"``)."""

    step: float
    on: GridOn = "value"

    def __post_init__(self) -> None:
        try:
            step = float(self.step)
        except (TypeError, ValueError):
            step = math.nan
        if not (math.isfinite(step) and step > 0.0):
            raise ValueError(f"Grid step must be a number > 0, got {self.step!r}")
        if self.on not in ("value", "target"):
            raise ValueError(f"Grid on must be 'value' or 'target', got {self.on!r}")
        object.__setattr__(self, "step", step)

    def apply(self, field: Any, value: float, idx: int) -> float:
        """``value`` with what ``on`` names put on the grid, within the
        action's reach now - still the action's value, as it commands."""
        low, high = field.reach(idx)
        if self.on == "value":
            return on_grid(value, self.step, low, high)
        nominal = float(field.nominal(idx))
        target = on_grid(nominal + float(value), self.step, nominal + low, nominal + high)
        return target - nominal

    def nth(self, field: Any, k: int, idx: int, every: int = 1) -> float:
        """The value ``k`` choices of ``every`` grid steps from the present
        one, within the action's reach (out of it, the reachable grid value
        nearest).

        On ``"value"``, ``k * every`` whole steps: ``k * every * step``. On
        ``"target"``, ``k * every`` grid values above (``k > 0``) or below the
        present target - the nearest at 0: from 23,344 ft or 23,600 ft, +1 is
        FL240; from FL240 itself, FL250; with ``every=2``, FL250 and FL260."""
        low, high = field.reach(idx)
        nominal = 0.0 if self.on == "value" else float(field.nominal(idx))
        at = nominal / self.step
        n = k * int(every)
        if self.on == "value" or k == 0:
            index = round(at) + n
        elif k > 0:
            index = math.floor(at + 1e-9) + n
        else:
            index = math.ceil(at - 1e-9) + n
        lowest = math.ceil((nominal + float(low)) / self.step - 1e-9)
        highest = math.floor((nominal + float(high)) / self.step + 1e-9)
        if lowest > highest:  # no grid value in reach: the value nearest it
            return min(max(0.0, float(low)), float(high))
        return min(max(index, lowest), highest) * self.step - nominal
