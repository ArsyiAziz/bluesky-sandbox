"""A crossover speed action above its crossover: in Mach.

Below a speed schedule's crossover altitude a jet's speed is a CAS; above it, a
Mach - and each is given on its own scale: a CAS in knots (in 10 kt steps, say),
a Mach in hundredths. A crossover speed action (``ApSpdDeltaCrossover``,
``ActiveRouteWaypointSpdDeltaCrossover``) given a :class:`MachRegime` acts that
way: below the regime's :class:`~bluesky_sandbox.sim.performance.Crossover` its
value is a change in knots, as without one; above it, a change in Mach from the
nominal Mach, on its own scale - its own bounds, normalizer and grid - in the
same part of the action space, the same size, so the policy's action is one
shape at every level. The policy tells the two scales apart by the regime flag
below.

Passing the crossover, an aircraft's speed hold is handed over as an FMS does -
a CAS held becomes the Mach it is there, climbing; a Mach, its CAS, descending
- which BlueSky does not do (``MachRegime.handover``, on by default, for the
aircraft the action commands). The regime changes ``Crossover.margin_ft`` past
the crossover, so a level-off near it does not flip the hold back and forth.

``AboveCrossover(crossover=...)``, given the same crossover, is the flag that
says which regime the action is in::

    xo = Crossover(cas_kts=300, mach=0.78)
    act.ApSpdDeltaCrossover(
        normalizer=SymmetricNormalizer(),
        above_crossover=MachRegime(xo, low=-0.04, high=0.04),
    )
    obs.AboveCrossover(crossover=xo)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Annotated, Any

import bluesky as bs
from bluesky.tools.aero import vcas2mach

from bluesky_sandbox.sim.performance.speeds import Crossover

from .._common import _MIN_DYNAMIC_SPAN, on_reset
from ..base import Unit, action_kind
from ..grid import Grid
from ._targets import _CrossoverSpeedAxis, _reachable_delta

__all__ = ["Crossover", "MachRegime"]


@dataclass(frozen=True)
class MachRegime:
    """How a crossover speed action acts above ``crossover``: its value a change
    in Mach from the nominal - the aircraft's Mach, or the waypoint speed's at
    its altitude - bounded by ``low``/``high`` (None: symmetric about the
    nominal, within what the aircraft can fly), through ``normalizer`` (None:
    the action's own, where it has no unit of its own) and onto ``grid``
    (None: none - the action's grid is in knots). ``handover`` - hand the
    speed hold of the aircraft it commands over at the crossover, as an FMS
    does (see the module)."""

    crossover: Annotated[Crossover, "the speed schedule whose crossover it starts at"] = field(
        default_factory=Crossover
    )
    low: Annotated[float | None, "Mach delta (low); None = runtime speed envelope"] = None
    high: Annotated[float | None, "Mach delta (high); None = runtime speed envelope"] = None
    normalizer: Annotated[Any | None, "above crossover; None = the action's own"] = None
    grid: Annotated[Grid | None, "whole Mach steps above crossover; None = none"] = None
    handover: Annotated[bool, "hand a CAS hold over to Mach climbing past the crossover, and back"] = True

    def __post_init__(self) -> None:
        if not isinstance(self.crossover, Crossover):
            raise TypeError(f"MachRegime crossover must be a Crossover, got {self.crossover!r}")
        if (self.low is None) != (self.high is None):
            raise ValueError("MachRegime takes both low and high, or neither")
        if self.low is not None and not float(self.low) < float(self.high):
            raise ValueError(f"MachRegime low ({self.low}) must be below high ({self.high})")
        if self.grid is not None and not isinstance(self.grid, Grid):
            raise TypeError(f"MachRegime grid must be a Grid or None, got {self.grid!r}")


def check_mach_regime(action: Any) -> None:
    """Refuse a regime that would change the action's place in the action
    space: its normalizer must make the same part, of the same size and range,
    as the action's own - and reuse the action's own only where it has no unit."""
    regime = action.above_crossover
    if regime is None:
        return
    if not isinstance(regime, MachRegime):
        raise TypeError(
            f"{type(action).__name__} above_crossover must be a MachRegime or None, got {regime!r}"
        )
    name = type(action).__name__
    own, mach = action.normalizer, regime.normalizer
    if mach is None:
        if getattr(own, "step", None) is not None and action.grid is None:
            raise ValueError(
                f"{name}'s {type(own).__name__} steps in knots: give its MachRegime a "
                "normalizer of its own"
            )
        if getattr(own, "discrete", False) and regime.grid is None:
            raise ValueError(
                f"{name} steps along its grid: give its MachRegime a grid (whole Mach "
                "steps) to step along above crossover"
            )
        return
    if own is None:
        raise ValueError(
            f"{name} has no normalizer, so its MachRegime cannot have one: the action "
            "space would change shape at the crossover"
        )
    view = _MachView(action)
    if action_kind(action) != action_kind(view) or own.output_size(action) != mach.output_size(view):
        raise ValueError(
            f"{name}'s MachRegime normalizer {type(mach).__name__} puts it in another "
            f"part of the action space than its {type(own).__name__}"
        )
    if getattr(own, "discrete", False):
        same = own.n_choices == mach.n_choices
    else:
        same = own.output_bounds(action) == mach.output_bounds(view)
    if not same:
        raise ValueError(
            f"{name}'s MachRegime normalizer {type(mach).__name__} has another range "
            f"than its {type(own).__name__}: the action space would change at the crossover"
        )


# The aircraft each crossover speed action has commanded, by action: whose
# speed hold it hands over at the crossover.
_COMMANDED: dict[int, set[str]] = {}


@on_reset
def _reset_commanded(_seed: int | None = None) -> None:
    _COMMANDED.clear()


@dataclass(frozen=True)
class _SpeedAboveCrossover(_CrossoverSpeedAxis):
    """A crossover speed action that, given a :class:`MachRegime`, acts in Mach
    above its crossover (see the module)."""

    above_crossover: Annotated[
        MachRegime | None,
        "how it acts above a crossover: in Mach, with its own bounds, normalizer "
        "and grid; None = in knots at every level",
    ] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        check_mach_regime(self)

    def acting(self, idx: int) -> Any:
        """Itself below the crossover; above it, itself acting in Mach."""
        regime = self.above_crossover
        if regime is None or not bool(regime.crossover.above(idx)[0]):
            return self
        return _MachView(self)

    def set(self, idx: int, value: float) -> None:
        acting = self.acting(idx)
        if acting is not self:
            acting.set(idx, value)
            return
        super().set(idx, value)

    # ---- the hand-over at the crossover (see the module) ------------------- #

    def on_action_applied(self, acid: str, action) -> None:
        regime = self.above_crossover
        if regime is not None and regime.handover:
            _COMMANDED.setdefault(id(self), set()).add(acid)

    def on_step(self, ctx) -> None:
        """Hand over the speed holds of the aircraft it commands that passed
        the crossover this step."""
        regime = self.above_crossover
        commanded = _COMMANDED.get(id(self))
        if regime is None or not regime.handover or not commanded:
            return
        index = {acid: i for i, acid in enumerate(ctx.ids)}
        indices = [index[a] for a in commanded if a in index]
        for i, speed in regime.crossover.handover(indices):
            text = f"{speed:.4f}" if speed < 1.0 else f"{speed:.6f}"
            bs.stack.stack(f"SPD {bs.traf.id[i]} {text}")

    def on_aircraft_removed(self, acid: str) -> None:
        _COMMANDED.get(id(self), set()).discard(acid)

    def on_episode_reset(self, seed: int | None = None) -> None:
        _COMMANDED.pop(id(self), None)


class _MachView:
    """A crossover speed action as it acts above its crossover: the same
    action, its value a change in Mach - what a normalizer and a grid read
    (``bounds``, ``reach``, ``nominal``) and what it commands (``set``)."""

    def __init__(self, action: Any) -> None:
        regime: MachRegime = action.above_crossover
        self._action = action
        self.meta = dataclasses.replace(action.meta, unit=Unit.UNITLESS)
        self.kind = action.kind
        self.low = regime.low
        self.high = regime.high
        self.normalizer = action.normalizer if regime.normalizer is None else regime.normalizer
        self.grid = regime.grid

    @property
    def bounds_overridden(self) -> bool:
        return self.low is not None

    def acting(self, idx: int) -> _MachView:
        return self

    def nominal(self, idx: int) -> float:
        """The Mach a zero change holds: the nominal CAS as Mach here, at most Mmo."""
        cas_ms = self._action._nominal(idx) / self._action._scale
        mach = float(vcas2mach(cas_ms, float(bs.traf.alt[idx])))
        return min(mach, float(bs.traf.perf.mmo[idx]))

    def _commandable(self, idx: int) -> tuple[float, float]:
        """The Machs the aircraft may be commanded here: its minimum speed to
        Mmo or its maximum CAS, whichever is lower, within the action's command
        floor and ceiling (knots, as Mach)."""
        alt = float(bs.traf.alt[idx])
        low = float(vcas2mach(float(bs.traf.perf.vmin[idx]), alt))
        high = min(float(bs.traf.perf.mmo[idx]), float(vcas2mach(float(bs.traf.perf.vmax[idx]), alt)))
        action = self._action
        if action.command_floor is not None:
            low = max(low, float(vcas2mach(_to_ms(action, action.command_floor), alt)))
        if action.command_ceiling is not None:
            high = min(high, float(vcas2mach(_to_ms(action, action.command_ceiling), alt)))
        if high <= low:
            high = low + _MIN_DYNAMIC_SPAN
        return low, high

    def bounds(self, idx: int) -> tuple[float, float]:
        if self.bounds_overridden:
            return float(self.low), float(self.high)
        # Symmetric about the nominal, as the knots' are (see _reachable_delta).
        nominal = self.nominal(idx)
        low, high = self._commandable(idx)
        anchor = min(max(nominal, low), high)
        delta_low, delta_high = _reachable_delta(low, high, anchor)
        shift = anchor - nominal
        return delta_low + shift, delta_high + shift

    def reach(self, idx: int) -> tuple[float, float]:
        low, high = self._commandable(idx)
        nominal = self.nominal(idx)
        low, high = low - nominal, high - nominal
        if self.bounds_overridden:
            low, high = max(low, float(self.low)), min(high, float(self.high))
        return low, high

    def set(self, idx: int, value: float) -> None:
        low, high = self._commandable(idx)
        mach = min(max(self.nominal(idx) + float(value), low), high)
        bs.stack.stack(f"SPD {bs.traf.id[idx]} {mach:.4f}")


def _to_ms(action: Any, value: float) -> float:
    """A speed in the action's unit as m/s."""
    return float(value) / action._scale
