"""What the speed and altitude actions are built from.

An action commanding a speed or an altitude is an axis (:class:`_SpeedAxis`,
:class:`_AltitudeAxis`, ...), which works in SI, plus a unit mixin and a kind:
absolute (:class:`_AbsoluteTarget`) or a delta from a nominal
(:class:`_DeltaTarget`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

import bluesky as bs
from bluesky.tools.aero import kts

from bluesky_sandbox.sim.performance.speeds import cas_ceiling_ms as _cas_ceiling_ms
from bluesky_sandbox.sim.performance.speeds import crossover_speed_state

from .._common import _M_TO_FT, _MIN_DYNAMIC_SPAN, _MS_TO_KTS
from .._route import _active_route_waypoint
from ..base import ActionField

_FMT = ".6f"


def _clip(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))


def _reachable_delta(lo: float, hi: float, nominal: float) -> tuple[float, float]:
    """Symmetric delta bounds that keep ``a = 0`` meaning "fly the nominal".

    An ASYMMETRIC ``(lo - nominal, hi - nominal)`` spans the reachable set exactly
    and removes the dead zone - but ``SymmetricNormalizer`` maps ``a = 0`` to the
    interval MIDPOINT, so a neutral action stops meaning ``delta = 0``. On
    safe_rl_v38d that put the zero action 16,176 ft BELOW the waypoint. The whole
    design leans on "0 is the nominal" (the reward's framing, the zero-action
    baseline, a beta actor's ``alpha = beta`` init), so that invariant wins.

    Symmetric about the nominal at the INSCRIBED half-range: every command is
    reachable (no dead zone) and ``a = 0`` is exactly the nominal. The cost is
    reach on the wider side - here the descent range is capped at the climb range.
    Restoring the full asymmetric span needs a signed normalizer that maps
    ``[-1, 0] -> [lo, 0]`` and ``[0, 1] -> [0, hi]``; until then, correctness of
    the neutral point beats a wider one-sided range.
    """
    half = max(min(nominal - lo, hi - nominal), _MIN_DYNAMIC_SPAN)
    return -half, half


@dataclass(frozen=True)
class _TargetAction(ActionField):
    """An action that commands a target on a linear axis: a speed or an altitude.

    Built like the unit observation fields: the axis (:class:`_SpeedAxis`,
    :class:`_AltitudeAxis`, ...) works in SI, a unit mixin (``_InKnots``,
    ``_InFeet``, ...) sets ``_scale``, and the action takes its values - and its
    bounds, floor and ceiling - in that unit. ``_command`` converts the target
    to the unit BlueSky's command expects.

    ``command_floor`` / ``command_ceiling`` bound the TARGET sent to BlueSky,
    not the action value: "never command below 1,000 ft" holds whether the
    action is absolute or a delta. Where they leave nothing reachable - an
    aircraft already below the floor - the floor wins, so the command is never
    below it.
    """

    _scale: ClassVar[float] = 1.0

    command_floor: Annotated[
        float | None, "lowest target ever commanded, in this action's unit; None = none"
    ] = None
    command_ceiling: Annotated[
        float | None,
        "highest target ever commanded, in this action's unit; None = none",
    ] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("command_floor", "command_ceiling"):
            value = getattr(self, name)
            if value is not None and value < 0.0:
                raise ValueError(
                    f"{type(self).__name__} {name} must be >= 0.0 or None, got {value}"
                )
        if (
            self.command_floor is not None
            and self.command_ceiling is not None
            and self.command_floor > self.command_ceiling
        ):
            raise ValueError(
                f"{type(self).__name__} command_floor ({self.command_floor}) must "
                f"not exceed command_ceiling ({self.command_ceiling})"
            )

    # --- the axis: SI, provided by _SpeedAxis / _AltitudeAxis ---------------
    def _envelope_si(self, idx: int) -> tuple[float, float]:
        raise NotImplementedError

    def _current_si(self, idx: int) -> float:
        raise NotImplementedError

    def _command(self, idx: int, target: float) -> None:
        raise NotImplementedError

    # --- in this action's unit ------------------------------------------------
    def _convert(self, si: float) -> float:
        return si * self._scale

    def _limit(self, low: float, high: float) -> tuple[float, float]:
        """``(low, high)`` narrowed to the command floor and ceiling; where
        nothing is left, the floor wins."""
        if self.command_floor is not None:
            low = max(low, float(self.command_floor))
        if self.command_ceiling is not None:
            high = min(high, float(self.command_ceiling))
        if high <= low:
            high = low + _MIN_DYNAMIC_SPAN
        return low, high

    def _commandable(self, idx: int) -> tuple[float, float]:
        """The targets this aircraft may be commanded: its envelope, limited."""
        low, high = self._envelope_si(idx)
        return self._limit(self._convert(low), self._convert(high))

    def _nominal(self, idx: int) -> float:
        """What a zero delta means: the current value, unless overridden."""
        return self._convert(self._current_si(idx))


@dataclass(frozen=True)
class _AbsoluteTarget(_TargetAction):
    """The action value IS the target."""

    def set(self, idx: int, value: float) -> None:
        low, high = self.bounds(idx)
        self._command(idx, _clip(value, low, high))

    def bounds(self, idx: int) -> tuple[float, float]:
        low, high = self._dynamic_or_configured_bounds(
            lambda: tuple(self._convert(v) for v in self._envelope_si(idx))
        )
        return self._limit(low, high)


@dataclass(frozen=True)
class _DeltaTarget(_TargetAction):
    """The action value is added to a nominal (``_nominal``) to make the target."""

    def set(self, idx: int, value: float) -> None:
        low, high = self._commandable(idx)
        target = self._nominal(idx) + value
        self._command(idx, min(max(target, low), high))

    def bounds(self, idx: int) -> tuple[float, float]:
        def resolve() -> tuple[float, float]:
            # Symmetric about the nominal (see _reachable_delta), within what may
            # be commanded - so a floor narrows it on both sides, keeping 0 the
            # nominal. A nominal outside that - an aircraft above its ceiling or
            # below the floor - anchors at the nearest limit, and every value
            # commands that limit.
            nominal = self._nominal(idx)
            low, high = self._commandable(idx)
            anchor = min(max(nominal, low), high)
            delta_low, delta_high = _reachable_delta(low, high, anchor)
            shift = anchor - nominal
            return delta_low + shift, delta_high + shift

        return self._dynamic_or_configured_bounds(resolve)


class _SpeedAxis:
    """Calibrated airspeed: the performance envelope, commanded with ``SPD`` (kts)."""

    _route_constraint: ClassVar[int] = 3  # waypoint speed, m/s

    def _envelope_si(self, idx: int) -> tuple[float, float]:
        return float(bs.traf.perf.vmin[idx]), float(bs.traf.perf.vmax[idx])

    def _current_si(self, idx: int) -> float:
        return float(bs.traf.cas[idx])

    def _command(self, idx: int, target: float) -> None:
        target_kts = target * (_MS_TO_KTS / self._scale)
        bs.stack.stack(f"SPD {bs.traf.id[idx]} {target_kts:{_FMT}}")


class _CrossoverSpeedAxis(_SpeedAxis):
    """CAS capped at the Mach limit, commanded as Mach above the crossover."""

    def _envelope_si(self, idx: int) -> tuple[float, float]:
        return float(bs.traf.perf.vmin[idx]), float(_cas_ceiling_ms(idx))

    def _command(self, idx: int, target: float) -> None:
        _issue_crossover_speed(idx, target * (_MS_TO_KTS / self._scale))


class _AltitudeAxis:
    """Altitude: 0 to the performance ceiling, commanded with ``ALT`` (ft)."""

    _route_constraint: ClassVar[int] = 2  # waypoint altitude, m

    def _envelope_si(self, idx: int) -> tuple[float, float]:
        return 0.0, float(bs.traf.perf.hmax[idx])

    def _current_si(self, idx: int) -> float:
        return float(bs.traf.alt[idx])

    def _command(self, idx: int, target: float) -> None:
        target_ft = target * (_M_TO_FT / self._scale)
        bs.stack.stack(f"ALT {bs.traf.id[idx]} {target_ft:{_FMT}}")


class _FromRouteWaypoint:
    """A delta from the active route waypoint's constraint - or, where it has
    none, from the current value."""

    def _nominal(self, idx: int) -> float:
        wp = _active_route_waypoint(idx)
        constraint = None if wp is None else wp[self._route_constraint]
        si = self._current_si(idx) if constraint is None else constraint
        return self._convert(si)


def _issue_crossover_speed(idx: int, target_cas_kts: float) -> None:
    """Command a CAS target regime-aware: clamp to the feasible envelope, then
    issue **Mach** above the CAS/Mach crossover altitude and **CAS** below (so the
    autopilot holds the right quantity and never a Mach-exceeding CAS). BlueSky's
    SPD reads values below ``casmach_thr`` (~3.9 kt) as Mach. Regime and clamping
    are shared with the waypoint speed constraint via ``crossover_speed_state``."""
    state = crossover_speed_state(idx, target_cas_kts * kts)
    acid = bs.traf.id[idx]
    if state.in_mach:
        bs.stack.stack(f"SPD {acid} {state.target_mach:.4f}")
    else:
        bs.stack.stack(f"SPD {acid} {state.target_ms * _MS_TO_KTS:{_FMT}}")
