from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

import bluesky as bs
from bluesky.tools.aero import ft, kts
from bluesky.tools.geo import kwikqdrdist

from bluesky_sandbox.sim.performance.speeds import cas_ceiling_ms as _cas_ceiling_ms
from bluesky_sandbox.sim.performance.speeds import crossover_speed_state

from ._common import (
    _M_TO_FT,
    _MIN_DYNAMIC_SPAN,
    _MS_TO_KTS,
    _InFeet,
    _InKnots,
    _InMeters,
    _InMetersPerSecond,
)
from ._route import _active_route_waypoint
from ._state import record_comm_message
from .base import (
    ActionField,
    ActionMeta,
    ActionMode,
    ControlAxis,
    SwitchActionMixin,
    Unit,
)

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


def _switch(enabled: bool) -> str:
    return "ON" if enabled else "OFF"


@dataclass(frozen=True)
class HdgDeg(ActionField):
    """Set target heading in degrees.

    Metadata:
        name: hdg_deg
        unit: deg
        control_axis: heading
        mode: absolute
    """

    meta = ActionMeta(
        "hdg_deg",
        Unit.DEG,
        control_axis=ControlAxis.HEADING,
        mode=ActionMode.ABSOLUTE,
    )
    low: Annotated[float, "heading degrees"] = 0.0
    high: Annotated[float, "heading degrees"] = 360.0

    def set(self, idx: int, value: float) -> None:
        bs.stack.stack(f"HDG {bs.traf.id[idx]} {value:{_FMT}}")

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


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


@dataclass(frozen=True)
class SpdKts(_InKnots, _SpeedAxis, _AbsoluteTarget):
    """Set target calibrated airspeed in knots.

    Metadata:
        name: spd_kts
        unit: kts
        control_axis: speed
        mode: absolute
        dynamic_bounds: True

    ``low=None`` and ``high=None`` mean action bounds are read from BlueSky's
    current aircraft performance envelope at runtime.
    """

    meta = ActionMeta(
        "spd_kts",
        Unit.KTS,
        control_axis=ControlAxis.SPEED,
        mode=ActionMode.ABSOLUTE,
        dynamic_bounds=True,
    )
    low: Annotated[
        float | None,
        "target CAS knots lower bound; None = BlueSky perf.vmin at runtime",
    ] = None
    high: Annotated[
        float | None,
        "target CAS knots upper bound; None = BlueSky perf.vmax at runtime",
    ] = None


@dataclass(frozen=True)
class SpdMs(_InMetersPerSecond, _SpeedAxis, _AbsoluteTarget):
    """Set target calibrated airspeed in m/s.

    Metadata:
        name: spd_ms
        unit: m/s
        control_axis: speed
        mode: absolute
        dynamic_bounds: True

    ``low=None`` and ``high=None`` mean action bounds are read from BlueSky's
    current aircraft performance envelope at runtime.
    """

    meta = ActionMeta(
        "spd_ms",
        Unit.M_PER_S,
        control_axis=ControlAxis.SPEED,
        mode=ActionMode.ABSOLUTE,
        dynamic_bounds=True,
    )
    low: Annotated[
        float | None,
        "target CAS m/s lower bound; None = BlueSky perf.vmin at runtime",
    ] = None
    high: Annotated[
        float | None,
        "target CAS m/s upper bound; None = BlueSky perf.vmax at runtime",
    ] = None


@dataclass(frozen=True)
class AltFt(_InFeet, _AltitudeAxis, _AbsoluteTarget):
    """Set target altitude in feet.

    Metadata:
        name: alt_ft
        unit: ft
        control_axis: altitude
        mode: absolute
        dynamic_bounds: True

    ``low=None`` and ``high=None`` mean action bounds are read from BlueSky's
    current aircraft performance envelope at runtime.
    """

    meta = ActionMeta(
        "alt_ft",
        Unit.FT,
        control_axis=ControlAxis.ALTITUDE,
        mode=ActionMode.ABSOLUTE,
        dynamic_bounds=True,
    )
    low: Annotated[
        float | None,
        "target altitude feet lower bound; None = 0 ft at runtime",
    ] = None
    high: Annotated[
        float | None,
        "target altitude feet upper bound; None = BlueSky perf ceiling at runtime",
    ] = None


@dataclass(frozen=True)
class AltM(_InMeters, _AltitudeAxis, _AbsoluteTarget):
    """Set target altitude in meters.

    Metadata:
        name: alt_m
        unit: m
        control_axis: altitude
        mode: absolute
        dynamic_bounds: True

    ``low=None`` and ``high=None`` mean action bounds are read from BlueSky's
    current aircraft performance envelope at runtime.
    """

    meta = ActionMeta(
        "alt_m",
        Unit.M,
        control_axis=ControlAxis.ALTITUDE,
        mode=ActionMode.ABSOLUTE,
        dynamic_bounds=True,
    )
    low: Annotated[
        float | None,
        "target altitude meters lower bound; None = 0 m at runtime",
    ] = None
    high: Annotated[
        float | None,
        "target altitude meters upper bound; None = BlueSky perf ceiling at runtime",
    ] = None


@dataclass(frozen=True)
class HdgDeltaDeg(ActionField):
    """Adjust heading by a delta in degrees.

    Metadata:
        name: hdg_delta_deg
        unit: deg
        control_axis: heading
        mode: delta
    """

    meta = ActionMeta(
        "hdg_delta_deg",
        Unit.DEG,
        control_axis=ControlAxis.HEADING,
        mode=ActionMode.DELTA,
    )
    low: Annotated[float, "heading delta degrees"] = -180.0
    high: Annotated[float, "heading delta degrees"] = 180.0

    def set(self, idx: int, value: float) -> None:
        bs.stack.stack(
            f"HDG {bs.traf.id[idx]} {(bs.traf.hdg[idx] + value) % 360.0:{_FMT}}"
        )

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class ApHdgDeltaDeg(ActionField):
    """Set autopilot selected heading relative to current track.

    Metadata:
        name: ap_hdg_delta_deg
        unit: deg
        control_axis: heading
        mode: delta
        dynamic_bounds: True
    """

    meta = ActionMeta(
        "ap_hdg_delta_deg",
        Unit.DEG,
        control_axis=ControlAxis.HEADING,
        mode=ActionMode.DELTA,
        dynamic_bounds=True,
    )
    low: Annotated[
        float | None,
        "autopilot heading offset degrees; None = full turn range",
    ] = None
    high: Annotated[
        float | None,
        "autopilot heading offset degrees; None = full turn range",
    ] = None

    def set(self, idx: int, value: float) -> None:
        target = (bs.traf.trk[idx] + value) % 360.0
        bs.stack.stack(f"HDG {bs.traf.id[idx]} {target:{_FMT}}")

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(lambda: (-180.0, 180.0))


@dataclass(frozen=True)
class AltDeltaFt(_InFeet, _AltitudeAxis, _DeltaTarget):
    """Adjust target altitude by a delta in feet.

    Metadata:
        name: alt_delta_ft
        unit: ft
        control_axis: altitude
        mode: delta
    """

    meta = ActionMeta(
        "alt_delta_ft",
        Unit.FT,
        control_axis=ControlAxis.ALTITUDE,
        mode=ActionMode.DELTA,
    )
    low: Annotated[float, "altitude delta feet"] = -1000.0
    high: Annotated[float, "altitude delta feet"] = 1000.0


@dataclass(frozen=True)
class ApAltDeltaFt(_InFeet, _AltitudeAxis, _DeltaTarget):
    """Set autopilot selected altitude relative to current altitude.

    Metadata:
        name: ap_alt_delta_ft
        unit: ft
        control_axis: altitude
        mode: delta
        dynamic_bounds: True
    """

    meta = ActionMeta(
        "ap_alt_delta_ft",
        Unit.FT,
        control_axis=ControlAxis.ALTITUDE,
        mode=ActionMode.DELTA,
        dynamic_bounds=True,
    )
    low: Annotated[
        float | None,
        "autopilot altitude offset feet; None = runtime altitude envelope",
    ] = None
    high: Annotated[
        float | None,
        "autopilot altitude offset feet; None = runtime altitude envelope",
    ] = None


@dataclass(frozen=True)
class AltDeltaM(_InMeters, _AltitudeAxis, _DeltaTarget):
    """Adjust target altitude by a delta in meters.

    Metadata:
        name: alt_delta_m
        unit: m
        control_axis: altitude
        mode: delta
    """

    meta = ActionMeta(
        "alt_delta_m",
        Unit.M,
        control_axis=ControlAxis.ALTITUDE,
        mode=ActionMode.DELTA,
    )
    low: Annotated[float, "altitude delta meters"] = -1000.0 * ft
    high: Annotated[float, "altitude delta meters"] = 1000.0 * ft


@dataclass(frozen=True)
class ApAltDeltaM(_InMeters, _AltitudeAxis, _DeltaTarget):
    """Set autopilot selected altitude relative to current altitude."""

    meta = ActionMeta(
        "ap_alt_delta_m",
        Unit.M,
        control_axis=ControlAxis.ALTITUDE,
        mode=ActionMode.DELTA,
        dynamic_bounds=True,
    )
    low: Annotated[
        float | None,
        "autopilot altitude offset meters; None = runtime altitude envelope",
    ] = None
    high: Annotated[
        float | None,
        "autopilot altitude offset meters; None = runtime altitude envelope",
    ] = None


@dataclass(frozen=True)
class SpdDeltaKts(_InKnots, _SpeedAxis, _DeltaTarget):
    """Adjust target calibrated airspeed by a delta in knots.

    Metadata:
        name: spd_delta_kts
        unit: kts
        control_axis: speed
        mode: delta
    """

    meta = ActionMeta(
        "spd_delta_kts",
        Unit.KTS,
        control_axis=ControlAxis.SPEED,
        mode=ActionMode.DELTA,
    )
    low: Annotated[float, "CAS delta knots"] = -100.0
    high: Annotated[float, "CAS delta knots"] = 100.0


@dataclass(frozen=True)
class ApSpdDeltaKts(_InKnots, _SpeedAxis, _DeltaTarget):
    """Set autopilot selected calibrated airspeed relative to current CAS.

    Metadata:
        name: ap_spd_delta_kts
        unit: kts
        control_axis: speed
        mode: delta
        dynamic_bounds: True
    """

    meta = ActionMeta(
        "ap_spd_delta_kts",
        Unit.KTS,
        control_axis=ControlAxis.SPEED,
        mode=ActionMode.DELTA,
        dynamic_bounds=True,
    )
    low: Annotated[
        float | None,
        "autopilot CAS offset knots; None = runtime speed envelope",
    ] = None
    high: Annotated[
        float | None,
        "autopilot CAS offset knots; None = runtime speed envelope",
    ] = None


@dataclass(frozen=True)
class SpdDeltaMs(_InMetersPerSecond, _SpeedAxis, _DeltaTarget):
    """Adjust target calibrated airspeed by a delta in m/s.

    Metadata:
        name: spd_delta_ms
        unit: m/s
        control_axis: speed
        mode: delta
    """

    meta = ActionMeta(
        "spd_delta_ms",
        Unit.M_PER_S,
        control_axis=ControlAxis.SPEED,
        mode=ActionMode.DELTA,
    )
    low: Annotated[float, "CAS delta m/s"] = -100.0 * kts
    high: Annotated[float, "CAS delta m/s"] = 100.0 * kts


# --------------------------------------------------------------------------- #
# Deviation-from-nominal: command relative to the active route waypoint        #
# guidance (the LNAV/VNAV target), so value 0 flies the nominal. Falls back to  #
# the aircraft's current state when there is no active waypoint / constraint,   #
# so 0 still means "hold".                                                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ActiveRouteWaypointHdgDeltaDeg(ActionField):
    """Steer relative to the bearing toward the active route waypoint.

    Commands ``bearing_to_active_waypoint + value`` (degrees); ``value == 0``
    flies straight at the waypoint. Falls back to current track when there is
    no active waypoint.

    Metadata:
        name: active_route_waypoint_hdg_delta_deg
        unit: deg
        control_axis: heading
        mode: delta
    """

    meta = ActionMeta(
        "active_route_waypoint_hdg_delta_deg",
        Unit.DEG,
        control_axis=ControlAxis.HEADING,
        mode=ActionMode.DELTA,
    )
    low: Annotated[float, "heading delta degrees from waypoint bearing"] = -180.0
    high: Annotated[float, "heading delta degrees from waypoint bearing"] = 180.0

    def set(self, idx: int, value: float) -> None:
        wp = _active_route_waypoint(idx)
        if wp is None:
            nominal = float(bs.traf.trk[idx])
        else:
            qdr, _dist = kwikqdrdist(
                float(bs.traf.lat[idx]), float(bs.traf.lon[idx]), wp[0], wp[1]
            )
            nominal = float(qdr)
        bs.stack.stack(f"HDG {bs.traf.id[idx]} {(nominal + value) % 360.0:{_FMT}}")

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class ActiveRouteWaypointAltDeltaFt(
    _InFeet, _FromRouteWaypoint, _AltitudeAxis, _DeltaTarget
):
    """Command altitude relative to the active route waypoint's altitude.

    Commands ``waypoint_altitude + value`` (feet); ``value == 0`` targets the
    waypoint's altitude constraint. Falls back to current altitude when the
    waypoint has no altitude constraint. The result is clamped to ``[0, ceiling]``.

    Metadata:
        name: active_route_waypoint_alt_delta_ft
        unit: ft
        control_axis: altitude
        mode: delta
        dynamic_bounds: True

    ``low=None`` and ``high=None`` mean the offset bounds are resolved at
    runtime as a symmetric span around the nominal, reaching both ``0`` and the
    aircraft altitude ceiling. Pair with a normalizer for a fixed action space.
    """

    meta = ActionMeta(
        "active_route_waypoint_alt_delta_ft",
        Unit.FT,
        control_axis=ControlAxis.ALTITUDE,
        mode=ActionMode.DELTA,
        dynamic_bounds=True,
    )
    low: Annotated[
        float | None,
        "altitude delta feet from waypoint altitude; None = runtime envelope",
    ] = None
    high: Annotated[
        float | None,
        "altitude delta feet from waypoint altitude; None = runtime envelope",
    ] = None


@dataclass(frozen=True)
class ActiveRouteWaypointSpdDeltaKts(
    _InKnots, _FromRouteWaypoint, _SpeedAxis, _DeltaTarget
):
    """Command CAS relative to the active route waypoint's speed constraint.

    Commands ``waypoint_speed + value`` (knots); ``value == 0`` targets the
    waypoint's speed constraint. Falls back to current CAS when the waypoint
    has no speed constraint. The result is clamped to the aircraft's
    performance speed envelope.

    Metadata:
        name: active_route_waypoint_spd_delta_kts
        unit: kts
        control_axis: speed
        mode: delta
        dynamic_bounds: True

    ``low=None`` and ``high=None`` mean the offset bounds are resolved at
    runtime as a symmetric span around the nominal, reaching both the minimum
    and maximum operating speed. Pair with a normalizer for a fixed action space.
    """

    meta = ActionMeta(
        "active_route_waypoint_spd_delta_kts",
        Unit.KTS,
        control_axis=ControlAxis.SPEED,
        mode=ActionMode.DELTA,
        dynamic_bounds=True,
    )
    low: Annotated[
        float | None,
        "CAS delta knots from waypoint speed; None = runtime speed envelope",
    ] = None
    high: Annotated[
        float | None,
        "CAS delta knots from waypoint speed; None = runtime speed envelope",
    ] = None


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


@dataclass(frozen=True)
class ActiveRouteWaypointSpdDeltaCrossover(
    _InKnots, _FromRouteWaypoint, _CrossoverSpeedAxis, _DeltaTarget
):
    """Regime-aware speed command relative to the waypoint's speed constraint.

    Like :class:`ActiveRouteWaypointSpdDeltaKts`, but honors the CAS/Mach
    crossover so the command is always *feasible* and holds the right quantity
    per regime. The CAS target is capped at the Mach limit (Mmo) expressed as CAS
    for the current altitude, and the command is issued as **Mach** above the
    crossover altitude and **CAS** below. This avoids the plain-CAS action's
    failure mode at cruise, where a commanded CAS silently exceeds Mmo and
    saturates - and it keeps the action's upper bound tracking the *achievable*
    speed as altitude changes.

    Metadata:
        name: active_route_waypoint_spd_delta_crossover
        unit: kts
        control_axis: speed
        mode: delta
        dynamic_bounds: True

    ``low=None`` / ``high=None`` resolve the offset bounds at runtime as a
    symmetric span around the nominal reaching the minimum operating speed and
    the (Mach-limited) maximum. Pair with a normalizer for a fixed action space.
    """

    meta = ActionMeta(
        "active_route_waypoint_spd_delta_crossover",
        Unit.KTS,
        control_axis=ControlAxis.SPEED,
        mode=ActionMode.DELTA,
        dynamic_bounds=True,
    )
    low: Annotated[
        float | None,
        "CAS delta knots from waypoint speed; None = runtime speed envelope",
    ] = None
    high: Annotated[
        float | None,
        "CAS delta knots from waypoint speed; None = runtime speed envelope",
    ] = None


@dataclass(frozen=True)
class ApSpdDeltaCrossover(_InKnots, _CrossoverSpeedAxis, _DeltaTarget):
    """Autopilot speed relative to *current* CAS, regime-aware (CAS/Mach crossover).

    The autopilot counterpart of :class:`ActiveRouteWaypointSpdDeltaCrossover`:
    the nominal is the aircraft's current CAS (not a waypoint constraint), so the
    action nudges speed from where it is. The CAS target is capped at the
    altitude's Mach limit and issued as **Mach** above the crossover altitude /
    **CAS** below - so it never commands a Mach-exceeding CAS at cruise.

    Metadata:
        name: ap_spd_delta_crossover
        unit: kts
        control_axis: speed
        mode: delta
        dynamic_bounds: True
    """

    meta = ActionMeta(
        "ap_spd_delta_crossover",
        Unit.KTS,
        control_axis=ControlAxis.SPEED,
        mode=ActionMode.DELTA,
        dynamic_bounds=True,
    )
    low: Annotated[
        float | None,
        "autopilot CAS offset knots; None = runtime speed envelope",
    ] = None
    high: Annotated[
        float | None,
        "autopilot CAS offset knots; None = runtime speed envelope",
    ] = None


@dataclass(frozen=True)
class _AutopilotHoldSwitch(SwitchActionMixin, ActionField):
    """Autopilot switch with ON/OFF commands and a no-op hold band."""

    switch_names: ClassVar[tuple[str, ...]] = ()
    low: Annotated[float, "switch scalar"] = 0.0
    high: Annotated[float, "switch scalar"] = 1.0
    threshold: Annotated[float, "switch ON threshold"] = 0.5
    off_threshold: Annotated[float, "switch OFF threshold"] = 0.2

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.low <= self.off_threshold <= self.switch_on_value() <= self.high:
            raise ValueError(
                f"{self.__class__.__name__} requires "
                "low <= off_threshold <= threshold <= high"
            )

    def switch_on_value(self) -> float:
        return float(self.threshold)

    def switch_command(self, value: float) -> bool | None:
        if value >= self.switch_on_value():
            return True
        if value <= self.off_threshold:
            return False
        return None

    @staticmethod
    def capture_lnav_reference(idx: int) -> None:
        # Neutralize LNAV OFF so selected-heading mode starts from current track.
        bs.traf.ap.trk[idx] = bs.traf.trk[idx]

    @staticmethod
    def capture_vnav_reference(idx: int) -> None:
        # Neutralize VNAV OFF so selected speed/altitude start from current state.
        bs.traf.selspd[idx] = bs.traf.cas[idx]
        bs.traf.selalt[idx] = bs.traf.alt[idx]

    def capture_off_reference(self, idx: int) -> None:
        return None

    def set(self, idx: int, value: float) -> None:
        command = self.switch_command(value)
        if command is None:
            return
        acid = bs.traf.id[idx]
        if command is False:
            self.capture_off_reference(idx)
        state = _switch(command)
        for switch_name in self.switch_names:
            bs.stack.stack(f"{switch_name} {acid} {state}")

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class AutopilotLnav(_AutopilotHoldSwitch):
    """Command BlueSky LNAV with an ON/OFF hold band.

    Metadata:
        name: autopilot_lnav
        unit: switch
        control_axis: autopilot
        mode: switch
    """

    meta = ActionMeta(
        "autopilot_lnav",
        Unit.SWITCH,
        control_axis=ControlAxis.AUTOPILOT,
        mode=ActionMode.SWITCH,
        suppresses_when_on=(ControlAxis.HEADING,),
    )
    switch_names: ClassVar[tuple[str, ...]] = ("LNAV",)

    def current_switch_state(self, idx: int) -> bool:
        return bool(bs.traf.swlnav[idx])

    def capture_off_reference(self, idx: int) -> None:
        self.capture_lnav_reference(idx)


@dataclass(frozen=True)
class AutopilotVnav(_AutopilotHoldSwitch):
    """Command BlueSky VNAV with an ON/OFF hold band.

    Metadata:
        name: autopilot_vnav
        unit: switch
        control_axis: autopilot
        mode: switch
        requires_on: autopilot_lnav
    """

    meta = ActionMeta(
        "autopilot_vnav",
        Unit.SWITCH,
        control_axis=ControlAxis.AUTOPILOT,
        mode=ActionMode.SWITCH,
        requires_on=("autopilot_lnav",),
        suppresses_when_on=(ControlAxis.SPEED, ControlAxis.ALTITUDE),
    )
    switch_names: ClassVar[tuple[str, ...]] = ("VNAV",)

    def current_switch_state(self, idx: int) -> bool:
        return bool(bs.traf.swvnav[idx])

    def capture_off_reference(self, idx: int) -> None:
        self.capture_vnav_reference(idx)


@dataclass(frozen=True)
class AutopilotLnavVnav(_AutopilotHoldSwitch):
    """Command BlueSky LNAV and VNAV together with an ON/OFF hold band.

    Metadata:
        name: autopilot_lnav_vnav
        unit: switch
        control_axis: autopilot
        mode: switch
    """

    meta = ActionMeta(
        "autopilot_lnav_vnav",
        Unit.SWITCH,
        control_axis=ControlAxis.AUTOPILOT,
        mode=ActionMode.SWITCH,
        suppresses_when_on=(
            ControlAxis.HEADING,
            ControlAxis.SPEED,
            ControlAxis.ALTITUDE,
        ),
    )
    switch_names: ClassVar[tuple[str, ...]] = ("LNAV", "VNAV")

    def current_switch_state(self, idx: int) -> bool:
        return bool(bs.traf.swlnav[idx]) and bool(bs.traf.swvnav[idx])

    def capture_off_reference(self, idx: int) -> None:
        self.capture_lnav_reference(idx)
        self.capture_vnav_reference(idx)


@dataclass(frozen=True)
class CommBroadcast(ActionField):
    """Broadcast one channel of a learned communication message.

    Stores the (squashed, ``[-1, 1]``) action value in the per-process comm
    registry under this aircraft's callsign; other agents read it back on the
    next step through ``IntruderCommMessage`` with the same ``channel``. No
    aircraft-control effect - a pure signaling channel whose meaning the
    shared policy must learn (emergent communication). Exclude these dims from
    any action-magnitude penalty, or the reward will train the channel silent.

    Metadata:
        name: comm_broadcast
        unit: unitless
    """

    meta = ActionMeta(
        "comm_broadcast",
        Unit.UNITLESS,
    )
    channel: Annotated[int, "message channel index"] = 0
    low: Annotated[float, "message value"] = -1.0
    high: Annotated[float, "message value"] = 1.0

    def set(self, idx: int, value: float) -> None:
        record_comm_message(idx, self.channel, float(value))

    def bounds(self, idx: int) -> tuple[float, float]:
        del idx
        return float(self.low), float(self.high)
