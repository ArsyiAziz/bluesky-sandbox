from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

import bluesky as bs
from bluesky.tools.aero import ft, kts
from bluesky.tools.geo import kwikqdrdist

from bluesky_sandbox.sim.performance.speeds import cas_ceiling_ms as _cas_ceiling_ms
from bluesky_sandbox.sim.performance.speeds import crossover_speed_state

from .base import (
    ActionField,
    ActionMeta,
    ActionMode,
    ControlAxis,
    SwitchActionMixin,
    Unit,
)
from .observations import _active_route_waypoint, record_comm_message

_M_TO_FT = 1.0 / ft
_MS_TO_KTS = 1.0 / kts
_MIN_DYNAMIC_SPAN = 1e-6
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
class SpdKts(ActionField):
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
    command_floor_kts: Annotated[
        float | None,
        "minimum CAS knots command sent through BlueSky SPD",
    ] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.command_floor_kts is not None and self.command_floor_kts < 0.0:
            raise ValueError(
                f"SpdKts command_floor_kts must be >= 0.0 or None, "
                f"got {self.command_floor_kts}"
            )

    def set(self, idx: int, value: float) -> None:
        lo, hi = self.bounds(idx)
        target = _clip(value, lo, hi)
        bs.stack.stack(f"SPD {bs.traf.id[idx]} {target:{_FMT}}")

    def bounds(self, idx: int) -> tuple[float, float]:
        low, high = self._dynamic_or_configured_bounds(
            lambda: (
                bs.traf.perf.vmin[idx] * _MS_TO_KTS,
                bs.traf.perf.vmax[idx] * _MS_TO_KTS,
            )
        )
        if self.command_floor_kts is None:
            return low, high
        floor = float(self.command_floor_kts)
        if floor > high:
            raise ValueError(
                "SpdKts command_floor_kts must not exceed the resolved upper "
                f"speed bound ({high:.3f} kt), got {floor:.3f} kt"
            )
        return max(low, floor), high


@dataclass(frozen=True)
class SpdMs(ActionField):
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

    def set(self, idx: int, value: float) -> None:
        lo, hi = self.bounds(idx)
        target = _clip(value, lo, hi)
        bs.stack.stack(f"SPD {bs.traf.id[idx]} {target * _MS_TO_KTS:{_FMT}}")

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(
            lambda: (bs.traf.perf.vmin[idx], bs.traf.perf.vmax[idx])
        )


@dataclass(frozen=True)
class _AltitudeAction(ActionField):
    """Shared helpers for altitude action fields."""

    @staticmethod
    def altitude_ceiling_m(idx: int) -> float:
        return bs.traf.perf.hmax[idx]

    @classmethod
    def altitude_ceiling_ft(cls, idx: int) -> float:
        return cls.altitude_ceiling_m(idx) * _M_TO_FT

    @staticmethod
    def command_altitude_ft(idx: int, altitude_ft: float) -> None:
        bs.stack.stack(f"ALT {bs.traf.id[idx]} {altitude_ft:{_FMT}}")

    # Instance methods (not classmethods): the floor comes from the field's own
    # ``min_altitude_ft``, which is per-instance configuration.
    def clip_altitude_m(self, idx: int, altitude_m: float) -> float:
        return min(
            max(altitude_m, self.altitude_floor_m()), self.altitude_ceiling_m(idx)
        )

    def clip_altitude_ft(self, idx: int, altitude_ft: float) -> float:
        return min(
            max(altitude_ft, self.altitude_floor_m() * _M_TO_FT),
            self.altitude_ceiling_ft(idx),
        )

    def altitude_floor_m(self) -> float:
        """Lowest commandable altitude, in metres.

        MUST match whatever floor the task enforces on ``bs.traf.selalt`` (tasks
        typically clamp a minimum-safe-altitude in ``on_sim_step``). If this is
        lower than the task's clamp, every command in the gap produces the SAME
        flown altitude, so that slice of the action range is a dead zone the
        policy gets no gradient in - and with envelope-relative bounds that slice
        can be large (~1/6 of the altitude channel at a 25000 ft rung with a
        1000 ft clamp). Set ``min_altitude_ft`` on the field to close the gap.
        """
        return float(getattr(self, "min_altitude_ft", 0.0) or 0.0) / _M_TO_FT

    def altitude_delta_bounds_ft(self, idx: int) -> tuple[float, float]:
        current_ft = bs.traf.alt[idx] * _M_TO_FT
        return _reachable_delta(
            self.altitude_floor_m() * _M_TO_FT,
            self.altitude_ceiling_ft(idx),
            current_ft,
        )

    def altitude_delta_bounds_m(self, idx: int) -> tuple[float, float]:
        current_m = bs.traf.alt[idx]
        return _reachable_delta(
            self.altitude_floor_m(), self.altitude_ceiling_m(idx), current_m
        )


@dataclass(frozen=True)
class AltFt(_AltitudeAction):
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

    def set(self, idx: int, value: float) -> None:
        lo, hi = self.bounds(idx)
        self.command_altitude_ft(idx, _clip(value, lo, hi))

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(
            lambda: (self.altitude_floor_m() * _M_TO_FT, self.altitude_ceiling_ft(idx))
        )


@dataclass(frozen=True)
class AltM(_AltitudeAction):
    """Set target altitude in metres.

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
        "target altitude metres lower bound; None = 0 m at runtime",
    ] = None
    high: Annotated[
        float | None,
        "target altitude metres upper bound; None = BlueSky perf ceiling at runtime",
    ] = None

    def set(self, idx: int, value: float) -> None:
        lo, hi = self.bounds(idx)
        self.command_altitude_ft(idx, _clip(value, lo, hi) * _M_TO_FT)

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(
            lambda: (self.altitude_floor_m(), self.altitude_ceiling_m(idx))
        )


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
class AltDeltaFt(_AltitudeAction):
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

    def set(self, idx: int, value: float) -> None:
        self.command_altitude_ft(idx, max(0.0, bs.traf.alt[idx] * _M_TO_FT + value))

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class ApAltDeltaFt(_AltitudeAction):
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
    command_floor_ft: Annotated[
        float | None,
        "minimum selected altitude feet; None = no command floor",
    ] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.command_floor_ft is not None and self.command_floor_ft < 0.0:
            raise ValueError(
                f"ApAltDeltaFt command_floor_ft must be >= 0.0 or None, "
                f"got {self.command_floor_ft}"
            )

    def set(self, idx: int, value: float) -> None:
        target = bs.traf.alt[idx] * _M_TO_FT + value
        if self.command_floor_ft is not None:
            target = max(float(self.command_floor_ft), target)
        self.command_altitude_ft(idx, self.clip_altitude_ft(idx, target))

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(
            lambda: self._altitude_delta_bounds_with_floor_ft(idx)
        )

    def _altitude_delta_bounds_with_floor_ft(self, idx: int) -> tuple[float, float]:
        low, high = self.altitude_delta_bounds_ft(idx)
        if self.command_floor_ft is None:
            return low, high
        current_ft = bs.traf.alt[idx] * _M_TO_FT
        return max(low, float(self.command_floor_ft) - current_ft), high


@dataclass(frozen=True)
class AltDeltaM(_AltitudeAction):
    """Adjust target altitude by a delta in metres.

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
    low: Annotated[float, "altitude delta metres"] = -152.4
    high: Annotated[float, "altitude delta metres"] = 152.4

    def set(self, idx: int, value: float) -> None:
        self.command_altitude_ft(idx, max(0.0, bs.traf.alt[idx] + value) * _M_TO_FT)

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class ApAltDeltaM(_AltitudeAction):
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
        "autopilot altitude offset metres; None = runtime altitude envelope",
    ] = None
    high: Annotated[
        float | None,
        "autopilot altitude offset metres; None = runtime altitude envelope",
    ] = None

    def set(self, idx: int, value: float) -> None:
        target = bs.traf.alt[idx] + value
        self.command_altitude_ft(idx, self.clip_altitude_m(idx, target) * _M_TO_FT)

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(
            lambda: self.altitude_delta_bounds_m(idx)
        )


@dataclass(frozen=True)
class SpdDeltaKts(ActionField):
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

    def set(self, idx: int, value: float) -> None:
        target = bs.traf.cas[idx] * _MS_TO_KTS + value
        lo = bs.traf.perf.vmin[idx] * _MS_TO_KTS
        hi = bs.traf.perf.vmax[idx] * _MS_TO_KTS
        bs.stack.stack(f"SPD {bs.traf.id[idx]} {min(max(target, lo), hi):{_FMT}}")

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class ApSpdDeltaKts(ActionField):
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
    command_floor_kts: Annotated[
        float | None,
        "minimum CAS knots command sent through BlueSky SPD",
    ] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.command_floor_kts is not None and self.command_floor_kts < 0.0:
            raise ValueError(
                f"ApSpdDeltaKts command_floor_kts must be >= 0.0 or None, "
                f"got {self.command_floor_kts}"
            )

    def set(self, idx: int, value: float) -> None:
        target = bs.traf.cas[idx] * _MS_TO_KTS + value
        lo, hi = self._target_bounds_kts(idx)
        bs.stack.stack(f"SPD {bs.traf.id[idx]} {min(max(target, lo), hi):{_FMT}}")

    def bounds(self, idx: int) -> tuple[float, float]:
        current = bs.traf.cas[idx] * _MS_TO_KTS
        lo, hi = self._target_bounds_kts(idx)
        return self._dynamic_or_configured_bounds(
            lambda: _reachable_delta(lo, hi, current)
        )

    def _target_bounds_kts(self, idx: int) -> tuple[float, float]:
        lo = bs.traf.perf.vmin[idx] * _MS_TO_KTS
        hi = bs.traf.perf.vmax[idx] * _MS_TO_KTS
        if self.command_floor_kts is None:
            return lo, hi
        floor = float(self.command_floor_kts)
        if floor > hi:
            raise ValueError(
                "ApSpdDeltaKts command_floor_kts must not exceed the resolved "
                f"upper speed bound ({hi:.3f} kt), got {floor:.3f} kt"
            )
        return max(lo, floor), hi


@dataclass(frozen=True)
class SpdDeltaMs(ActionField):
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
    low: Annotated[float, "CAS delta m/s"] = -5.144444
    high: Annotated[float, "CAS delta m/s"] = 5.144444

    def set(self, idx: int, value: float) -> None:
        target = (bs.traf.cas[idx] + value) * _MS_TO_KTS
        lo = bs.traf.perf.vmin[idx] * _MS_TO_KTS
        hi = bs.traf.perf.vmax[idx] * _MS_TO_KTS
        bs.stack.stack(f"SPD {bs.traf.id[idx]} {min(max(target, lo), hi):{_FMT}}")

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


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
class ActiveRouteWaypointAltDeltaFt(_AltitudeAction):
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
    min_altitude_ft: Annotated[
        float, "lowest commandable altitude, ft; match the task's selalt clamp"
    ] = 0.0

    def _nominal_ft(self, idx: int) -> float:
        wp = _active_route_waypoint(idx)
        if wp is not None and wp[2] is not None:
            return wp[2] * _M_TO_FT
        return bs.traf.alt[idx] * _M_TO_FT

    def set(self, idx: int, value: float) -> None:
        target_ft = self._nominal_ft(idx) + value
        self.command_altitude_ft(idx, self.clip_altitude_ft(idx, target_ft))

    def bounds(self, idx: int) -> tuple[float, float]:
        def resolve() -> tuple[float, float]:
            nominal_ft = self._nominal_ft(idx)
            return _reachable_delta(
                self.altitude_floor_m() * _M_TO_FT,
                self.altitude_ceiling_ft(idx),
                nominal_ft,
            )

        return self._dynamic_or_configured_bounds(resolve)


@dataclass(frozen=True)
class ActiveRouteWaypointSpdDeltaKts(ActionField):
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

    def _nominal_kts(self, idx: int) -> float:
        wp = _active_route_waypoint(idx)
        if wp is not None and wp[3] is not None:
            return wp[3] * _MS_TO_KTS
        return bs.traf.cas[idx] * _MS_TO_KTS

    def set(self, idx: int, value: float) -> None:
        target = self._nominal_kts(idx) + value
        lo = bs.traf.perf.vmin[idx] * _MS_TO_KTS
        hi = bs.traf.perf.vmax[idx] * _MS_TO_KTS
        bs.stack.stack(f"SPD {bs.traf.id[idx]} {min(max(target, lo), hi):{_FMT}}")

    def bounds(self, idx: int) -> tuple[float, float]:
        def resolve() -> tuple[float, float]:
            nominal = self._nominal_kts(idx)
            lo = bs.traf.perf.vmin[idx] * _MS_TO_KTS
            hi = bs.traf.perf.vmax[idx] * _MS_TO_KTS
            return _reachable_delta(lo, hi, nominal)

        return self._dynamic_or_configured_bounds(resolve)


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
class ActiveRouteWaypointSpdDeltaCrossover(ActionField):
    """Regime-aware speed command relative to the waypoint's speed constraint.

    Like :class:`ActiveRouteWaypointSpdDeltaKts`, but honours the CAS/Mach
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

    def _nominal_kts(self, idx: int) -> float:
        wp = _active_route_waypoint(idx)
        if wp is not None and wp[3] is not None:
            return wp[3] * _MS_TO_KTS
        return bs.traf.cas[idx] * _MS_TO_KTS

    def set(self, idx: int, value: float) -> None:
        _issue_crossover_speed(idx, self._nominal_kts(idx) + value)

    def bounds(self, idx: int) -> tuple[float, float]:
        def resolve() -> tuple[float, float]:
            nominal = self._nominal_kts(idx)
            lo = bs.traf.perf.vmin[idx] * _MS_TO_KTS
            hi = _cas_ceiling_ms(idx) * _MS_TO_KTS
            return _reachable_delta(lo, hi, nominal)

        return self._dynamic_or_configured_bounds(resolve)


@dataclass(frozen=True)
class ApSpdDeltaCrossover(ActionField):
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

    def set(self, idx: int, value: float) -> None:
        _issue_crossover_speed(idx, bs.traf.cas[idx] * _MS_TO_KTS + value)

    def bounds(self, idx: int) -> tuple[float, float]:
        def resolve() -> tuple[float, float]:
            current = bs.traf.cas[idx] * _MS_TO_KTS
            lo = bs.traf.perf.vmin[idx] * _MS_TO_KTS
            hi = _cas_ceiling_ms(idx) * _MS_TO_KTS
            return _reachable_delta(lo, hi, current)

        return self._dynamic_or_configured_bounds(resolve)


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
        record_comm_message(str(bs.traf.id[idx]), self.channel, float(value))

    def bounds(self, idx: int) -> tuple[float, float]:
        del idx
        return float(self.low), float(self.high)
