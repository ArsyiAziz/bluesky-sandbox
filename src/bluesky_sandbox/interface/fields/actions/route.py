"""Actions relative to the active route waypoint: a deviation from the guidance
LNAV/VNAV would fly, so a zero action flies the nominal. Where there is no
active waypoint or constraint, the nominal is the aircraft's current state, so
zero still means "hold".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import bluesky as bs
from bluesky.tools.geo import kwikqdrdist

from .._common import _InFeet, _InKnots
from .._route import _active_route_waypoint
from ..base import ActionMeta, ActionMode, ControlAxis, Unit
from .crossover import _SpeedAboveCrossover
from ._targets import (
    _AltitudeAxis,
    _DeltaTarget,
    _FromRouteWaypoint,
    _HeadingTarget,
    _SpeedAxis,
)


@dataclass(frozen=True)
class ActiveRouteWaypointHdgDeltaDeg(_HeadingTarget):
    """Steer relative to the bearing toward the active route waypoint.

    Commands ``bearing_to_active_waypoint + value`` (degrees); ``value == 0``
    flies straight at the waypoint. Falls back to current track when there is
    no active waypoint.
    """

    meta = ActionMeta(
        "active_route_waypoint_hdg_delta_deg",
        Unit.DEG,
        control_axis=ControlAxis.HEADING,
        mode=ActionMode.DELTA,
    )
    low: Annotated[float, "heading delta degrees from waypoint bearing"] = -180.0
    high: Annotated[float, "heading delta degrees from waypoint bearing"] = 180.0

    def nominal(self, idx: int) -> float:
        wp = _active_route_waypoint(idx)
        if wp is None:
            nominal = float(bs.traf.trk[idx])
        else:
            qdr, _dist = kwikqdrdist(
                float(bs.traf.lat[idx]), float(bs.traf.lon[idx]), wp[0], wp[1]
            )
            nominal = float(qdr)
        return nominal

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
    waypoint's speed constraint - its gate, or where it has a target arrival
    time instead, the speed that meets it: 0 is then "on schedule". Falls back
    to current CAS when the waypoint has neither. The result is clamped to the aircraft's
    performance speed envelope.

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


@dataclass(frozen=True)
class ActiveRouteWaypointSpdDeltaCrossover(
    _InKnots, _FromRouteWaypoint, _SpeedAboveCrossover, _DeltaTarget
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
