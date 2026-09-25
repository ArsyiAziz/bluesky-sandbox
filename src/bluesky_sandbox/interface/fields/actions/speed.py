"""Speed actions: a calibrated airspeed target, absolute or a delta from the
current CAS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from .._common import _InKnots, _InMetersPerSecond
from ..base import ActionMeta, ActionMode, ControlAxis, Unit
from ._targets import _AbsoluteTarget, _CrossoverSpeedAxis, _DeltaTarget, _SpeedAxis


@dataclass(frozen=True)
class SpdKts(_InKnots, _SpeedAxis, _AbsoluteTarget):
    """Set target calibrated airspeed in knots.

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
class SpdDeltaKts(_InKnots, _SpeedAxis, _DeltaTarget):
    """Adjust target calibrated airspeed by a delta in knots.

    ``low=None`` and ``high=None`` resolve the delta bounds at runtime as a
    symmetric span around the current CAS, reaching both the minimum and maximum
    operating speed. Pass ``low`` and ``high`` for a fixed range instead.
    """

    meta = ActionMeta(
        "spd_delta_kts",
        Unit.KTS,
        control_axis=ControlAxis.SPEED,
        mode=ActionMode.DELTA,
        dynamic_bounds=True,
    )
    low: Annotated[float | None, "CAS delta knots; None = runtime speed envelope"] = (
        None
    )
    high: Annotated[float | None, "CAS delta knots; None = runtime speed envelope"] = (
        None
    )


@dataclass(frozen=True)
class ApSpdDeltaKts(SpdDeltaKts):
    """Set autopilot selected calibrated airspeed relative to current CAS.

    The same action as :class:`SpdDeltaKts` - BlueSky's ``SPD`` sets the
    autopilot selection either way - under the name existing configs use.
    """

    meta = ActionMeta(
        "ap_spd_delta_kts",
        Unit.KTS,
        control_axis=ControlAxis.SPEED,
        mode=ActionMode.DELTA,
        dynamic_bounds=True,
    )


@dataclass(frozen=True)
class SpdDeltaMs(_InMetersPerSecond, _SpeedAxis, _DeltaTarget):
    """Adjust target calibrated airspeed by a delta in m/s.

    ``low=None`` and ``high=None`` resolve the delta bounds at runtime as a
    symmetric span around the current CAS, reaching both the minimum and maximum
    operating speed. Pass ``low`` and ``high`` for a fixed range instead.
    """

    meta = ActionMeta(
        "spd_delta_ms",
        Unit.M_PER_S,
        control_axis=ControlAxis.SPEED,
        mode=ActionMode.DELTA,
        dynamic_bounds=True,
    )
    low: Annotated[float | None, "CAS delta m/s; None = runtime speed envelope"] = None
    high: Annotated[float | None, "CAS delta m/s; None = runtime speed envelope"] = None


@dataclass(frozen=True)
class ApSpdDeltaCrossover(_InKnots, _CrossoverSpeedAxis, _DeltaTarget):
    """Autopilot speed relative to *current* CAS, regime-aware (CAS/Mach crossover).

    The autopilot counterpart of :class:`ActiveRouteWaypointSpdDeltaCrossover`:
    the nominal is the aircraft's current CAS (not a waypoint constraint), so the
    action nudges speed from where it is. The CAS target is capped at the
    altitude's Mach limit and issued as **Mach** above the crossover altitude /
    **CAS** below - so it never commands a Mach-exceeding CAS at cruise.
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
