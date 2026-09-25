"""Actions on the aircraft's own state: a heading, speed or altitude target,
absolute or a delta from the current value, commanded directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import bluesky as bs

from .._common import _InFeet, _InKnots, _InMeters, _InMetersPerSecond
from ..base import ActionField, ActionMeta, ActionMode, ControlAxis, Unit
from ._targets import (
    _FMT,
    _AbsoluteTarget,
    _AltitudeAxis,
    _DeltaTarget,
    _SpeedAxis,
)


@dataclass(frozen=True)
class HdgDeg(ActionField):
    """Set target heading in degrees."""

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
class HdgDeltaDeg(ActionField):
    """Adjust heading by a delta in degrees."""

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
class AltFt(_InFeet, _AltitudeAxis, _AbsoluteTarget):
    """Set target altitude in feet.

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
class AltDeltaFt(_InFeet, _AltitudeAxis, _DeltaTarget):
    """Adjust target altitude by a delta in feet.

    ``low=None`` and ``high=None`` resolve the delta bounds at runtime as a
    symmetric span around the current altitude, reaching both 0 and the aircraft's
    altitude ceiling. Pass ``low`` and ``high`` for a fixed range instead.
    """

    meta = ActionMeta(
        "alt_delta_ft",
        Unit.FT,
        control_axis=ControlAxis.ALTITUDE,
        mode=ActionMode.DELTA,
        dynamic_bounds=True,
    )
    low: Annotated[
        float | None, "altitude delta feet; None = runtime altitude envelope"
    ] = None
    high: Annotated[
        float | None, "altitude delta feet; None = runtime altitude envelope"
    ] = None


@dataclass(frozen=True)
class AltDeltaM(_InMeters, _AltitudeAxis, _DeltaTarget):
    """Adjust target altitude by a delta in meters.

    ``low=None`` and ``high=None`` resolve the delta bounds at runtime as a
    symmetric span around the current altitude, reaching both 0 and the aircraft's
    altitude ceiling. Pass ``low`` and ``high`` for a fixed range instead.
    """

    meta = ActionMeta(
        "alt_delta_m",
        Unit.M,
        control_axis=ControlAxis.ALTITUDE,
        mode=ActionMode.DELTA,
        dynamic_bounds=True,
    )
    low: Annotated[
        float | None, "altitude delta meters; None = runtime altitude envelope"
    ] = None
    high: Annotated[
        float | None, "altitude delta meters; None = runtime altitude envelope"
    ] = None
