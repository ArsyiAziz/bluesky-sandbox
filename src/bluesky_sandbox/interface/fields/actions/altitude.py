"""Altitude actions: an altitude target, absolute or a delta from the current
altitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from .._common import _InFeet, _InMeters
from ..base import ActionMeta, ActionMode, ControlAxis, Unit
from ._targets import _AbsoluteTarget, _AltitudeAxis, _DeltaTarget


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
class ApAltDeltaFt(AltDeltaFt):
    """Set autopilot selected altitude relative to current altitude.

    The same action as :class:`AltDeltaFt` - BlueSky's ``ALT`` sets the
    autopilot selection either way - under the name existing configs use.
    """

    meta = ActionMeta(
        "ap_alt_delta_ft",
        Unit.FT,
        control_axis=ControlAxis.ALTITUDE,
        mode=ActionMode.DELTA,
        dynamic_bounds=True,
    )


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


@dataclass(frozen=True)
class ApAltDeltaM(AltDeltaM):
    """Set autopilot selected altitude relative to current altitude.

    The same action as :class:`AltDeltaM` - BlueSky's ``ALT`` sets the
    autopilot selection either way - under the name existing configs use.
    """

    meta = ActionMeta(
        "ap_alt_delta_m",
        Unit.M,
        control_axis=ControlAxis.ALTITUDE,
        mode=ActionMode.DELTA,
        dynamic_bounds=True,
    )
