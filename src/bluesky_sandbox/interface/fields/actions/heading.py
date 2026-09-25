"""Heading actions: an absolute heading, or a turn from the current heading or
track.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import bluesky as bs

from ..base import ActionField, ActionMeta, ActionMode, ControlAxis, Unit
from ._targets import _FMT


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
class ApHdgDeltaDeg(ActionField):
    """Set autopilot selected heading relative to current track."""

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
