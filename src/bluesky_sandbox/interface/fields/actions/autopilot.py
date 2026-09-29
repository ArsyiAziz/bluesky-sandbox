"""The autopilot: its selected heading, speed and altitude, set relative to the
current value, and its LNAV / VNAV modes, turned on with 1 and off with 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

import bluesky as bs

from .._common import _InKnots
from ..base import (
    ActionField,
    ActionMeta,
    ActionMode,
    ControlAxis,
    SwitchActionMixin,
    Unit,
)
from ._targets import (
    _FMT,
    _CrossoverSpeedAxis,
    _DeltaTarget,
)
from .kinematics import AltDeltaFt, AltDeltaM, SpdDeltaKts


def _switch(enabled: bool) -> str:
    return "ON" if enabled else "OFF"


@dataclass(frozen=True)
class _AutopilotSwitch(SwitchActionMixin, ActionField):
    """An autopilot mode, set ON with 1 and OFF with 0.

    A command goes to BlueSky only when the mode changes: turning a mode off
    also resets the selections it hands back to (see ``capture_off_reference``),
    which repeated every step would undo the other actions' targets.
    """

    switch_names: ClassVar[tuple[str, ...]] = ()

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
        if command == self.current_switch_state(idx):
            return
        acid = bs.traf.id[idx]
        if not command:
            self.capture_off_reference(idx)
        state = _switch(command)
        for switch_name in self.switch_names:
            bs.stack.stack(f"{switch_name} {acid} {state}")


@dataclass(frozen=True)
class AutopilotLnav(_AutopilotSwitch):
    """Turn BlueSky LNAV on (1) or off (0)."""

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
class AutopilotVnav(_AutopilotSwitch):
    """Turn BlueSky VNAV on (1) or off (0)."""

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
class AutopilotLnavVnav(_AutopilotSwitch):
    """Turn BlueSky LNAV and VNAV on (1) or off (0) together."""

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
class ResumeOwnNav(SwitchActionMixin, ActionField):
    """Resume own navigation: 1 turns LNAV back on, so the aircraft flies its
    route again from its active waypoint; 0 does nothing.

    The clearance that ends a vector. Unlike :class:`AutopilotLnav`, a 0 never
    turns LNAV off - the policy says "resume" when it means to and nothing
    otherwise - so it can sit at 0 every step. Turned on, it suppresses the
    heading actions that step: a vector and a resume given together resume. An
    aircraft already on LNAV, or with no active waypoint, is sent nothing.
    """

    meta = ActionMeta(
        "resume_own_nav",
        Unit.SWITCH,
        control_axis=ControlAxis.AUTOPILOT,
        mode=ActionMode.SWITCH,
        suppresses_when_on=(ControlAxis.HEADING,),
    )

    def set(self, idx: int, value: float) -> None:
        if not self.switch_command(value) or self.current_switch_state(idx):
            return
        if int(getattr(bs.traf.ap.route[idx], "iactwp", -1)) < 0:
            return
        bs.stack.stack(f"LNAV {bs.traf.id[idx]} ON")

    def current_switch_state(self, idx: int) -> bool:
        return bool(bs.traf.swlnav[idx])


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
