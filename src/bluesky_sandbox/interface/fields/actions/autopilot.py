"""Autopilot mode switches: LNAV, VNAV, or both, turned on with 1 and off with 0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import bluesky as bs

from ..base import (
    ActionField,
    ActionMeta,
    ActionMode,
    ControlAxis,
    SwitchActionMixin,
    Unit,
)


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
