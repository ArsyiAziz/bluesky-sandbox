"""Autopilot mode switches: LNAV, VNAV, or both, with an ON/OFF hold band."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

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
    """Command BlueSky LNAV with an ON/OFF hold band."""

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
    """Command BlueSky VNAV with an ON/OFF hold band."""

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
    """Command BlueSky LNAV and VNAV together with an ON/OFF hold band."""

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
