"""Base class for package-owned GUI drivers.

Pygame and Panda3D extend this class. QtGL does not: its window is
BlueSky's own QtGL client, so pause, dtmult, and HUD text route through
BlueSky stack commands instead of fields on the driver.
"""

from __future__ import annotations

from .common import DISPLAY_TOGGLES, TIME_CONTROLS, AircraftReadoutMixin, TimeControlMixin
from .common.hud import Button, HudContent
from .human_driver import HumanSimDriver


class SandboxGUIDriver(TimeControlMixin, AircraftReadoutMixin, HumanSimDriver):
    """Human driver that owns its own window, HUD, and input loop - and draws
    the HUD the contract in ``common.hud`` describes."""

    CONTROLS = (*TIME_CONTROLS, *DISPLAY_TOGGLES)

    def __init__(self, realtime: bool = True) -> None:
        super().__init__(realtime=realtime)
        self._init_time_controls()

    def hud_content(self) -> HudContent:
        """What the HUD shows now: the toolbar's buttons, the status, the
        tracked aircraft's information and the note. A recorded frame
        (``offscreen``) has no toolbar: nothing there can press it."""
        tracked = self.tracked_acid()
        offscreen = bool(getattr(self, "offscreen", False))
        return HudContent(
            toolbar=() if offscreen else tuple(
                Button(c.name, c.button_text(self), c.is_on_for(self), c.tooltip, c.group)
                for c in self.controls
                if c.button
            ),
            status=tuple(self.format_status_line().split("\n")),
            info=tuple(self.format_aircraft_info_lines(tracked)) if tracked is not None else (),
            note=self.note,
            paused=bool(self._paused),
        )
