"""Base class for package-owned GUI drivers.

Pygame and Panda3D extend this class. QtGL does not: its window is
BlueSky's own QtGL client, so pause, dtmult, and HUD text route through
BlueSky stack commands instead of fields on the driver.
"""

from __future__ import annotations

from .common import AircraftReadoutMixin, TimeControlMixin
from .human_driver import HumanSimDriver


class SandboxGUIDriver(TimeControlMixin, AircraftReadoutMixin, HumanSimDriver):
    """Human driver that owns its own window, HUD, and input loop."""

    def __init__(self, realtime: bool = True) -> None:
        super().__init__(realtime=realtime)
        self._init_time_controls()
