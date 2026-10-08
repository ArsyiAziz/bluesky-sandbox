"""Display toggles: what a viewer can show or hide, declared once - one kind
of control in the HUD contract (see :mod:`.hud`).

Each :class:`DisplayToggle` names the driver attribute holding its value, the
key that steps it, its button title and the values it steps through. The
drivers derive everything else from the list - key bindings, buttons, the note
shown when one changes - so a new toggle is one entry in
:data:`DISPLAY_TOGGLES` plus the drawing that reads its attribute, in each
driver that can draw it. A driver leaves out the ones it cannot draw with
``UNSUPPORTED_CONTROLS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .hud import key_text

__all__ = ["DISPLAY_TOGGLES", "DisplayToggle"]


@dataclass(frozen=True)
class DisplayToggle:
    """One thing a viewer can show or hide.

    ``name`` - the driver attribute holding its value. ``key`` - the key that
    steps it, as Panda3D names keys (``"l"``, ``"shift-l"``). ``title`` - a
    button's text. ``label`` - what it is, in prose. ``values`` - what it steps
    through, the first the default; off is ``False`` or ``"off"``. ``button`` -
    offered as a button where a driver has a row of them."""

    name: str
    key: str
    title: str
    label: str
    values: tuple[Any, ...] = (False, True)
    button: bool = False

    def after(self, value: Any) -> Any:
        """The value after ``value``."""
        values = self.values
        index = values.index(value) if value in values else -1
        return values[(index + 1) % len(values)]

    @staticmethod
    def is_on(value: Any) -> bool:
        """Whether ``value`` shows anything."""
        return value not in (False, None, "off")

    def describe(self, value: Any) -> str:
        """``value`` in words: ``on``/``off``, or the value itself."""
        if isinstance(value, bool):
            return "on" if value else "off"
        return str(value)

    # ---- as a control (see .hud) -------------------------------------------

    #: The toolbar group it is drawn with.
    group = "display"

    @property
    def keys(self) -> tuple[str, ...]:
        return (self.key,)

    def run(self, driver: Any) -> None:
        driver.toggle(self.name)

    def button_text(self, driver: Any) -> str:
        """Its title, and - stepping through more than on and off - the
        value's initial: ``LBL·C``."""
        value = getattr(driver, self.name)
        if len(self.values) > 2 and self.is_on(value):
            return f"{self.title}\u00b7{str(value)[0].upper()}"
        return self.title

    def is_on_for(self, driver: Any) -> bool:
        return self.is_on(getattr(driver, self.name))

    def describe_for(self, driver: Any) -> str:
        return f"{self.label}: {self.describe(getattr(driver, self.name))}"

    @property
    def tooltip(self) -> str:
        return f"{self.label} ({key_text(self.key)})"


#: Every display toggle, in the order buttons and hints list them.
DISPLAY_TOGGLES: tuple[DisplayToggle, ...] = (
    DisplayToggle(
        "aircraft_labels",
        "l",
        "LBL",
        "aircraft labels",
        values=("full", "callsign", "off"),
        button=True,
    ),
    DisplayToggle("show_labels", "shift-l", "NAM", "region and waypoint names", values=(True, False), button=True),
    DisplayToggle("show_trails", "t", "TRL", "trails", button=True),
    DisplayToggle("show_all_routes", "o", "RTE", "all routes", button=True),
    DisplayToggle("show_velocity_obstacles", "v", "VO", "velocity obstacles", button=True),
)
