"""The HUD contract: what every viewer shows around its scene, and how a person
operates it - the same in each driver that owns its window (pygame, Panda3D).

**Controls.** Everything a person can operate is a control, declared once:

- a :class:`~.display_toggles.DisplayToggle` - something shown or hidden,
  stepped through its values (labels, trails, routes);
- an :class:`Action` - something done, a driver method run (pause, realtime,
  slower, faster, delete all aircraft, reset the episode).

Each has its keys (as Panda3D names them: ``"l"``, ``"shift-l"``), a button's
text and a tooltip. A driver binds the keys and draws the buttons from
:attr:`~..human_driver.HumanSimDriver.controls`, routes both to
:meth:`~..human_driver.HumanSimDriver.activate`, and leaves out what it cannot
do (``UNSUPPORTED_CONTROLS``). A new control is one entry in a list.

**Content.** What the HUD shows in a frame is a :class:`HudContent`, built
once by the driver base (:meth:`~..sandbox_gui_driver.SandboxGUIDriver.hud_content`):
the toolbar's buttons, the status lines, the tracked aircraft's information
and a short note on what just changed. A driver draws it as it is - it formats
none of it - so the text is the same everywhere.

**Layout.** Each part has its corner (:data:`HUD_LAYOUT`): the toolbar top
right, the aircraft information top left, the status bottom right, the note
top center. A driver places them in its window, keeps its aircraft labels
clear of them, and keeps a click on the toolbar from reaching the scene. A
recorded frame (an offscreen driver) has no toolbar.

**Colors.** What color says what - conflict orange, loss of separation red,
violation purple, the tracked aircraft yellow, the stopped clock amber - is
the state palette (:mod:`.palette`): one hue a state, a shade for the
driver's backdrop.

**A driver implements** :class:`HudDriver`: where its HUD is on the screen,
and which button is at a point.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

__all__ = [
    "HUD_LAYOUT",
    "TIME_CONTROLS",
    "Action",
    "Anchor",
    "Button",
    "HudContent",
    "HudDriver",
    "Rect",
    "key_text",
]

#: A rectangle on the screen in pixels, y down: ``(left, top, right, bottom)``.
Rect = tuple[float, float, float, float]


def key_text(key: str) -> str:
    """A key as a person reads it: ``"shift-l"`` is ``Shift+L``."""
    *mods, name = key.split("-") if key != "-" else ["-"]
    return "+".join([*(m.capitalize() for m in mods), name.upper() if len(name) == 1 else name.capitalize()])


@dataclass(frozen=True)
class Action:
    """Something a person does: ``method`` - a driver method taking no
    arguments - run by its ``keys`` or its button.

    ``title`` - the button's text, or ``text`` - a function of the driver
    giving it, for a button that shows a state (``REAL 2x``). ``on`` - a
    function of the driver: whether the button shows as on. ``note`` - a
    function of the driver: what it did, said once it has (default: its
    label). ``button`` - whether it is one."""

    name: str
    keys: tuple[str, ...]
    title: str
    label: str
    method: str
    button: bool = True
    text: Callable[[Any], str] | None = None
    on: Callable[[Any], bool] | None = None
    note: Callable[[Any], str] | None = None
    #: The toolbar group it is drawn with.
    group: str = "time"

    def run(self, driver: Any) -> None:
        getattr(driver, self.method)()

    def button_text(self, driver: Any) -> str:
        return self.text(driver) if self.text is not None else self.title

    def is_on_for(self, driver: Any) -> bool | None:
        """Whether its button shows as on; None for an action with no state
        (``slower``), drawn as a plain button."""
        return bool(self.on(driver)) if self.on is not None else None

    def describe_for(self, driver: Any) -> str:
        return self.note(driver) if self.note is not None else self.label

    @property
    def tooltip(self) -> str:
        keys = " / ".join(key_text(k) for k in self.keys)
        return f"{self.label} ({keys})" if keys else self.label


def _speed_text(driver: Any) -> str:
    if not driver.realtime:
        return "FAST"
    mult = getattr(driver, "_desired_dtmult", 1.0)
    return "REAL" if mult == 1.0 else f"REAL {mult:g}x"


#: The time controls, for drivers that run the clock (``SandboxGUIDriver``).
TIME_CONTROLS: tuple[Action, ...] = (
    Action(
        "pause", ("space", "p"), "PAUSE", "pause / run", "toggle_pause",
        on=lambda d: bool(getattr(d, "_paused", False)),
        note=lambda d: "paused" if getattr(d, "_paused", False) else "running",
    ),
    Action(
        "realtime", ("r",), "REAL", "realtime / fast-time", "toggle_realtime",
        text=_speed_text, on=lambda d: bool(d.realtime), note=_speed_text,
    ),
    Action("slower", ("-",), "-", "half the speed (realtime)", "slow_down", note=_speed_text),
    Action("faster", ("+", "=", "shift-="), "+", "double the speed (realtime)", "speed_up", note=_speed_text),
    Action("delete_aircraft", ("backspace",), "DEL", "delete every aircraft", "delete_all_aircraft", button=False),
    Action("reset_episode", ("shift-backspace",), "RESET", "end the episode", "request_episode_reset", button=False),
)


@dataclass(frozen=True)
class Button:
    """One toolbar button as it shows now: its control's name, text, whether
    it is on (None: an action with no state, a plain button), its tooltip, and
    the group it is drawn with (a gap between groups)."""

    name: str
    text: str
    on: bool | None
    tooltip: str
    group: str = ""


@dataclass(frozen=True)
class HudContent:
    """What the HUD shows in one frame (see the module)."""

    toolbar: tuple[Button, ...] = ()
    status: tuple[str, ...] = ()
    info: tuple[str, ...] = ()
    note: str | None = None
    #: The clock is stopped: the status is drawn in the palette's ``paused``.
    paused: bool = False


class Anchor(StrEnum):
    """Where a part of the HUD sits in the window."""

    TOP_LEFT = "top-left"
    TOP_CENTER = "top-center"
    TOP_RIGHT = "top-right"
    BOTTOM_RIGHT = "bottom-right"


#: Each part of :class:`HudContent`, and its corner.
HUD_LAYOUT: dict[str, Anchor] = {
    "toolbar": Anchor.TOP_RIGHT,
    "info": Anchor.TOP_LEFT,
    "status": Anchor.BOTTOM_RIGHT,
    "note": Anchor.TOP_CENTER,
}


class HudDriver(Protocol):
    """What a driver drawing the HUD provides, beyond drawing it."""

    def hud_rects_px(self) -> list[Rect]:
        """Where the HUD is on the screen - what aircraft labels keep clear of."""
        ...

    def toolbar_button_at(self, x: float, y: float) -> str | None:
        """The control whose button is at pixel ``(x, y)``, if any - a click
        there runs it and goes no further."""
        ...
