"""The state palette - part of the HUD contract (see :mod:`.hud`): what color
says what, the same in every driver.

An aircraft's state - and every mark of it: its chevron, protection zone,
trail, label, TSAS row - is drawn in its state's color: ``conflict`` orange
(predicted), ``los`` red (separation lost), ``violation`` purple (a task
constraint broken), ``normal`` neutral; ``selected`` marks the tracked
aircraft and ``paused`` the stopped clock. A state has one hue and a shade for
each :class:`Backdrop` - dark on a light background, light on a dark one, each
legible on its own - so pygame's sky blue and Panda3D's navy say the same
thing. A driver declares its backdrop and takes its colors from here; it
defines none of its own.

A design's own colors - for regions, routes, waypoints - never take an
alert's hue: :func:`overlay_rgb` turns one within :data:`RESERVED_DEG` of an
alert hue to the nearest hue outside, at the same saturation and brightness
(``"red"`` is drawn rose, ``"orange"`` gold, ``"purple"`` orchid), so a region
or an aircraft inside one is never taken for an alert. Grays, and colors too
pale to read as a hue, are left as they are.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "ALERTS",
    "BACKDROP_RGB",
    "RESERVED_DEG",
    "STATES",
    "Backdrop",
    "StateColor",
    "overlay_rgb",
    "reserved_hues",
    "state_rgb",
    "state_rgbf",
]

RGB = tuple[int, int, int]


class Backdrop(StrEnum):
    """What a driver draws its scene on."""

    LIGHT = "light"  # pygame: sky blue
    DARK = "dark"  # Panda3D: navy


#: Each backdrop's color - what its shades are made legible against.
BACKDROP_RGB: dict[Backdrop, RGB] = {
    Backdrop.LIGHT: (135, 206, 235),
    Backdrop.DARK: (18, 26, 36),
}


@dataclass(frozen=True)
class StateColor:
    """One state's color: its shade on a light backdrop and on a dark one."""

    light: RGB
    dark: RGB

    def on(self, backdrop: Backdrop) -> RGB:
        return self.light if backdrop is Backdrop.LIGHT else self.dark


#: Every state's color (see the module).
STATES: dict[str, StateColor] = {
    "normal": StateColor(light=(0, 0, 0), dark=(179, 230, 255)),
    "conflict": StateColor(light=(170, 80, 0), dark=(255, 140, 0)),
    "los": StateColor(light=(140, 0, 20), dark=(255, 77, 102)),
    "violation": StateColor(light=(120, 50, 190), dark=(170, 110, 240)),
    "selected": StateColor(light=(255, 215, 30), dark=(255, 214, 31)),
    "paused": StateColor(light=(140, 90, 0), dark=(255, 200, 60)),
}

#: The states that are alerts, most urgent first.
ALERTS = ("los", "conflict", "violation")


def state_rgb(state: str, backdrop: Backdrop) -> RGB:
    """``state``'s color on ``backdrop``, 0-255."""
    return STATES[state].on(backdrop)


def state_rgbf(state: str, backdrop: Backdrop) -> tuple[float, float, float]:
    """``state``'s color on ``backdrop``, 0-1."""
    r, g, b = state_rgb(state, backdrop)
    return r / 255.0, g / 255.0, b / 255.0


#: How close to an alert's hue (deg) a design color may not come.
RESERVED_DEG = 25.0
# Below this saturation a color reads as a gray, not a hue: left as it is.
_GRAY_SATURATION = 0.25


def _hue_deg(rgb: RGB) -> float:
    return colorsys.rgb_to_hsv(*(c / 255.0 for c in rgb))[0] * 360.0


def reserved_hues() -> list[tuple[float, float]]:
    """The hue bands (deg, ``(start, end)`` going up, wrapping at 360) kept
    for alerts: each alert's hue, +/- :data:`RESERVED_DEG`, overlapping ones
    joined."""
    centers = sorted({round(_hue_deg(STATES[s].dark), 1) for s in ALERTS})
    bands = [((c - RESERVED_DEG) % 360.0, (c + RESERVED_DEG) % 360.0) for c in centers]
    # Join bands that overlap: walk the circle from each band's start.
    joined: list[list[float]] = []
    for start, end in sorted(bands, key=lambda b: b[0]):
        if joined and _within(start, joined[-1][0], joined[-1][1]):
            if not _within(end, joined[-1][0], joined[-1][1]):
                joined[-1][1] = end
        else:
            joined.append([start, end])
    if len(joined) > 1 and _within(joined[0][0], joined[-1][0], joined[-1][1]):
        joined[0][0] = joined.pop()[0]
    return [(a, b) for a, b in joined]


def _within(hue: float, start: float, end: float) -> bool:
    """Whether ``hue`` is in the band from ``start`` up to ``end`` (wrapping)."""
    return (hue - start) % 360.0 <= (end - start) % 360.0


def overlay_rgb(rgb: RGB) -> RGB:
    """A design color as it is drawn: outside every alert's hue band (see the
    module) - turned to the nearer edge of the band it is in, at the same
    saturation and brightness; as it is when it is in none, or a gray."""
    h, s, v = colorsys.rgb_to_hsv(*(c / 255.0 for c in rgb))
    if s < _GRAY_SATURATION:
        return tuple(int(c) for c in rgb)
    hue = h * 360.0
    for start, end in reserved_hues():
        if _within(hue, start, end):
            down = (hue - start) % 360.0
            up = (end - hue) % 360.0
            hue = (start - 1.0) % 360.0 if down <= up else (end + 1.0) % 360.0
            r, g, b = colorsys.hsv_to_rgb(hue / 360.0, s, v)
            return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
    return tuple(int(c) for c in rgb)
