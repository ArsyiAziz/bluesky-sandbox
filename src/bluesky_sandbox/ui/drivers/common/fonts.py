"""Shared font preferences for human GUI drivers."""

from __future__ import annotations

from pathlib import Path

UI_FONT_NAMES = (
    "SF Mono",
    "Menlo",
    "Monaco",
    "Consolas",
    "DejaVu Sans Mono",
    "Courier New",
    "SF Pro Text",
    "Avenir Next",
    "Avenir",
    "Helvetica Neue",
    "Segoe UI",
    "Arial",
)

UI_FONT_PATHS = (
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Avenir.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
)

PANDA3D_FONT_PATHS = (
    "/System/Library/Fonts/Monaco.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
)


def preferred_ui_font_path() -> str | None:
    """Return the first known UI font path available on this machine."""
    for path in UI_FONT_PATHS:
        if Path(path).exists():
            return path
    return None


def preferred_panda3d_font_path() -> str | None:
    """Return a Panda3D-friendly UI font path.

    Panda3D's font loader can be brittle with some macOS collection/variable
    fonts. Keep this list conservative so labels stay visible.
    """
    for path in PANDA3D_FONT_PATHS:
        if Path(path).exists():
            return path
    return None
