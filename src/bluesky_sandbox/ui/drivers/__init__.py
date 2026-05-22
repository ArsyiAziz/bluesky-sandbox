from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from .human_driver import HumanSimDriver
from .sandbox_gui_driver import SandboxGUIDriver
from .sim_driver import SimDriver

# NOTE: ``None`` stays INSIDE the Literal. ``base_environment`` derives its
# ``metadata["render_modes"]`` from ``get_args(RenderMode)``, which yields the
# four members here but ``(Literal[...], NoneType)`` for the "equivalent"
# ``Literal[...] | None`` - and every render_mode check would then fail.
# ruff's PYI061 suggests that rewrite; it is not safe here.
RenderMode = Literal["qtgl", "pygame", "panda3d", None]  # noqa: PYI061


def get_driver_class(render_mode: RenderMode) -> type[SimDriver]:
    """Return the driver class for *render_mode*, importing GUI stacks lazily."""
    if render_mode is None:
        return SimDriver
    if render_mode == "qtgl":
        from .qtgl import QtGLSimDriver

        return QtGLSimDriver
    if render_mode == "pygame":
        from .pygame import PygameSimDriver

        return PygameSimDriver
    if render_mode == "panda3d":
        from .panda3d import Panda3DSimDriver

        return Panda3DSimDriver
    raise KeyError(render_mode)


class _DriverRegistry(Mapping):
    """Compatibility mapping for older code that indexes ``DRIVERS``."""

    _keys = (None, "qtgl", "pygame", "panda3d")

    def __getitem__(self, key):
        return get_driver_class(key)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)


DRIVERS = _DriverRegistry()


def __getattr__(name: str):
    if name == "QtGLSimDriver":
        from .qtgl import QtGLSimDriver

        return QtGLSimDriver
    if name in {
        "HSplit",
        "HorizontalView",
        "PygameSimDriver",
        "PygameView",
        "TSASView",
        "VSplit",
        "VerticalView",
    }:
        from . import pygame as _pygame

        return getattr(_pygame, name)
    if name == "Panda3DSimDriver":
        from .panda3d import Panda3DSimDriver

        return Panda3DSimDriver
    raise AttributeError(name)

__all__ = [
    "DRIVERS",
    "HSplit",
    "HorizontalView",
    "HumanSimDriver",
    "Panda3DSimDriver",
    "PygameSimDriver",
    "PygameView",
    "QtGLSimDriver",
    "RenderMode",
    "SandboxGUIDriver",
    "SimDriver",
    "TSASView",
    "VSplit",
    "VerticalView",
    "get_driver_class",
]
