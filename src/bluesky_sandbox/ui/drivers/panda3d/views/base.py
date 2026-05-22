"""Base class for composable panda3d view panels.

A view is a self-contained slice of the 3D viewer's output - the main
3D scene, a HUD overlay, a side-panel text block.  Mirrors the pygame
driver's view abstraction: the driver coordinates window / event /
camera lifecycle and fans render-primitive dispatch out to every view;
each view owns its own NodePaths / OnscreenText and refreshes itself
per frame.

Unlike pygame there is no draggable layout - the 3D viewport always
fills the window and HUD-style views position themselves in
``aspect2d`` (corners) without overlap.  Subclasses opt in to only the
hooks they need; every default is a no-op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bluesky_sandbox.ui.drivers.common import CursorHintName

if TYPE_CHECKING:
    from bluesky_sandbox.ui.display.overlays import Point, Polygon, Polyline
    from bluesky_sandbox.ui.drivers.panda3d.driver import Panda3DSimDriver


class Panda3DView:
    """A composable slice of the panda3d driver's output.

    Lifecycle hooks fire in this order:

    * :meth:`on_start` - once, after the Panda3D window is open.
    * :meth:`on_reset` - every ``env.reset()``.
    * :meth:`on_step`  - every sim step, after BlueSky has advanced.
    * :meth:`close`    - once, before the window is destroyed.

    Render primitives are fanned out from the driver via
    :meth:`draw_polygon` / :meth:`draw_point` / :meth:`draw_polyline`.
    Subclasses that don't paint world geometry (HUD-only views) leave
    these as no-ops.
    """

    def on_start(self, driver: Panda3DSimDriver) -> None:
        """One-time setup after the Panda3D window is open."""

    def on_reset(self, driver: Panda3DSimDriver, env) -> None:
        """Rebuild per-env state.  Called every ``env.reset()``."""

    def on_step(self, driver: Panda3DSimDriver) -> None:
        """Refresh per-frame state after the sim has advanced."""

    def close(self) -> None:
        """Release any resources before the window is destroyed."""

    # ------------------------------------------------------------------
    # Optional HUD interaction hooks. Coordinates are aspect2d units.
    # ------------------------------------------------------------------

    def on_mouse_down(
        self,
        driver: Panda3DSimDriver,
        pos: tuple[float, float],
    ) -> bool:
        """Return True to claim a left-button drag starting at *pos*."""
        return False

    def on_mouse_drag(
        self,
        driver: Panda3DSimDriver,
        pos: tuple[float, float],
    ) -> None:
        """Called while this view owns a left-button drag."""

    def on_mouse_up(
        self,
        driver: Panda3DSimDriver,
        pos: tuple[float, float],
    ) -> None:
        """Called when a claimed left-button drag ends."""

    def cursor_hint(
        self,
        driver: Panda3DSimDriver,
        pos: tuple[float, float],
    ) -> CursorHintName | None:
        """Return a semantic cursor hint for *pos*, or None for the default."""
        return None

    # ------------------------------------------------------------------
    # Render-primitive vocabulary - no-ops by default so HUD-style
    # subclasses don't have to override what they don't care about.
    # ------------------------------------------------------------------

    def draw_polygon(self, driver: Panda3DSimDriver, polygon: Polygon) -> None:
        """Render a closed lat/lon polygon (no-op by default)."""

    def draw_point(self, driver: Panda3DSimDriver, point: Point) -> None:
        """Render a single named lat/lon point (no-op by default)."""

    def draw_polyline(self, driver: Panda3DSimDriver, polyline: Polyline) -> None:
        """Render an open chain of lat/lon points (no-op by default)."""
