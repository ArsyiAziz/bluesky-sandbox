"""Base class + shared helpers for pygame view panels.

Each :class:`PygameView` occupies a horizontal slice of the window
assigned by the driver's layout, and is responsible for everything
inside that slice - drawing, hover detection, render-primitive
ingestion (polygons / points / polylines), and per-aircraft highlight.

Cross-view concerns (which view produced the hover, where the aircraft
is on every panel) are coordinated by :class:`PygameSimDriver`, which
asks each view's :meth:`hover` and :meth:`aircraft_position` methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pygame

from bluesky_sandbox.sim.queryables import QueryRegion
from bluesky_sandbox.ui.drivers.common import CursorHintName
from bluesky_sandbox.ui.drivers.pygame import colors as C

if TYPE_CHECKING:
    from bluesky_sandbox.ui.display.overlays import Point, Polygon, Polyline
    from bluesky_sandbox.ui.drivers.pygame.driver import PygameSimDriver


class PygameView(ABC):
    """Base class for pygame view panels.

    Subclasses override the abstract :meth:`render` and may opt in to:

    * :meth:`on_reset` - recompute static state when the env resets.
    * :meth:`add_polygon` / :meth:`add_point` / :meth:`add_polyline` -
      ingest render primitives the driver dispatches via ``draw()``.
    * :meth:`hover` - report what's under the cursor when the cursor is
      inside this view's :attr:`rect`.
    * :meth:`highlight_aircraft` - draw a highlight ring on aircraft
      *idx* in this view (called for the cross-view hover).
    * :meth:`aircraft_position` - report where aircraft *idx* sits on
      this view's panel, for the cross-view link line.

    Attributes
    ----------
    default_height_fraction:
        Default share of the window's height when the driver normalises
        view heights for layout.  Override on the class.
    rect:
        The pygame rect the driver assigns each frame (can change on
        resize).  Views must clip their drawing to this rect.
    """

    default_height_fraction: float = 0.5

    def __init__(self) -> None:
        self.rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_reset(self, driver: PygameSimDriver, env) -> None:
        """Recompute static state when the env resets.  Default: no-op."""

    @abstractmethod
    def render(self, canvas: pygame.Surface, driver: PygameSimDriver) -> None:
        """Draw this view's content onto *canvas* within :attr:`rect`."""

    # ------------------------------------------------------------------
    # Render-primitive ingestion (driver fans out via ``draw_*``).
    # Default: no-op so views opt in only to the primitives they care about.
    # ------------------------------------------------------------------

    def add_polygon(self, driver: PygameSimDriver, polygon: Polygon) -> None:
        """Receive a Polygon primitive (no-op by default)."""

    def add_point(self, driver: PygameSimDriver, point: Point) -> None:
        """Receive a Point primitive (no-op by default)."""

    def add_polyline(self, driver: PygameSimDriver, polyline: Polyline) -> None:
        """Receive a Polyline primitive (no-op by default)."""

    # ------------------------------------------------------------------
    # Hover & cross-view highlight
    # ------------------------------------------------------------------

    def hover(
        self,
        mouse_pos: tuple[int, int],
        driver: PygameSimDriver,
    ) -> dict | None:
        """Return a hover descriptor when the cursor is over a hit-target.

        Conventional shapes:

        * ``{'kind': 'label', 'info': dict}`` - a region/point label.
        * ``{'kind': 'aircraft', 'idx': int}`` - an aircraft.
        * ``None`` - no hit in this view.

        Default: ``None`` (no hover targets).
        """
        return None

    def highlight_aircraft(
        self,
        canvas: pygame.Surface,
        driver: PygameSimDriver,
        idx: int,
    ) -> None:
        """Draw a highlight ring on aircraft *idx* in this view (no-op by default)."""

    def aircraft_position(
        self,
        driver: PygameSimDriver,
        idx: int,
    ) -> tuple[float, float] | None:
        """Return aircraft *idx*'s on-screen position in this view, if any.

        Used by the driver to draw a link line between aircraft markers
        across views.  Default: ``None`` (this view doesn't show aircraft).
        """
        return None

    def _query_color_for_aircraft(
        self,
        driver: PygameSimDriver,
        lat_deg: float,
        lon_deg: float,
        alt_ft: float,
    ) -> tuple[int, int, int] | None:
        """Return the colour of the first :class:`QueryRegion` containing
        the aircraft, or ``None`` if it lies outside every region."""
        env = getattr(driver, "_env", None)
        if env is None:
            return None
        for qable in env.episode_queryables.values():
            if isinstance(qable, QueryRegion) and qable.bounds.contains(
                lat_deg,
                lon_deg,
                alt_ft,
            ):
                return C.named(qable.color)
        return None

    # ------------------------------------------------------------------
    # View-private drags (axis rotation, custom widgets, ...)
    # ------------------------------------------------------------------
    # The driver dispatches these *before* its own header/divider drags,
    # so a view can claim a click on its own widget.  Default: views
    # claim nothing.

    def hit_test_drag(self, pos, driver: PygameSimDriver):
        """Return a non-``None`` handle if *pos* starts a view-private drag.

        The handle is opaque to the driver - it stashes it and feeds it
        back to :meth:`on_drag_motion` / :meth:`on_drag_end` so the
        view can interpret what kind of drag it kicked off.
        """
        return

    def on_drag_motion(self, handle, pos, driver: PygameSimDriver) -> None:
        """Called on each MOUSEMOTION while a view-private drag is active."""

    def on_drag_end(self, handle, pos, driver: PygameSimDriver) -> None:
        """Called on MOUSEBUTTONUP ending a view-private drag."""

    def cursor_hint(self, pos, driver: PygameSimDriver) -> CursorHintName | None:
        """Return a semantic cursor hint for a view-owned interaction."""
        return None

    # ------------------------------------------------------------------
    # Optional viewport zoom/pan
    # ------------------------------------------------------------------

    def zoom_view_at(
        self,
        pos: tuple[int, int],
        factor: float,
        driver: PygameSimDriver,
    ) -> bool:
        """Zoom around *pos* when the view supports viewport zooming."""
        return False

    def pan_view_by(
        self,
        delta: tuple[int, int],
        driver: PygameSimDriver,
    ) -> bool:
        """Pan the view by a pixel delta when supported."""
        return False

    def reset_viewport(self, driver: PygameSimDriver) -> bool:
        """Reset optional viewport zoom/pan state."""
        return False
