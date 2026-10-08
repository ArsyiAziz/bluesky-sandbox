"""Shared render-primitive dispatch helpers for sim drivers."""

from __future__ import annotations

import warnings
from collections.abc import Iterable

from bluesky_sandbox.ui.display.overlays import Point, Polygon, Polyline, Renderable


class PrimitiveDrawMixin:
    """Mixin for drivers that consume :mod:`bluesky_sandbox.ui.display.overlays` primitives."""

    _required_draws: tuple[str, ...] = ("draw_polygon", "draw_point", "draw_polyline")

    def _check_draws_implemented(self) -> None:
        """Warn if any required ``draw_*`` method is still the base no-op."""
        missing = [
            name for name in self._required_draws
            if getattr(type(self), name) is getattr(PrimitiveDrawMixin, name)
        ]
        if missing:
            warnings.warn(
                f"{type(self).__name__} does not override {missing}; "
                "primitives of those kinds will silently disappear in this view. "
                "Either implement them or shrink `_required_draws`.",
                stacklevel=3,
            )

    def draw_polygon(self, polygon: Polygon) -> None:
        """Render a closed lat/lon polygon (no-op by default)."""

    def draw_point(self, point: Point) -> None:
        """Render a single named lat/lon point (no-op by default)."""

    def draw_polyline(self, polyline: Polyline) -> None:
        """Render an open chain of lat/lon points (no-op by default)."""

    def draw(self, renderable: Renderable) -> None:
        """Dispatch each primitive from *renderable* to its ``draw_*`` hook."""
        for primitive in renderable.render_primitives():
            if isinstance(primitive, Polygon):
                version = _moving_version(primitive)
                if version is not None:
                    # Drawn as it is now: due again once it moves.
                    primitive.meta["_version"] = version
                    self.__dict__.setdefault("_moving_polygons", []).append(primitive)
                self.draw_polygon(primitive)
            elif isinstance(primitive, Point):
                self.draw_point(primitive)
            elif isinstance(primitive, Polyline):
                self.draw_polyline(primitive)
            else:
                raise TypeError(
                    f"Unknown render primitive: {type(primitive).__name__}"
                )

    def draw_renderables(self, renderables: Iterable[Renderable]) -> None:
        """Draw every renderable in order."""
        self.__dict__["_moving_polygons"] = []
        for renderable in renderables:
            self.draw(renderable)

    # ---- regions that move during an episode ----------------------------- #
    # A region with motion (a ``MovingFootprint``) is drawn from its shape at
    # the episode's start; each frame, :meth:`sync_moving` brings the polygons
    # of those that have moved since up to date - in place, so a view that
    # projects ``polygon.vertices`` each frame animates with no more - and
    # hands them to :meth:`on_polygons_moved` for whatever a driver caches.

    def sync_moving(self) -> list[Polygon]:
        """Update the polygons of regions that moved since the last call;
        return them (and tell :meth:`on_polygons_moved`)."""
        moved = []
        for polygon in self.__dict__.get("_moving_polygons", ()):
            version = _moving_version(polygon)
            if version is None or polygon.meta.get("_version") == version:
                continue
            bounds = polygon.meta["bounds"]
            polygon.vertices = bounds.vertices
            if polygon.per_vertex_alt is not None:
                polygon.per_vertex_alt = bounds.per_vertex_alt_range()
            polygon.meta["_version"] = version
            moved.append(polygon)
        if moved:
            self.on_polygons_moved(moved)
        return moved

    def on_polygons_moved(self, polygons: list[Polygon]) -> None:
        """React to ``polygons`` having new vertices (no-op by default: a view
        drawing ``vertices`` each frame needs nothing more)."""


def _moving_version(polygon: Polygon) -> int | None:
    """How many times the region behind ``polygon`` has moved, or None when
    it does not move."""
    footprint = getattr(polygon.meta.get("bounds"), "footprint", None)
    return getattr(footprint, "version", None)


class ViewPrimitiveFanoutMixin(PrimitiveDrawMixin):
    """Fan render primitives out to composed view objects."""

    _primitive_view_methods = {
        "polygon": "draw_polygon",
        "point": "draw_point",
        "polyline": "draw_polyline",
    }

    @property
    def primitive_targets(self) -> Iterable[object]:
        """Views that should receive render primitives."""
        raise NotImplementedError

    def _fanout_primitive(self, primitive_kind: str, primitive: object) -> None:
        method_name = self._primitive_view_methods[primitive_kind]
        for target in self.primitive_targets:
            getattr(target, method_name)(self, primitive)

    def draw_polygon(self, polygon: Polygon) -> None:
        self._fanout_primitive("polygon", polygon)

    def draw_point(self, point: Point) -> None:
        self._fanout_primitive("point", point)

    def draw_polyline(self, polyline: Polyline) -> None:
        self._fanout_primitive("polyline", polyline)
