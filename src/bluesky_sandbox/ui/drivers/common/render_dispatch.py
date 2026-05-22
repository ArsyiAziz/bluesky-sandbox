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
        for renderable in renderables:
            self.draw(renderable)


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
