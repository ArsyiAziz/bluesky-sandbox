"""Overlay primitives + the ``Renderable`` protocol for static environment
overlays.

A small data-only vocabulary that lets resources (bounds, waypoints, routes)
describe what they want drawn without knowing about any specific driver, and
lets drivers (qtgl, pygame, future ones) consume those primitives without
knowing about any specific resource type.

Adding a new resource type -> implement :class:`Renderable.render_primitives`,
every driver picks it up automatically.

Adding a new driver -> override the ``draw_*`` methods on
:class:`~bluesky_sandbox.ui.drivers.SimDriver`, every existing resource works.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol, Union, runtime_checkable

from bluesky_sandbox.sim.bounds import Bounds


class _SelfRenderable:
    """Mixin: a primitive renders as itself.

    Lets callers treat primitives and richer resources uniformly through
    :class:`Renderable` - the env can yield a flat list of primitives and
    drivers always go through ``draw(renderable)``.
    """

    def render_primitives(self) -> Iterable[Primitive]:
        yield self  # type: ignore[misc]


@dataclass
class Polygon(_SelfRenderable):
    """A closed lat/lon polygon, optionally with an altitude band.

    Parameters
    ----------
    vertices:
        Boundary vertices as ``(lat_deg, lon_deg)`` tuples.
    color:
        Display color name (e.g. ``"red"``, ``"green"``) - drivers map to
        their own palette.
    label:
        Human-readable display label.  Drivers also derive a stable
        identifier from this for systems (like BlueSky's stack) that need
        unique polygon names; spaces are replaced with underscores.
    alt_range:
        Optional ``(alt_min_ft, alt_max_ft)`` band shown in profile-style
        views.  ``None`` means "no altitude band".
    per_vertex_alt:
        Optional per-vertex ``(alt_min_ft, alt_max_ft)`` aligned with
        :attr:`vertices`.  When set, 3D-capable drivers should use it to
        render a *slanted* (or in general, vertex-varying) altitude
        envelope rather than a flat band.  ``None`` falls back to
        :attr:`alt_range`.
    meta:
        Free-form data carried alongside the polygon.  Drivers may special-
        case behavior on keys like ``meta["kind"]`` (e.g. ``"spawn"``,
        ``"airspace"``, ``"query"``) without resources knowing about them.
    """

    vertices: list[tuple[float, float]]
    color: str = "white"
    label: str = ""
    alt_range: tuple[float, float] | None = None
    per_vertex_alt: list[tuple[float, float]] | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Point(_SelfRenderable):
    """A single named lat/lon (and optional altitude) point.

    The display primitive used for nav fixes, waypoints, aircraft
    waypoints, or any other "this thing has a position" overlay.
    """

    lat: float
    lon: float
    alt_ft: float | None = None
    label: str = ""
    color: str = "blue"
    meta: dict = field(default_factory=dict)


@dataclass
class Polyline(_SelfRenderable):
    """An open chain of ``(lat_deg, lon_deg)`` points (e.g. a planned route)."""

    points: list[tuple[float, float]]
    color: str = "white"
    label: str = ""
    meta: dict = field(default_factory=dict)


Primitive = Union[Polygon, Point, Polyline]


@runtime_checkable
class Renderable(Protocol):
    """Anything that yields render primitives.

    Implementations must not touch any driver - they only describe what to
    draw.  The driver's ``draw()`` dispatch decides how to render each
    primitive.
    """

    def render_primitives(self) -> Iterable[Primitive]: ...


@dataclass
class BoundsResource:
    """Adapts a :class:`~bluesky_sandbox.sim.bounds.Bounds` into a :class:`Renderable`.

    Reads vertices and altitude limits straight off the wrapped bounds -
    no copying - so :mod:`bounds` stays purely geometric while still being
    paintable through the render layer.

    Parameters
    ----------
    bounds:
        The geometric region to display.
    color:
        Display color name passed through to :class:`Polygon`.
    label:
        Display label, also used as the polygon identifier.
    kind:
        Free-form tag stashed in ``meta["kind"]`` so drivers can special-
        case rendering (e.g. ``"airspace"`` gets a thicker alt line).
    alt_range_override:
        Optional ``(lo, hi)`` that wins over the bounds' own alt limits.
        Used for spawn regions where the rendered band should reflect the
        spawn distribution range, not the (typically infinite) spatial
        bounds altitude limits.
    extra_meta:
        Additional fields merged into the primitive's ``meta`` dict -
        e.g. spawn regions add ``{"spawn_alt": params['alt_ft']}`` so the
        pygame hover tooltip can display the distribution range.
    """

    bounds: Bounds
    color: str
    label: str
    kind: str = ""
    alt_range_override: tuple[float, float] | None = None
    extra_meta: dict = field(default_factory=dict)

    def render_primitives(self) -> Iterable[Primitive]:
        if self.alt_range_override is not None:
            alt_range = self.alt_range_override
        else:
            lo = getattr(self.bounds, "alt_min_ft", float("-inf"))
            hi = getattr(self.bounds, "alt_max_ft", float("inf"))
            alt_range = (lo, hi) if math.isfinite(lo) and math.isfinite(hi) else None

        meta = {"kind": self.kind, "name": self.label, "bounds": self.bounds}
        meta.update(self.extra_meta)

        # Sloped bounds expose per-vertex altitudes; carry them through
        # to the primitive so 3D drivers can paint a slanted envelope
        # instead of a flat band.  Flat-band bounds return ``None`` so
        # this stays a no-op in the common case.
        per_vertex_alt = self.bounds.per_vertex_alt_range()
        # An ``alt_range_override`` is meant to win over the bounds'
        # native altitudes (e.g. spawn band overriding +/-inf), so drop
        # per-vertex info in that case rather than silently mixing the
        # two.
        if self.alt_range_override is not None:
            per_vertex_alt = None

        yield Polygon(
            vertices=self.bounds.vertices,
            color=self.color,
            label=self.label,
            alt_range=alt_range,
            per_vertex_alt=per_vertex_alt,
            meta=meta,
        )
