"""What the environment asks to be drawn, independent of any driver.

A data-only vocabulary: :mod:`.overlays` carries the geometric primitives and
the ``Renderable`` protocol, :mod:`.readouts` builds the per-aircraft
annotations that sit beside them. Resources describe what they want drawn
without knowing a driver; drivers consume it without knowing a resource type.
"""

from __future__ import annotations

from .overlays import (
    BoundsResource,
    Point,
    Polygon,
    Polyline,
    Renderable,
)
from .readouts import waypoint_readouts

__all__ = [
    "BoundsResource",
    "Point",
    "Polygon",
    "Polyline",
    "Renderable",
    "waypoint_readouts",
]
