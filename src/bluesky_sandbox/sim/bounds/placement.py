"""Where a region sits each episode: a placement draws its position - and
turn - anew, whatever its shape.

A :class:`Placement` moves a footprint to a drawn spot
(:meth:`Placement.place`), and says where it can then be
(:meth:`Placement.envelope`) - what stands for it wherever a fixed shape is
needed. A named region whose footprint is a :class:`PlacedFootprint` is placed
each episode by :func:`~.generators.generate_regions`, after its shape is
drawn (a generated one) - so a shape can be both drawn and placed anew.

Placing carries the shape through a point map (``Footprint.mapped``), so it
keeps its form: a circle stays a circle, a generator's draw its outline.

- :class:`InRegion` - its center at a point drawn uniformly inside a region,
  turned by a drawn angle; optionally kept wholly inside the region and clear
  of others.

A placement of your own is a subclass with ``place(footprint, rng)`` and
``envelope(footprint)``, referenced by import path where a name is asked for.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import shapely

from bluesky_sandbox.sim.sampling.distributions import sample_scalar

from .base import Bounds, Footprint, RegionBounds
from .coordinates import LatLon, LocalFrame
from .footprints import ShapelyFootprint

__all__ = ["InRegion", "PlacedFootprint", "Placement", "PlacementError", "moved_to"]

#: A parameter: a fixed value, a ``(low, high)`` range, or a scipy distribution.
Value = Any


class PlacementError(ValueError):
    """A placement could not find a spot meeting its constraints."""


def _footprint(region: Bounds | Footprint) -> Footprint:
    return region.footprint if isinstance(region, RegionBounds) else region


def moved_to(footprint: Footprint, to: LatLon, turn_deg: float = 0.0) -> Footprint:
    """``footprint`` with its center moved to ``to``, turned ``turn_deg``
    clockwise about it - in the flat frames of where it was and where it goes,
    so its size in nm is kept."""
    start = footprint.center_point()
    here, there = LocalFrame(start), LocalFrame(to)
    a = math.radians(turn_deg)
    c, s = math.cos(a), math.sin(a)

    def point(lat: float, lon: float) -> tuple[float, float]:
        x, y = here.to_xy_nm(LatLon(lat, lon))
        p = there.from_xy_nm(x * c + y * s, -x * s + y * c)
        return p.lat_deg, p.lon_deg

    return footprint.mapped(point)


@dataclass(frozen=True)
class Placement(ABC):
    """Draws where a footprint sits each episode."""

    @abstractmethod
    def place(self, footprint: Footprint, rng: np.random.Generator) -> Footprint:
        """``footprint`` at this episode's spot."""

    @abstractmethod
    def envelope(self, footprint: Footprint) -> Footprint:
        """A footprint ``footprint`` lies inside wherever it is placed."""

    def mapped(self, point) -> Placement:
        """This placement carried through ``point`` - what it places within
        moved with it. Placements with no position of their own are unchanged."""
        return self


@dataclass(frozen=True)
class InRegion(Placement):
    """The shape's center at a point drawn uniformly inside ``within``,
    turned ``turn_deg`` (a fixed angle, a range, a distribution; None - not
    turned). ``keep_inside`` - the whole shape inside ``within``, not only its
    center; ``avoid`` - regions it must not overlap. A spot missing either is
    redrawn, up to ``max_tries`` times."""

    within: Bounds | Footprint | None = None
    turn_deg: Value | None = None
    keep_inside: bool = False
    avoid: Sequence[Bounds | Footprint] = ()
    max_tries: int = 200

    def __post_init__(self) -> None:
        if self.within is None:
            raise ValueError("InRegion needs a region to place within")
        object.__setattr__(self, "avoid", tuple(self.avoid))

    def place(self, footprint: Footprint, rng: np.random.Generator) -> Footprint:
        region = _footprint(self.within)
        outline = region.outline().buffer(1e-9)
        avoid = [_footprint(a).outline() for a in self.avoid]
        for _ in range(max(1, int(self.max_tries))):
            lat, lon = region.sample_point(rng)
            turn = 0.0 if self.turn_deg is None else sample_scalar(self.turn_deg, rng)
            placed = moved_to(footprint, LatLon(lat, lon), turn)
            shape = placed.outline()
            if self.keep_inside and not outline.covers(shape):
                continue
            if any(_overlaps(shape, a) for a in avoid):
                continue
            return placed
        what = "spot inside its region" if self.keep_inside else "spot"
        what += " clear of the regions to avoid" if self.avoid else ""
        raise PlacementError(f"InRegion found no {what} in {self.max_tries} tries")

    def envelope(self, footprint: Footprint) -> Footprint:
        region = _footprint(self.within)
        if self.keep_inside:
            return region
        # The center anywhere in the region: the region, grown by how far the
        # shape reaches from its center.
        center = footprint.center_point()
        frame = LocalFrame(center)
        reach = 0.0
        for lat, lon in footprint.vertices:
            x, y = frame.to_xy_nm(LatLon(lat, lon))
            reach = max(reach, math.hypot(x, y))
        region_frame = LocalFrame(region.center_point())
        grown = _outline_xy(region, region_frame).buffer(reach)
        return _to_footprint(grown, region_frame)

    def mapped(self, point) -> Placement:
        def move(region):
            if isinstance(region, RegionBounds):
                return RegionBounds(region.footprint.mapped(point), region.altitude, region.name)
            return region.mapped(point)

        return InRegion(
            within=move(self.within),
            turn_deg=self.turn_deg,
            keep_inside=self.keep_inside,
            avoid=tuple(move(a) for a in self.avoid),
            max_tries=self.max_tries,
        )


def _overlaps(shape, other) -> bool:
    """Whether ``shape`` overlaps ``other`` - by area, so touching is not;
    where either has none (a point), where one covers the other: a shape is
    clear of a point it doesn't cover."""
    if shape.area == 0 or other.area == 0:
        return other.covers(shape) or shape.covers(other)
    return shape.intersects(other) and shape.intersection(other).area > 0


def _outline_xy(footprint: Footprint, frame: LocalFrame):
    """``footprint``'s outline in ``frame``'s nm."""

    def to_xy(coords):
        lon, lat = np.asarray(coords, dtype=np.float64).T
        x, y = frame.to_xy_nm(LatLon(lat, lon))
        return np.column_stack([x, y])

    return shapely.transform(footprint.outline(), to_xy)


def _to_footprint(shape_xy, frame: LocalFrame) -> Footprint:
    """A shape in ``frame``'s nm as a footprint."""

    def to_lonlat(coords):
        xy = np.asarray(coords, dtype=np.float64)
        p = frame.from_xy_nm(xy[:, 0], xy[:, 1])
        return np.column_stack([p.lon_deg, p.lat_deg])

    return ShapelyFootprint(shapely.transform(shape_xy, to_lonlat))


@dataclass
class PlacedFootprint(Footprint):
    """A named region's ``footprint`` placed by ``placement`` each episode.

    As a fixed shape - before any episode, in an environment's support - it is
    where the placement can put it (its envelope). The footprint may itself be
    a generated one: drawn, then placed."""

    footprint: Footprint
    placement: Placement

    def __post_init__(self) -> None:
        self._envelope = self.placement.envelope(self.footprint)

    @property
    def shape(self):
        return self._envelope.shape

    @property
    def bounding_box(self):
        return self._envelope.bounding_box

    @property
    def vertices(self):
        return self._envelope.vertices

    def contains(self, lat_deg: float, lon_deg: float) -> bool:
        return self._envelope.contains(lat_deg, lon_deg)

    def sample_point(self, rng: np.random.Generator) -> tuple[float, float]:
        return self._envelope.sample_point(rng)

    def outline(self):
        return self._envelope.outline()

    def center_point(self) -> LatLon:
        return self._envelope.center_point()

    def mapped(self, point) -> Footprint:
        return PlacedFootprint(self.footprint.mapped(point), self.placement.mapped(point))
