from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .coordinates import LatLon, LocalFrame
from .geometry import Boundary, Frame, boundary_of


def _boolean(op: str, left: Footprint, right: Footprint) -> Footprint:
    """Compose two footprints into a ``BooleanFootprint``.

    Late import: ``.derived`` subclasses :class:`Footprint`, so it imports this
    module at load time - naming it at module scope here would close the cycle
    and leave ``derived`` inheriting from a half-built ``base``. By the time any
    of these run, both modules are loaded. Kept in one helper so the three
    operators below share a single deferral rather than repeating it.
    """
    from .derived import BooleanFootprint  # noqa: PLC0415

    return BooleanFootprint(op, left, right)


class Footprint(ABC):
    """Horizontal region primitive in lat/lon space."""

    @property
    @abstractmethod
    def bounding_box(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return ``((lat_min, lat_max), (lon_min, lon_max))``."""

    @property
    @abstractmethod
    def vertices(self) -> list[tuple[float, float]]:
        """Boundary vertices as ``(lat_deg, lon_deg)`` for rendering."""

    @property
    @abstractmethod
    def shape(self):
        """Shapely geometry used for fallback boolean ops and rendering."""

    @abstractmethod
    def contains(self, lat_deg: float, lon_deg: float) -> bool:
        """Return ``True`` when the horizontal point lies inside."""

    @abstractmethod
    def sample_point(self, rng: np.random.Generator) -> tuple[float, float]:
        """Draw a random ``(lat_deg, lon_deg)`` point inside this footprint."""

    # ---- its geometry, for a region's boundary (see .geometry) ---------- #
    def outline(self):
        """The footprint's exact outline, shapely in lon/lat degrees: its arcs
        traced finely. ``shape`` is coarser, for drawing."""
        return self.shape

    def circles(self) -> list[tuple[LatLon, float]]:
        """The circles its arcs lie on: ``(center, radius_nm)`` each."""
        return []

    def center_point(self) -> LatLon:
        """Its center: a round shape's own, else the outline's centroid."""
        c = self.outline().centroid
        return LatLon(c.y, c.x)

    def axis_deg(self) -> float | None:
        """The bearing it points along (a corridor's, a sector's), or None."""
        return None

    def mapped(self, point) -> Footprint:
        """This footprint carried through ``point`` (a ``(lat, lon) -> (lat,
        lon)`` map: a rotation, translation, scaling): the same primitive
        where it can be, else the polygon of its outline."""
        from .footprints import PolygonFootprint  # noqa: PLC0415

        return PolygonFootprint([point(lat, lon) for lat, lon in self.vertices])

    def union(self, other: Footprint) -> Footprint:
        return _boolean("union", self, other)

    def intersection(self, other: Footprint) -> Footprint:
        return _boolean("intersection", self, other)

    def difference(self, other: Footprint) -> Footprint:
        return _boolean("difference", self, other)

    def __or__(self, other: Footprint) -> Footprint:
        return self.union(other)

    def __and__(self, other: Footprint) -> Footprint:
        return self.intersection(other)

    def __sub__(self, other: Footprint) -> Footprint:
        return self.difference(other)


class AltitudeBand(ABC):
    """Altitude rule attached to a horizontal footprint."""

    @property
    @abstractmethod
    def min_ft(self) -> float:
        """Lowest possible altitude accepted by this band."""

    @property
    @abstractmethod
    def max_ft(self) -> float:
        """Highest possible altitude accepted by this band."""

    @abstractmethod
    def band_at(self, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        """Return ``(alt_min_ft, alt_max_ft)`` at a horizontal position."""

    def contains(self, lat_deg: float, lon_deg: float, alt_ft: float) -> bool:
        lo, hi = self.band_at(lat_deg, lon_deg)
        return lo <= alt_ft <= hi

    def per_vertex_alt_range(
        self,
        vertices: list[tuple[float, float]],
    ) -> list[tuple[float, float]] | None:
        del vertices
        return None


class Bounds(ABC):
    """Abstract base for spatial bounds with optional altitude range.

    Subclasses define the lateral shape of the region used for spawn sampling
    and in-bounds checks. All lat/lon coordinates are degrees; altitudes are
    feet.
    """

    @property
    @abstractmethod
    def bounding_box(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return ``((lat_min_deg, lat_max_deg), (lon_min_deg, lon_max_deg))``."""

    @abstractmethod
    def contains(
        self,
        lat_deg: float,
        lon_deg: float,
        alt_ft: float | None = None,
    ) -> bool:
        """Return ``True`` if ``(lat, lon[, alt])`` lies inside this region."""

    @abstractmethod
    def sample_point(self, rng: np.random.Generator) -> tuple[float, float]:
        """Draw a uniform random ``(lat_deg, lon_deg)`` point from the region."""

    @property
    @abstractmethod
    def vertices(self) -> list[tuple[float, float]]:
        """Return ``(lat_deg, lon_deg)`` boundary vertices for drawing."""

    def per_vertex_alt_range(self) -> list[tuple[float, float]] | None:
        return None


@dataclass
class RegionBounds(Bounds):
    """Bounds built from a horizontal footprint and an altitude band.

    ``name`` - the named region this is, where it stands in for one (a
    generator's parent, a region a placement keeps clear of): so an episode's
    draw of that region can stand in for it in turn."""

    footprint: Footprint
    altitude: AltitudeBand | None = None
    name: str | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.altitude is None:
            # Late import: .altitude imports this module for AltitudeBand.
            from .altitude import ConstantAltitudeBand  # noqa: PLC0415

            self.altitude = ConstantAltitudeBand()
        self._shape = self.footprint.shape
        self.alt_min_ft = self.altitude.min_ft
        self.alt_max_ft = self.altitude.max_ft

    @property
    def bounding_box(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self.footprint.bounding_box

    @property
    def vertices(self) -> list[tuple[float, float]]:
        return self.footprint.vertices

    def alt_band_at(self, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        return self.altitude.band_at(lat_deg, lon_deg)

    def per_vertex_alt_range(self) -> list[tuple[float, float]] | None:
        return self.altitude.per_vertex_alt_range(self.vertices)

    def contains(
        self,
        lat_deg: float | LatLon,
        lon_deg: float | None = None,
        alt_ft: float | None = None,
    ) -> bool:
        """Whether ``(lat, lon[, alt])`` lies inside - or a point: ``contains(p,
        alt_ft=...)``, whose arrays give one answer per entry."""
        if isinstance(lat_deg, LatLon):
            if lon_deg is not None:
                raise TypeError("contains(point) takes its altitude as alt_ft=...")
            return _each(lat_deg, alt_ft, self.contains)
        if alt_ft is not None and not (self.alt_min_ft <= alt_ft <= self.alt_max_ft):
            return False
        if not self.footprint.contains(lat_deg, lon_deg):
            return False
        if alt_ft is None:
            return True
        return self.altitude.contains(lat_deg, lon_deg, alt_ft)

    def sample_point(self, rng: np.random.Generator) -> tuple[float, float]:
        return self.footprint.sample_point(rng)

    # ---- its geometry (see .geometry) ------------------------------------ #
    @property
    def center(self) -> LatLon:
        """Its center, absolute: a circle's or sector's own, a corridor's
        midpoint, else the area centroid (which can lie outside, in a ring)."""
        return self.footprint.center_point()

    @property
    def frame(self) -> Frame:
        """Coordinates relative to the center, nm: ``frame.xy_nm(p)`` east and
        north, or ``frame(oriented=True)`` along its axis and across it."""
        axis = self.footprint.axis_deg()
        return Frame(self.center, getattr(self, "_orientation_deg", 0.0) if axis is None else axis)

    @property
    def boundary(self) -> Boundary:
        """Its boundary as primitives: ``faces`` (Segment, Arc), ``vertices``,
        and ``nearest(p)``."""
        # A moving footprint counts its moves: the boundary is of the shape now.
        version = getattr(self.footprint, "version", None)
        cached = getattr(self, "_boundary", None)
        if cached is None or cached[0] != version:
            cached = (
                version,
                boundary_of(
                    self.footprint.outline(),
                    self.footprint.circles(),
                    LocalFrame(self.center),
                    lambda p: self.contains(p),
                ),
            )
            self._boundary = cached
        return cached[1]

    def floor_ft(self, p: LatLon):
        """The lowest altitude inside, at ``p``."""
        return _each(p, None, lambda lat, lon, _: self.alt_band_at(lat, lon)[0])

    def ceiling_ft(self, p: LatLon):
        """The highest altitude inside, at ``p``."""
        return _each(p, None, lambda lat, lon, _: self.alt_band_at(lat, lon)[1])

    @property
    def operation(self) -> str | None:
        """How a combined shape is made - ``"union"``, ``"intersection"``,
        ``"difference"`` - or None for a plain one."""
        return getattr(self.footprint, "op", None) if self.parts else None

    @property
    def parts(self) -> list[RegionBounds]:
        """A combined shape's two pieces (each a region, with this altitude), in
        order; [] for a plain shape."""
        left = getattr(self.footprint, "left", None)
        right = getattr(self.footprint, "right", None)
        if not isinstance(left, Footprint) or not isinstance(right, Footprint):
            return []
        return [RegionBounds(left, self.altitude), RegionBounds(right, self.altitude)]


def _each(p: LatLon, alt_ft, fn):
    """``fn(lat, lon, alt_ft)`` at ``p`` - one answer per entry, as an array,
    when it holds arrays."""
    lat = np.asarray(p.lat_deg, dtype=np.float64)
    lon = np.asarray(p.lon_deg, dtype=np.float64)
    if lat.ndim == 0 and lon.ndim == 0 and np.ndim(alt_ft) == 0:
        return fn(float(lat), float(lon), alt_ft)
    lat, lon = np.broadcast_arrays(np.atleast_1d(lat), np.atleast_1d(lon))
    alts = [None] * lat.shape[0] if alt_ft is None else np.broadcast_to(alt_ft, lat.shape)
    return np.asarray(
        [fn(float(a), float(b), None if h is None else float(h)) for a, b, h in zip(lat, lon, alts)]
    )
