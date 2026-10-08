from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point as _ShPoint
from shapely.geometry import Polygon as _Polygon
from shapely.geometry import box as _shapely_box
from shapely.ops import unary_union
from shapely.prepared import prep

from .base import Footprint
from .coordinates import LatLon, LocalFrame

#: The widest step between points of an arc in an outline, deg: within 0.004 %
#: of the radius of the true circle.
_OUTLINE_STEP_DEG = 1.0


def arc_points(frame: LocalFrame, from_deg: float, to_deg: float, radius_nm: float) -> list[LatLon]:
    """Points on the circle of ``radius_nm`` about ``frame``'s origin, from
    bearing ``from_deg`` to ``to_deg`` (both included), finely."""
    n = max(1, math.ceil(abs(to_deg - from_deg) / _OUTLINE_STEP_DEG))
    return [frame.offset(from_deg + (to_deg - from_deg) * i / n, radius_nm) for i in range(n + 1)]


def circle_points(frame: LocalFrame, radius_nm: float) -> list[LatLon]:
    """The whole circle, finely, without repeating its first point."""
    return arc_points(frame, 0.0, 360.0, radius_nm)[:-1]


def polygon_of(points: list[LatLon], holes: list[list[LatLon]] = ()):
    """A shapely polygon (lon/lat) through ``points``, with ``holes``."""
    return _Polygon(
        [(p.lon_deg, p.lat_deg) for p in points],
        [[(p.lon_deg, p.lat_deg) for p in hole] for hole in holes],
    )


def similarity(point, center: LatLon, radius_nm: float) -> tuple[LatLon, float, float]:
    """What ``point`` (a rotation, translation, scaling) does about ``center``:
    where it goes, how much it scales ``radius_nm``, and how far it turns
    bearings (deg) - read from where a rim point due north lands."""
    rim = LocalFrame(center).offset(0.0, radius_nm)
    moved = LatLon(*point(center.lat_deg, center.lon_deg))
    x, y = LocalFrame(moved).to_xy_nm(LatLon(*point(rim.lat_deg, rim.lon_deg)))
    return moved, math.hypot(x, y) / radius_nm, math.degrees(math.atan2(x, y))


def _mapped_shape(shape, point):
    """A shapely polygon or multipolygon (lon/lat) with every ring carried
    through ``point``."""
    from shapely.geometry import MultiPolygon  # noqa: PLC0415

    def ring(coords):
        return [point(lat, lon)[::-1] for lon, lat in coords]

    def polygon(p):
        return _Polygon(ring(p.exterior.coords), [ring(h.coords) for h in p.interiors])

    if shape.geom_type == "MultiPolygon":
        return MultiPolygon([polygon(p) for p in shape.geoms])
    return polygon(shape)


@dataclass
class BoxFootprint(Footprint):
    """Axis-aligned rectangular footprint."""

    lat_min_deg: float
    lat_max_deg: float
    lon_min_deg: float
    lon_max_deg: float

    def __post_init__(self) -> None:
        if self.lat_min_deg >= self.lat_max_deg:
            raise ValueError(
                f"lat_min_deg ({self.lat_min_deg}) must be < lat_max_deg ({self.lat_max_deg})."
            )
        if self.lon_min_deg >= self.lon_max_deg:
            raise ValueError(
                f"lon_min_deg ({self.lon_min_deg}) must be < lon_max_deg ({self.lon_max_deg})."
            )
        self._shape = _shapely_box(
            self.lon_min_deg,
            self.lat_min_deg,
            self.lon_max_deg,
            self.lat_max_deg,
        )

    @property
    def shape(self):
        return self._shape

    @property
    def bounding_box(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (self.lat_min_deg, self.lat_max_deg), (self.lon_min_deg, self.lon_max_deg)

    @property
    def vertices(self) -> list[tuple[float, float]]:
        return vertices_from_shape(self._shape)

    def contains(self, lat_deg: float, lon_deg: float) -> bool:
        return (
            self.lat_min_deg < lat_deg < self.lat_max_deg
            and self.lon_min_deg < lon_deg < self.lon_max_deg
        )

    def sample_point(self, rng: np.random.Generator) -> tuple[float, float]:
        return (
            float(rng.uniform(self.lat_min_deg, self.lat_max_deg)),
            float(rng.uniform(self.lon_min_deg, self.lon_max_deg)),
        )

    def center_point(self) -> LatLon:
        return LatLon(
            (self.lat_min_deg + self.lat_max_deg) / 2.0,
            (self.lon_min_deg + self.lon_max_deg) / 2.0,
        )


@dataclass
class PointFootprint(Footprint):
    """A single position, ``center``: where something is - a fix, a spawn -
    not a region to be inside. It has no area, so nothing is
    inside it (``contains`` is False) and a point drawn from it is itself. It
    places, moves and turns with a group like any shape; its boundary is one
    vertex and no face.

    ``fix`` - the navdb fix it is at, by name, if any (see :meth:`at_fix`): a
    waypoint at it goes by that name. Moved off it, it is a position again.
    """

    center: LatLon
    fix: str | None = None

    @classmethod
    def at_fix(cls, ident: str) -> PointFootprint:
        """The point at navdb fix ``ident`` (``"EKROS"``)."""
        import bluesky as bs  # noqa: PLC0415 - the navdb, on demand

        from bluesky_sandbox.sim.queryables import _ensure_navdb_loaded  # noqa: PLC0415

        _ensure_navdb_loaded()
        idx = bs.navdb.getwpidx(ident.upper())
        if idx < 0:
            raise ValueError(f"fix {ident!r} not found in the BlueSky navdb")
        return cls(LatLon(float(bs.navdb.wplat[idx]), float(bs.navdb.wplon[idx])), ident.upper())

    def __post_init__(self) -> None:
        self.center = LatLon(float(self.center.lat_deg), float(self.center.lon_deg))
        self._shape = _ShPoint(self.center.lon_deg, self.center.lat_deg)

    @property
    def shape(self):
        return self._shape

    @property
    def bounding_box(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (self.center.lat_deg, self.center.lat_deg), (self.center.lon_deg, self.center.lon_deg)

    @property
    def vertices(self) -> list[tuple[float, float]]:
        return [(self.center.lat_deg, self.center.lon_deg)]

    def contains(self, lat_deg: float, lon_deg: float) -> bool:
        del lat_deg, lon_deg
        return False

    def sample_point(self, rng: np.random.Generator) -> tuple[float, float]:
        del rng
        return self.center.lat_deg, self.center.lon_deg

    def center_point(self) -> LatLon:
        return self.center

    def mapped(self, point) -> PointFootprint:
        to = LatLon(*point(self.center.lat_deg, self.center.lon_deg))
        stays = abs(to.lat_deg - self.center.lat_deg) < 1e-9 and abs(to.lon_deg - self.center.lon_deg) < 1e-9
        return PointFootprint(to, self.fix if stays else None)


@dataclass
class DiskFootprint(Footprint):
    """Circular footprint in a local tangent plane."""

    center: LatLon
    radius_nm: float
    n_vertices: int = 72

    def __post_init__(self) -> None:
        if self.radius_nm <= 0.0:
            raise ValueError(f"radius_nm ({self.radius_nm}) must be > 0.")
        if self.n_vertices < 12:
            raise ValueError("DiskFootprint requires at least 12 vertices.")
        frame = LocalFrame(self.center)
        vertices = [
            frame.offset(bearing_deg, self.radius_nm)
            for bearing_deg in np.linspace(0.0, 360.0, self.n_vertices, endpoint=False)
        ]
        shape = _Polygon([(p.lon_deg, p.lat_deg) for p in vertices])
        self._frame = frame
        self._lat_radius_deg = self.radius_nm / 60.0
        self._lon_radius_deg = self.radius_nm / (60.0 * frame._cos_lat)
        self._vertices = vertices
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    @property
    def bounding_box(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            self.center.lat_deg - self._lat_radius_deg,
            self.center.lat_deg + self._lat_radius_deg,
        ), (
            self.center.lon_deg - self._lon_radius_deg,
            self.center.lon_deg + self._lon_radius_deg,
        )

    @property
    def vertices(self) -> list[tuple[float, float]]:
        return [(p.lat_deg, p.lon_deg) for p in self._vertices]

    def contains(self, lat_deg: float, lon_deg: float) -> bool:
        (lat_min, lat_max), (lon_min, lon_max) = self.bounding_box
        if not (lat_min <= lat_deg <= lat_max and lon_min <= lon_deg <= lon_max):
            return False
        x_nm, y_nm = self._frame.to_xy_nm(LatLon(lat_deg, lon_deg))
        return math.hypot(x_nm, y_nm) <= self.radius_nm

    def sample_point(self, rng: np.random.Generator) -> tuple[float, float]:
        radius_nm = self.radius_nm * math.sqrt(float(rng.random()))
        bearing_deg = float(rng.uniform(0.0, 360.0))
        p = self._frame.offset(bearing_deg, radius_nm)
        return p.lat_deg, p.lon_deg

    def outline(self):
        return polygon_of(circle_points(self._frame, self.radius_nm))

    def circles(self) -> list[tuple[LatLon, float]]:
        return [(self.center, self.radius_nm)]

    def center_point(self) -> LatLon:
        return self.center

    def mapped(self, point) -> DiskFootprint:
        center, scale, _ = similarity(point, self.center, self.radius_nm)
        return DiskFootprint(center, self.radius_nm * scale, self.n_vertices)


@dataclass
class PolygonFootprint(Footprint):
    """Arbitrary polygon footprint backed by Shapely."""

    coords: list[tuple[float, float]]

    def __post_init__(self) -> None:
        if len(self.coords) < 3:
            raise ValueError("PolygonFootprint requires at least 3 vertices.")
        shape = _Polygon([(lon_deg, lat_deg) for lat_deg, lon_deg in self.coords])
        if not shape.is_valid:
            shape = shape.buffer(0)
        prepared = prep(shape)
        min_lon, min_lat, max_lon, max_lat = shape.bounds
        self._shape = shape
        self._prepared = prepared
        self._min_lon = min_lon
        self._min_lat = min_lat
        self._max_lon = max_lon
        self._max_lat = max_lat

    def mapped(self, point) -> PolygonFootprint:
        return PolygonFootprint([point(lat, lon) for lat, lon in self.coords])

    @classmethod
    def from_shape(cls, shape) -> PolygonFootprint:
        if shape.geom_type == "MultiPolygon":
            shape = max(shape.geoms, key=lambda geom: geom.area)
        return cls(vertices_from_shape(shape))

    @property
    def shape(self):
        return self._shape

    @property
    def bounding_box(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (self._min_lat, self._max_lat), (self._min_lon, self._max_lon)

    @property
    def vertices(self) -> list[tuple[float, float]]:
        return vertices_from_shape(self._shape)

    def contains(self, lat_deg: float, lon_deg: float) -> bool:
        if not (
            self._min_lat < lat_deg < self._max_lat
            and self._min_lon < lon_deg < self._max_lon
        ):
            return False
        return bool(self._prepared.contains(_ShPoint(lon_deg, lat_deg)))

    def sample_point(self, rng: np.random.Generator) -> tuple[float, float]:
        while True:
            lon_deg = float(rng.uniform(self._min_lon, self._max_lon))
            lat_deg = float(rng.uniform(self._min_lat, self._max_lat))
            if self._prepared.contains(_ShPoint(lon_deg, lat_deg)):
                return lat_deg, lon_deg


@dataclass
class ShapelyFootprint(Footprint):
    """Fallback footprint for arbitrary Shapely polygonal results."""

    shape_value: object

    def __post_init__(self) -> None:
        shape = self.shape_value
        if shape.geom_type == "MultiPolygon":
            shape = unary_union(shape.geoms)
        if shape.is_empty or shape.area <= 0.0:
            raise ValueError("Shapely footprint must have positive polygonal area.")
        min_lon, min_lat, max_lon, max_lat = shape.bounds
        self._shape = shape
        self._prepared = prep(shape)
        self._min_lon = min_lon
        self._min_lat = min_lat
        self._max_lon = max_lon
        self._max_lat = max_lat

    @property
    def shape(self):
        return self._shape

    @property
    def bounding_box(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (self._min_lat, self._max_lat), (self._min_lon, self._max_lon)

    @property
    def vertices(self) -> list[tuple[float, float]]:
        return vertices_from_shape(self._shape)

    def contains(self, lat_deg: float, lon_deg: float) -> bool:
        if not (
            self._min_lat < lat_deg < self._max_lat
            and self._min_lon < lon_deg < self._max_lon
        ):
            return False
        return bool(self._prepared.contains(_ShPoint(lon_deg, lat_deg)))

    def sample_point(self, rng: np.random.Generator) -> tuple[float, float]:
        while True:
            lon_deg = float(rng.uniform(self._min_lon, self._max_lon))
            lat_deg = float(rng.uniform(self._min_lat, self._max_lat))
            if self._prepared.contains(_ShPoint(lon_deg, lat_deg)):
                return lat_deg, lon_deg

    def mapped(self, point) -> ShapelyFootprint:
        return ShapelyFootprint(_mapped_shape(self._shape, point))


def vertices_from_shape(shape) -> list[tuple[float, float]]:
    if shape.geom_type == "MultiPolygon":
        shape = max(shape.geoms, key=lambda geom: geom.area)
    coords = list(shape.exterior.coords)[:-1]
    return [(lat_deg, lon_deg) for lon_deg, lat_deg in coords]


def union_footprints(footprints: list[Footprint]) -> Footprint:
    """Union of footprints as one footprint (convex where shapely says so).

    Used to build the *support* shape of a footprint whose scalar parameters
    are sampled per episode: the union over the parameter endpoints covers
    (for parameters that grow/shrink the shape monotonically) every shape the
    sampler can draw. Positional parameters (e.g. a sampled bearing) are only
    covered at their endpoints - orient with a rotation transform instead.
    """
    if not footprints:
        raise ValueError("union_footprints requires at least one footprint")
    if len(footprints) == 1:
        return footprints[0]
    return ShapelyFootprint(unary_union([fp.shape for fp in footprints]))
