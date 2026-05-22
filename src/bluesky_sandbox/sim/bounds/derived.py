from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point as _ShPoint
from shapely.geometry import Polygon as _Polygon
from shapely.prepared import prep

from .base import Footprint
from .coordinates import LatLon, LocalFrame
from .footprints import ShapelyFootprint, vertices_from_shape


@dataclass
class SectorFootprint(Footprint):
    """Circular sector footprint with fast analytic containment."""

    center: LatLon
    radius_nm: float
    bearing_deg: float
    half_angle_deg: float
    n_vertices: int = 24

    def __post_init__(self) -> None:
        if self.radius_nm <= 0.0:
            raise ValueError(f"radius_nm ({self.radius_nm}) must be > 0.")
        if not (0.0 < self.half_angle_deg <= 180.0):
            raise ValueError(
                f"half_angle_deg ({self.half_angle_deg}) must be in (0, 180]."
            )
        if self.n_vertices < 3:
            raise ValueError("SectorFootprint requires at least 3 arc vertices.")
        frame = LocalFrame(self.center)
        arc = [
            frame.offset(
                self.bearing_deg
                - self.half_angle_deg
                + 2.0 * self.half_angle_deg * i / self.n_vertices,
                self.radius_nm,
            )
            for i in range(self.n_vertices + 1)
        ]
        vertices = [self.center, *arc]
        shape = _Polygon([(p.lon_deg, p.lat_deg) for p in vertices])
        self._frame = frame
        self._vertices = vertices
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    @property
    def bounding_box(self) -> tuple[tuple[float, float], tuple[float, float]]:
        min_lon, min_lat, max_lon, max_lat = self._shape.bounds
        return (min_lat, max_lat), (min_lon, max_lon)

    @property
    def vertices(self) -> list[tuple[float, float]]:
        return [(p.lat_deg, p.lon_deg) for p in self._vertices]

    def contains(self, lat_deg: float, lon_deg: float) -> bool:
        x_nm, y_nm = self._frame.to_xy_nm(LatLon(lat_deg, lon_deg))
        radius_nm = math.hypot(x_nm, y_nm)
        if radius_nm > self.radius_nm:
            return False
        if radius_nm <= 1e-12:
            return True
        bearing_deg = math.degrees(math.atan2(x_nm, y_nm)) % 360.0
        return (
            abs(_angle_diff_deg(bearing_deg, self.bearing_deg))
            <= self.half_angle_deg
        )

    def sample_point(self, rng: np.random.Generator) -> tuple[float, float]:
        radius_nm = self.radius_nm * math.sqrt(float(rng.random()))
        bearing_deg = float(
            rng.uniform(
                self.bearing_deg - self.half_angle_deg,
                self.bearing_deg + self.half_angle_deg,
            )
        )
        p = self._frame.offset(bearing_deg, radius_nm)
        return p.lat_deg, p.lon_deg


@dataclass
class AnnularSectorFootprint(Footprint):
    """Ring-shaped circular sector with fast analytic containment."""

    center: LatLon
    inner_radius_nm: float
    outer_radius_nm: float
    bearing_deg: float
    half_angle_deg: float
    n_vertices: int = 48

    def __post_init__(self) -> None:
        if self.inner_radius_nm < 0.0:
            raise ValueError(
                f"inner_radius_nm ({self.inner_radius_nm}) must be >= 0."
            )
        if self.outer_radius_nm <= self.inner_radius_nm:
            raise ValueError(
                "outer_radius_nm must be greater than inner_radius_nm; got "
                f"{self.outer_radius_nm}, {self.inner_radius_nm}."
            )
        if not (0.0 < self.half_angle_deg <= 180.0):
            raise ValueError(
                f"half_angle_deg ({self.half_angle_deg}) must be in (0, 180]."
            )
        if self.n_vertices < 3:
            raise ValueError("AnnularSectorFootprint requires at least 3 arc vertices.")

        frame = LocalFrame(self.center)
        start_bearing = self.bearing_deg - self.half_angle_deg
        end_bearing = self.bearing_deg + self.half_angle_deg
        step_deg = (end_bearing - start_bearing) / self.n_vertices

        inner_arc = [
            frame.offset(
                start_bearing + step_deg * i,
                self.inner_radius_nm,
            )
            for i in range(self.n_vertices + 1)
        ]
        outer_arc = [
            frame.offset(
                end_bearing - step_deg * i,
                self.outer_radius_nm,
            )
            for i in range(self.n_vertices + 1)
        ]
        vertices = [*inner_arc, *outer_arc]
        shape = _Polygon([(p.lon_deg, p.lat_deg) for p in vertices])
        self._frame = frame
        self._vertices = vertices
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    @property
    def bounding_box(self) -> tuple[tuple[float, float], tuple[float, float]]:
        min_lon, min_lat, max_lon, max_lat = self._shape.bounds
        return (min_lat, max_lat), (min_lon, max_lon)

    @property
    def vertices(self) -> list[tuple[float, float]]:
        return [(p.lat_deg, p.lon_deg) for p in self._vertices]

    def contains(self, lat_deg: float, lon_deg: float) -> bool:
        x_nm, y_nm = self._frame.to_xy_nm(LatLon(lat_deg, lon_deg))
        radius_nm = math.hypot(x_nm, y_nm)
        if not (self.inner_radius_nm <= radius_nm <= self.outer_radius_nm):
            return False
        if radius_nm <= 1e-12:
            return self.inner_radius_nm == 0.0
        bearing_deg = math.degrees(math.atan2(x_nm, y_nm)) % 360.0
        return (
            abs(_angle_diff_deg(bearing_deg, self.bearing_deg))
            <= self.half_angle_deg
        )

    def sample_point(self, rng: np.random.Generator) -> tuple[float, float]:
        radius_nm = math.sqrt(
            float(
                rng.uniform(
                    self.inner_radius_nm**2,
                    self.outer_radius_nm**2,
                )
            )
        )
        bearing_deg = float(
            rng.uniform(
                self.bearing_deg - self.half_angle_deg,
                self.bearing_deg + self.half_angle_deg,
            )
        )
        p = self._frame.offset(bearing_deg, radius_nm)
        return p.lat_deg, p.lon_deg


@dataclass
class CorridorFootprint(Footprint):
    """Rectangular corridor around a start-to-end centerline."""

    start: LatLon
    end: LatLon
    half_width_nm: float

    def __post_init__(self) -> None:
        if self.half_width_nm <= 0.0:
            raise ValueError(f"half_width_nm ({self.half_width_nm}) must be > 0.")
        frame = LocalFrame(self.start)
        end_x, end_y = frame.to_xy_nm(self.end)
        length_nm = math.hypot(end_x, end_y)
        if length_nm <= 1e-9:
            raise ValueError("CorridorFootprint start and end must be distinct.")
        axis_x = end_x / length_nm
        axis_y = end_y / length_nm
        left_x = -axis_y
        left_y = axis_x
        corners = [
            frame.from_xy_nm(left_x * self.half_width_nm, left_y * self.half_width_nm),
            frame.from_xy_nm(-left_x * self.half_width_nm, -left_y * self.half_width_nm),
            frame.from_xy_nm(
                end_x - left_x * self.half_width_nm,
                end_y - left_y * self.half_width_nm,
            ),
            frame.from_xy_nm(
                end_x + left_x * self.half_width_nm,
                end_y + left_y * self.half_width_nm,
            ),
        ]
        shape = _Polygon([(p.lon_deg, p.lat_deg) for p in corners])
        self._frame = frame
        self._axis_x = axis_x
        self._axis_y = axis_y
        self._left_x = left_x
        self._left_y = left_y
        self._length_nm = length_nm
        self._vertices = corners
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    @property
    def bounding_box(self) -> tuple[tuple[float, float], tuple[float, float]]:
        min_lon, min_lat, max_lon, max_lat = self._shape.bounds
        return (min_lat, max_lat), (min_lon, max_lon)

    @property
    def vertices(self) -> list[tuple[float, float]]:
        return [(p.lat_deg, p.lon_deg) for p in self._vertices]

    def contains(self, lat_deg: float, lon_deg: float) -> bool:
        x_nm, y_nm = self._frame.to_xy_nm(LatLon(lat_deg, lon_deg))
        along_nm = x_nm * self._axis_x + y_nm * self._axis_y
        if not (0.0 <= along_nm <= self._length_nm):
            return False
        lateral_nm = x_nm * self._left_x + y_nm * self._left_y
        return abs(lateral_nm) <= self.half_width_nm

    def sample_point(self, rng: np.random.Generator) -> tuple[float, float]:
        along_nm = float(rng.uniform(0.0, self._length_nm))
        lateral_nm = float(rng.uniform(-self.half_width_nm, self.half_width_nm))
        p = self._frame.from_xy_nm(
            self._axis_x * along_nm + self._left_x * lateral_nm,
            self._axis_y * along_nm + self._left_y * lateral_nm,
        )
        return p.lat_deg, p.lon_deg


@dataclass
class BooleanFootprint(Footprint):
    """Footprint composed from two other footprints."""

    op: str
    left: Footprint
    right: Footprint

    def __post_init__(self) -> None:
        if self.op == "union":
            shape = self.left.shape.union(self.right.shape)
        elif self.op == "intersection":
            shape = self.left.shape.intersection(self.right.shape)
        elif self.op == "difference":
            shape = self.left.shape.difference(self.right.shape)
        else:
            raise ValueError(f"unknown BooleanFootprint op {self.op!r}.")
        if shape.is_empty or shape.area <= 0.0:
            raise ValueError("BooleanFootprint produced an empty footprint.")
        self._shape = shape
        self._prepared = prep(shape)
        self._fallback = ShapelyFootprint(shape)

    @property
    def shape(self):
        return self._shape

    @property
    def bounding_box(self) -> tuple[tuple[float, float], tuple[float, float]]:
        min_lon, min_lat, max_lon, max_lat = self._shape.bounds
        return (min_lat, max_lat), (min_lon, max_lon)

    @property
    def vertices(self) -> list[tuple[float, float]]:
        return vertices_from_shape(self._shape)

    def contains(self, lat_deg: float, lon_deg: float) -> bool:
        if self.op == "union":
            return self.left.contains(lat_deg, lon_deg) or self.right.contains(
                lat_deg,
                lon_deg,
            )
        if self.op == "intersection":
            return self.left.contains(lat_deg, lon_deg) and self.right.contains(
                lat_deg,
                lon_deg,
            )
        if self.op == "difference":
            return self.left.contains(lat_deg, lon_deg) and not self.right.contains(
                lat_deg,
                lon_deg,
            )
        return bool(self._prepared.contains(_ShPoint(lon_deg, lat_deg)))

    def sample_point(self, rng: np.random.Generator) -> tuple[float, float]:
        if self.op == "difference":
            while True:
                lat_deg, lon_deg = self.left.sample_point(rng)
                if not self.right.contains(lat_deg, lon_deg):
                    return lat_deg, lon_deg
        return self._fallback.sample_point(rng)


def _angle_diff_deg(lhs_deg: float, rhs_deg: float) -> float:
    return (float(lhs_deg) - float(rhs_deg) + 180.0) % 360.0 - 180.0
