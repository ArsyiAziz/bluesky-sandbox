from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .base import AltitudeBand
from .coordinates import LatLon, LocalFrame


@dataclass
class ConstantAltitudeBand(AltitudeBand):
    """Flat altitude band."""

    min_ft: float = float("-inf")
    max_ft: float = float("inf")

    def __post_init__(self) -> None:
        if self.min_ft >= self.max_ft:
            raise ValueError(
                f"min_ft ({self.min_ft}) must be < max_ft ({self.max_ft})."
            )

    def band_at(self, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        del lat_deg, lon_deg
        return self.min_ft, self.max_ft


@dataclass
class LinearAltitudeBand(AltitudeBand):
    """Altitude band linearly interpolated along a start-to-end axis."""

    start: LatLon
    end: LatLon
    start_band_ft: tuple[float, float]
    end_band_ft: tuple[float, float]

    def __post_init__(self) -> None:
        self.validate_band(self.start_band_ft, "start_band_ft")
        self.validate_band(self.end_band_ft, "end_band_ft")
        frame = LocalFrame(self.start)
        end_x, end_y = frame.to_xy_nm(self.end)
        length2 = end_x * end_x + end_y * end_y
        if length2 <= 1e-12:
            raise ValueError("LinearAltitudeBand start and end must be distinct.")
        self._frame = frame
        self._axis_x_nm = end_x
        self._axis_y_nm = end_y
        self._axis_len2_nm = length2

    @property
    def min_ft(self) -> float:
        return min(self.start_band_ft[0], self.end_band_ft[0])

    @property
    def max_ft(self) -> float:
        return max(self.start_band_ft[1], self.end_band_ft[1])

    def band_at(self, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        x_nm, y_nm = self._frame.to_xy_nm(LatLon(lat_deg, lon_deg))
        t = (
            x_nm * self._axis_x_nm + y_nm * self._axis_y_nm
        ) / self._axis_len2_nm
        t = float(np.clip(t, 0.0, 1.0))
        lo = (1.0 - t) * self.start_band_ft[0] + t * self.end_band_ft[0]
        hi = (1.0 - t) * self.start_band_ft[1] + t * self.end_band_ft[1]
        return float(lo), float(hi)

    def per_vertex_alt_range(
        self,
        vertices: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        return [self.band_at(lat_deg, lon_deg) for lat_deg, lon_deg in vertices]

    @staticmethod
    def validate_band(band: tuple[float, float], name: str) -> None:
        if band[0] >= band[1]:
            raise ValueError(f"{name} min ({band[0]}) must be < max ({band[1]}).")


@dataclass
class RadialAltitudeBand(AltitudeBand):
    """Altitude band interpolated by distance from a center point."""

    center: LatLon
    radius_nm: float
    inner_band_ft: tuple[float, float]
    outer_band_ft: tuple[float, float]

    def __post_init__(self) -> None:
        if self.radius_nm <= 0.0:
            raise ValueError(f"radius_nm ({self.radius_nm}) must be > 0.")
        LinearAltitudeBand.validate_band(self.inner_band_ft, "inner_band_ft")
        LinearAltitudeBand.validate_band(self.outer_band_ft, "outer_band_ft")
        self._frame = LocalFrame(self.center)

    @property
    def min_ft(self) -> float:
        return min(self.inner_band_ft[0], self.outer_band_ft[0])

    @property
    def max_ft(self) -> float:
        return max(self.inner_band_ft[1], self.outer_band_ft[1])

    def band_at(self, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        x_nm, y_nm = self._frame.to_xy_nm(LatLon(lat_deg, lon_deg))
        t = float(np.clip(math.hypot(x_nm, y_nm) / self.radius_nm, 0.0, 1.0))
        lo = (1.0 - t) * self.inner_band_ft[0] + t * self.outer_band_ft[0]
        hi = (1.0 - t) * self.inner_band_ft[1] + t * self.outer_band_ft[1]
        return float(lo), float(hi)

    def per_vertex_alt_range(
        self,
        vertices: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        return [self.band_at(lat_deg, lon_deg) for lat_deg, lon_deg in vertices]


@dataclass
class VertexAltitudeBand(AltitudeBand):
    """Altitude band interpolated from per-vertex altitude bands."""

    vertices: list[tuple[float, float]]
    min_values_ft: float | list[float]
    max_values_ft: float | list[float]

    def __post_init__(self) -> None:
        n = len(self.vertices)
        if n < 3:
            raise ValueError("VertexAltitudeBand requires at least 3 vertices.")
        lows = self._broadcast(self.min_values_ft, n, "min_values_ft")
        highs = self._broadcast(self.max_values_ft, n, "max_values_ft")
        for i, (lo, hi) in enumerate(zip(lows, highs)):
            if lo >= hi:
                raise ValueError(f"vertex[{i}] min ({lo}) must be < max ({hi}).")
        self._lats = np.array([lat for lat, _ in self.vertices])
        self._lons = np.array([lon for _, lon in self.vertices])
        self._lows = np.array(lows, dtype=np.float64)
        self._highs = np.array(highs, dtype=np.float64)

    @property
    def min_ft(self) -> float:
        return float(self._lows.min())

    @property
    def max_ft(self) -> float:
        return float(self._highs.max())

    def band_at(self, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        eps = 1e-12
        dx = self._lons - lon_deg
        dy = self._lats - lat_deg
        d = np.sqrt(dx * dx + dy * dy)
        on_vertex = np.where(d < eps)[0]
        if len(on_vertex):
            i = int(on_vertex[0])
            return float(self._lows[i]), float(self._highs[i])

        dx_n = np.roll(dx, -1)
        dy_n = np.roll(dy, -1)
        d_n = np.roll(d, -1)
        cross = dx * dy_n - dy * dx_n
        dot = dx * dx_n + dy * dy_n
        on_edge = np.where((np.abs(cross) < eps) & (dot < 0))[0]
        if len(on_edge):
            i = int(on_edge[0])
            j = (i + 1) % len(d)
            t = d[i] / (d[i] + d[j])
            lo = (1.0 - t) * self._lows[i] + t * self._lows[j]
            hi = (1.0 - t) * self._highs[i] + t * self._highs[j]
            return float(lo), float(hi)

        tan_half = cross / (d * d_n + dot)
        weights = (np.roll(tan_half, 1) + tan_half) / d
        total = float(weights.sum())
        lo = float((weights * self._lows).sum() / total)
        hi = float((weights * self._highs).sum() / total)
        return lo, hi

    def per_vertex_alt_range(
        self,
        vertices: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        if vertices == self.vertices:
            return [(float(lo), float(hi)) for lo, hi in zip(self._lows, self._highs)]
        return [self.band_at(lat_deg, lon_deg) for lat_deg, lon_deg in vertices]

    @staticmethod
    def _broadcast(value: float | list[float], n: int, name: str) -> list[float]:
        if isinstance(value, (int, float)):
            return [float(value)] * n
        if len(value) != n:
            raise ValueError(f"{name} length {len(value)} != vertices length ({n}).")
        return [float(x) for x in value]
