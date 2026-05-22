from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LatLon:
    """Geographic point in degrees."""

    lat_deg: float
    lon_deg: float


@dataclass
class LocalFrame:
    """Small-area lat/lon <-> nautical-mile frame around one origin."""

    origin: LatLon

    def __post_init__(self) -> None:
        cos_lat = math.cos(math.radians(self.origin.lat_deg))
        if abs(cos_lat) < 1e-6:
            raise ValueError(
                "LocalFrame is not supported within ~0.00006 deg of the poles."
            )
        self._cos_lat = cos_lat

    def to_xy_nm(self, point: LatLon) -> tuple[float, float]:
        return (
            (point.lon_deg - self.origin.lon_deg) * 60.0 * self._cos_lat,
            (point.lat_deg - self.origin.lat_deg) * 60.0,
        )

    def from_xy_nm(self, x_nm: float, y_nm: float) -> LatLon:
        return LatLon(
            lat_deg=self.origin.lat_deg + y_nm / 60.0,
            lon_deg=self.origin.lon_deg + x_nm / (60.0 * self._cos_lat),
        )

    def offset(self, bearing_deg: float, distance_nm: float) -> LatLon:
        brg_rad = math.radians(bearing_deg)
        return self.from_xy_nm(
            x_nm=distance_nm * math.sin(brg_rad),
            y_nm=distance_nm * math.cos(brg_rad),
        )
