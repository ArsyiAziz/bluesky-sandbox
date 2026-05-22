from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


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

    def union(self, other: Footprint) -> Footprint:
        from .derived import BooleanFootprint

        return BooleanFootprint("union", self, other)

    def intersection(self, other: Footprint) -> Footprint:
        from .derived import BooleanFootprint

        return BooleanFootprint("intersection", self, other)

    def difference(self, other: Footprint) -> Footprint:
        from .derived import BooleanFootprint

        return BooleanFootprint("difference", self, other)

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
    """Bounds built from a horizontal footprint and an altitude band."""

    footprint: Footprint
    altitude: AltitudeBand | None = None

    def __post_init__(self) -> None:
        if self.altitude is None:
            from .altitude import ConstantAltitudeBand

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
        lat_deg: float,
        lon_deg: float,
        alt_ft: float | None = None,
    ) -> bool:
        if alt_ft is not None and not (self.alt_min_ft <= alt_ft <= self.alt_max_ft):
            return False
        if not self.footprint.contains(lat_deg, lon_deg):
            return False
        if alt_ft is None:
            return True
        return self.altitude.contains(lat_deg, lon_deg, alt_ft)

    def sample_point(self, rng: np.random.Generator) -> tuple[float, float]:
        return self.footprint.sample_point(rng)
