"""Vectorized containment over whole traffic arrays.

The :class:`~.base.Footprint` API is polymorphic and scalar: every shape answers
``contains(lat, lon, alt)`` for one point. That is the right contract for the
shapes themselves - a new footprint implements one method - but a caller testing
every live aircraft each step pays a Python call per aircraft per region.

:func:`contains_many` is the fast path for the two shapes that dominate
(:class:`~.footprints.BoxFootprint`, :class:`~.footprints.DiskFootprint`) and
returns ``None`` for anything else, so callers fall back to the scalar API
rather than every footprint class growing a vectorized twin.
"""

from __future__ import annotations

import numpy as np

from .altitude import ConstantAltitudeBand
from .base import Bounds, RegionBounds
from .footprints import BoxFootprint, DiskFootprint

__all__ = ["contains_many"]


def contains_many(
    bounds: Bounds,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    alt_ft: np.ndarray,
) -> np.ndarray | None:
    """Boolean mask of which points fall inside ``bounds``.

    Returns ``None`` for uncommon bounds so callers can fall back to the
    polymorphic scalar API without complicating every footprint class.
    """
    if not isinstance(bounds, RegionBounds):
        return None

    footprint = bounds.footprint
    altitude = bounds.altitude
    if isinstance(footprint, BoxFootprint):
        mask = (
            (footprint.lat_min_deg < lat_deg)
            & (lat_deg < footprint.lat_max_deg)
            & (footprint.lon_min_deg < lon_deg)
            & (lon_deg < footprint.lon_max_deg)
        )
    elif isinstance(footprint, DiskFootprint):
        (lat_min, lat_max), (lon_min, lon_max) = footprint.bounding_box
        mask = (
            (lat_min <= lat_deg)
            & (lat_deg <= lat_max)
            & (lon_min <= lon_deg)
            & (lon_deg <= lon_max)
        )
        x_nm = (lon_deg - footprint.center.lon_deg) * 60.0 * footprint._frame._cos_lat
        y_nm = (lat_deg - footprint.center.lat_deg) * 60.0
        mask &= (x_nm * x_nm + y_nm * y_nm) <= footprint.radius_nm * footprint.radius_nm
    else:
        return None

    mask &= (bounds.alt_min_ft <= alt_ft) & (alt_ft <= bounds.alt_max_ft)
    if isinstance(altitude, ConstantAltitudeBand):
        return mask

    # A non-constant band varies with position, so only the rows that passed the
    # cheap box/disk test are worth the per-point call.
    candidate_rows = np.flatnonzero(mask)
    for row in candidate_rows:
        if not altitude.contains(
            float(lat_deg[row]),
            float(lon_deg[row]),
            float(alt_ft[row]),
        ):
            mask[row] = False
    return mask
