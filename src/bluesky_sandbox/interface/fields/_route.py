"""Reading an aircraft's BlueSky route.

Shared by the active-route-waypoint observations and the route-relative actions,
so both read the same target.
"""

from __future__ import annotations

from typing import Any

import bluesky as bs
import numpy as np
from bluesky.tools.geo import kwikqdrdist


def _route_constraint(values: Any, iact: int) -> float | None:
    """Read a per-waypoint constraint (e.g. ``wpalt``/``wpspd``) at ``iact``.

    Returns ``None`` for a missing or unspecified constraint; BlueSky stores
    "not specified" as a negative sentinel.
    """
    try:
        value = float(values[iact])
    except (IndexError, TypeError, ValueError):
        return None
    if np.isfinite(value) and value >= 0.0:
        return value
    return None


def _active_route_waypoint(
    idx: int,
    offset: int = 0,
) -> tuple[float, float, float | None, float | None] | None:
    """Return ``(lat, lon, alt_m, spd_ms)`` for a route fix relative to the
    aircraft's active leg, or ``None`` when there is no usable fix there.

    ``offset`` selects which fix: ``0`` (default) is the active leg the
    autopilot is currently flying to; ``1`` is the *next* leg (e.g. the exit
    fix while still working a merge fix), ``2`` the one after, etc. ``None``
    whenever ``iactwp + offset`` falls outside the route (no next leg on a
    single-leg route, or beyond the final fix) - the same "no usable waypoint"
    convention offset ``0`` already used for a routeless aircraft.

    ``alt_m`` and ``spd_ms`` are ``None`` when the waypoint carries no altitude
    or speed constraint. Shared by the active-route-waypoint observation fields
    and the deviation-from-nominal action fields (which always use ``offset=0``
    - actions command the leg actually being flown, never a future one) so
    both read the same target.
    """
    routes = getattr(bs.traf.ap, "route", None) if bs.traf is not None else None
    if routes is None or idx < 0 or idx >= len(routes):
        return None
    route = routes[idx]
    try:
        iact = int(route.iactwp)
        nwp = int(route.nwp or 0)
    except (TypeError, ValueError):
        return None
    if iact < 0:
        return None
    target = iact + offset
    if target < 0 or target >= nwp:
        return None
    try:
        lat = float(route.wplat[target])
        lon = float(route.wplon[target])
    except (IndexError, TypeError, ValueError):
        return None
    if not (np.isfinite(lat) and np.isfinite(lon)):
        return None
    return (
        lat,
        lon,
        _route_constraint(route.wpalt, target),
        _route_constraint(route.wpspd, target),
    )


def _route_along_distance_nm(idx: int, offset: int) -> float | None:
    """Distance from the aircraft to the route fix at ``iactwp + offset``, nm,
    measured ALONG the remaining legs.

    Direct great-circle range to the active fix, then leg by leg out to the
    target fix - the distance actually flown, not the straight line to a fix
    two legs ahead. Reduces to the plain range at ``offset=0``, where it equals
    :class:`ActiveRouteWaypointDistanceNm`. ``None`` whenever there is no
    usable fix at that offset, mirroring :func:`_active_route_waypoint`'s
    validation and convention.
    """
    routes = getattr(bs.traf.ap, "route", None) if bs.traf is not None else None
    if routes is None or idx < 0 or idx >= len(routes):
        return None
    route = routes[idx]
    try:
        iact = int(route.iactwp)
        nwp = int(route.nwp or 0)
    except (TypeError, ValueError):
        return None
    target = iact + offset
    if iact < 0 or target < 0 or target >= nwp:
        return None
    lat = float(bs.traf.lat[idx])
    lon = float(bs.traf.lon[idx])
    total_nm = 0.0
    for leg in range(iact, target + 1):
        try:
            wp_lat = float(route.wplat[leg])
            wp_lon = float(route.wplon[leg])
        except (IndexError, TypeError, ValueError):
            return None
        if not (np.isfinite(wp_lat) and np.isfinite(wp_lon)):
            return None
        _qdr, dist = kwikqdrdist(lat, lon, wp_lat, wp_lon)
        total_nm += float(dist)
        lat, lon = wp_lat, wp_lon
    return total_nm
