"""Reading an aircraft's BlueSky route.

Shared by the active-route-waypoint observations and the route-relative actions,
so both read the same target.
"""

from __future__ import annotations

from typing import Any

import bluesky as bs
import numpy as np
from bluesky.tools.aero import nm, vtas2cas
from bluesky.tools.geo import kwikqdrdist

from ._state import arrival_time


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


def _route_index(idx: int, offset: int = 0) -> int | None:
    """The route index of the fix ``offset`` legs from the active one, or
    ``None`` when there is none there - as :func:`_active_route_waypoint`."""
    routes = getattr(bs.traf.ap, "route", None) if bs.traf is not None else None
    if routes is None or idx < 0 or idx >= len(routes):
        return None
    try:
        iact = int(routes[idx].iactwp)
        nwp = int(routes[idx].nwp or 0)
    except (TypeError, ValueError):
        return None
    target = iact + offset
    return target if iact >= 0 and 0 <= target < nwp else None


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
    or speed constraint. A target arrival time is a speed constraint too, only
    a derived one: where the fix has no speed gate but a time to be over it,
    ``spd_ms`` is the CAS that still meets that time from where the aircraft is
    (:func:`_scheduled_cas_ms`). So every reader of a fix's speed - the
    route-relative speed actions, whose zero is then "on schedule", and the
    speed observations - agrees on it. Shared by the active-route-waypoint observation fields
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
    speed = _route_constraint(route.wpspd, target)
    if speed is None:
        speed = _scheduled_cas_ms(idx, target, offset)
    return (lat, lon, _route_constraint(route.wpalt, target), speed)


def _scheduled_cas_ms(idx: int, route_index: int, offset: int) -> float | None:
    """The CAS that gets the aircraft at ``idx`` over route fix ``route_index``
    at its target arrival time, or ``None`` when it has none: the along-route
    distance left over the time to go, in the wind the aircraft flies in now.
    Overdue, its maximum. Late, that is faster; early, slower - and below its
    minimum speed, only a longer path loses the rest."""
    due = arrival_time(idx, route_index)
    dist_nm = None if due is None else _route_along_distance_nm(idx, offset)
    if dist_nm is None:
        return None
    time_to_go = due - float(bs.sim.simt)
    if time_to_go <= 0.0:
        return float(bs.traf.perf.vmax[idx])
    gs_needed = dist_nm * nm / time_to_go
    tas_needed = gs_needed + float(bs.traf.tas[idx]) - float(bs.traf.gs[idx])
    return float(vtas2cas(max(tas_needed, 0.0), float(bs.traf.alt[idx])))


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
