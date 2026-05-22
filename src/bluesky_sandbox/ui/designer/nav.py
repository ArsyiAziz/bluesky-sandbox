"""Navigation-data query layer backing the map tab.

A thin, read-only wrapper over BlueSky's ``navdb`` that returns plain
dataclasses (JSON-able via :func:`dataclasses.asdict`) instead of raw parallel
arrays. The airspace bounding box is the natural viewport: the global navdb has
~136k waypoints and ~14k airports, far too many to ship to a client, so every
query is scoped to a lat/lon window (typically an airspace's ``bounding_box``).

The navdb is primed without a full ``bs.init`` (reusing
``queryables._ensure_navdb_loaded``), so the map tab can resolve fixes and
airports before any environment is constructed.

This is the resolver half of the "reference by identifier, resolve at build"
principle: the spec stores ``"EKROS"`` / ``"EHAM"``; this module turns the
viewport into concrete features and resolves a single identifier to coords.
"""

from __future__ import annotations

from dataclasses import dataclass

import bluesky as bs

from bluesky_sandbox.sim.bounds import Bounds
from bluesky_sandbox.sim.queryables import _ensure_navdb_loaded


# --------------------------------------------------------------------------- #
# Result types                                                                #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NavWaypoint:
    ident: str
    lat_deg: float
    lon_deg: float
    wptype: str | None = None
    desc: str | None = None


@dataclass(frozen=True)
class NavRunwayThreshold:
    name: str
    lat_deg: float
    lon_deg: float


@dataclass(frozen=True)
class NavAirport:
    icao: str
    lat_deg: float
    lon_deg: float
    name: str | None = None
    elev_ft: float | None = None
    runways: tuple[NavRunwayThreshold, ...] = ()


@dataclass(frozen=True)
class NavAirwayLeg:
    """One airway segment connecting two fixes (BlueSky's ``aw*`` arrays)."""

    awid: str
    from_id: str
    to_id: str
    from_lat_deg: float
    from_lon_deg: float
    to_lat_deg: float
    to_lon_deg: float


# --------------------------------------------------------------------------- #
# Viewport helpers                                                            #
# --------------------------------------------------------------------------- #
LatLonWindow = tuple[float, float, float, float]  # (lat_min, lat_max, lon_min, lon_max)


def window_from_bounds(bounds: Bounds, margin_frac: float = 0.0) -> LatLonWindow:
    """Return a ``(lat_min, lat_max, lon_min, lon_max)`` window for a bounds.

    ``margin_frac`` pads each side by a fraction of the box span, so the map can
    fetch a little context around the airspace edge.
    """
    (lat_min, lat_max), (lon_min, lon_max) = bounds.bounding_box
    if margin_frac:
        dlat = (lat_max - lat_min) * margin_frac
        dlon = (lon_max - lon_min) * margin_frac
        lat_min, lat_max = lat_min - dlat, lat_max + dlat
        lon_min, lon_max = lon_min - dlon, lon_max + dlon
    return (float(lat_min), float(lat_max), float(lon_min), float(lon_max))


def _in_window(lat: float, lon: float, w: LatLonWindow) -> bool:
    return w[0] <= lat <= w[1] and w[2] <= lon <= w[3]


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


# --------------------------------------------------------------------------- #
# Queries                                                                     #
# --------------------------------------------------------------------------- #
def waypoints_in_window(
    window: LatLonWindow,
    *,
    limit: int | None = 2000,
) -> list[NavWaypoint]:
    """Return navdb waypoints whose position lies inside ``window``.

    ``limit`` caps the result (the map only needs what it can draw); ``None``
    returns everything in the window.
    """
    _ensure_navdb_loaded()
    nd = bs.navdb
    lats = nd.wplat
    lons = nd.wplon
    out: list[NavWaypoint] = []
    for i in range(len(lats)):
        lat = float(lats[i])
        lon = float(lons[i])
        if not _in_window(lat, lon, window):
            continue
        out.append(
            NavWaypoint(
                ident=_decode(nd.wpid[i]),
                lat_deg=lat,
                lon_deg=lon,
                wptype=_decode(nd.wptype[i]) if nd.wptype else None,
                desc=_decode(nd.wpdesc[i]) if nd.wpdesc else None,
            )
        )
        if limit is not None and len(out) >= limit:
            break
    return out


def airports_in_window(
    window: LatLonWindow,
    *,
    limit: int | None = 500,
    with_runways: bool = True,
) -> list[NavAirport]:
    """Return navdb airports inside ``window``, optionally with runway thresholds."""
    _ensure_navdb_loaded()
    nd = bs.navdb
    lats = nd.aptlat
    lons = nd.aptlon
    out: list[NavAirport] = []
    for i in range(len(lats)):
        lat = float(lats[i])
        lon = float(lons[i])
        if not _in_window(lat, lon, window):
            continue
        icao = _decode(nd.aptid[i])
        out.append(
            NavAirport(
                icao=icao,
                lat_deg=lat,
                lon_deg=lon,
                name=_decode(nd.aptname[i]) if nd.aptname else None,
                elev_ft=float(nd.aptelev[i]) if nd.aptelev is not None else None,
                runways=tuple(runway_thresholds(icao)) if with_runways else (),
            )
        )
        if limit is not None and len(out) >= limit:
            break
    return out


def airways_in_window(
    window: LatLonWindow,
    *,
    limit: int | None = 4000,
) -> list[NavAirwayLeg]:
    """Return airway legs with at least one endpoint inside ``window``.

    BlueSky stores airways as parallel from/to arrays (one entry per leg). A leg
    is kept if either endpoint is in view, so segments crossing the window edge
    still render. ``limit`` caps the result (airways are dense); ``None`` returns
    everything in the window.
    """
    _ensure_navdb_loaded()
    nd = bs.navdb
    awid = getattr(nd, "awid", None)
    if not awid:
        return []
    fromlat, fromlon = nd.awfromlat, nd.awfromlon
    tolat, tolon = nd.awtolat, nd.awtolon
    fromid, toid = nd.awfromwpid, nd.awtowpid
    out: list[NavAirwayLeg] = []
    for i in range(len(awid)):
        flat, flon = float(fromlat[i]), float(fromlon[i])
        tlat, tlon = float(tolat[i]), float(tolon[i])
        if not (_in_window(flat, flon, window) or _in_window(tlat, tlon, window)):
            continue
        out.append(
            NavAirwayLeg(
                awid=_decode(awid[i]),
                from_id=_decode(fromid[i]),
                to_id=_decode(toid[i]),
                from_lat_deg=flat,
                from_lon_deg=flon,
                to_lat_deg=tlat,
                to_lon_deg=tlon,
            )
        )
        if limit is not None and len(out) >= limit:
            break
    return out


def runway_thresholds(icao: str) -> list[NavRunwayThreshold]:
    """Return runway-threshold points for an airport, or ``[]`` if unknown."""
    _ensure_navdb_loaded()
    nd = bs.navdb
    table = getattr(nd, "rwythresholds", None)
    if not table:
        return []
    entry = table.get(icao.upper())
    if not entry:
        return []
    out: list[NavRunwayThreshold] = []
    for name, value in entry.items():
        # BlueSky stores (lat, lon, heading) per threshold.
        try:
            lat, lon = float(value[0]), float(value[1])
        except (TypeError, ValueError, IndexError):
            continue
        out.append(NavRunwayThreshold(name=str(name), lat_deg=lat, lon_deg=lon))
    return out


def resolve_waypoint(ident: str) -> NavWaypoint:
    """Resolve a single waypoint identifier to coordinates.

    Raises ``ValueError`` when the identifier is unknown - the same loud-failure
    contract as :class:`~bluesky_sandbox.sim.queryables.Waypoint`, so a stale spec
    reference surfaces clearly rather than silently drifting.
    """
    _ensure_navdb_loaded()
    nd = bs.navdb
    idx = nd.getwpidx(ident)
    if idx < 0:
        raise ValueError(f"waypoint {ident!r} not found in navdb.")
    return NavWaypoint(
        ident=ident.upper(),
        lat_deg=float(nd.wplat[idx]),
        lon_deg=float(nd.wplon[idx]),
        wptype=_decode(nd.wptype[idx]) if nd.wptype else None,
        desc=_decode(nd.wpdesc[idx]) if nd.wpdesc else None,
    )


def resolve_airport(icao: str) -> NavAirport:
    """Resolve an airport ICAO code to its position, name, and runways."""
    _ensure_navdb_loaded()
    nd = bs.navdb
    idx = nd.getaptidx(icao)
    if idx < 0:
        raise ValueError(f"airport {icao!r} not found in navdb.")
    return NavAirport(
        icao=icao.upper(),
        lat_deg=float(nd.aptlat[idx]),
        lon_deg=float(nd.aptlon[idx]),
        name=_decode(nd.aptname[idx]) if nd.aptname else None,
        elev_ft=float(nd.aptelev[idx]) if nd.aptelev is not None else None,
        runways=tuple(runway_thresholds(icao)),
    )


def search(
    query: str,
    *,
    kinds: tuple[str, ...] = ("waypoint", "airport"),
    limit: int = 30,
) -> dict[str, list]:
    """Search navdb by identifier (and airport name) for the global picker.

    Unlike :func:`features_in_bounds`, this is not window-scoped - it powers the
    "fly to / add a feature" search box. Matching is case-insensitive: exact and
    prefix hits rank ahead of substring hits, capped at ``limit`` per kind.
    """
    q = query.strip().upper()
    out: dict[str, list] = {"waypoints": [], "airports": []}
    if not q:
        return out
    _ensure_navdb_loaded()
    nd = bs.navdb

    if "waypoint" in kinds:
        out["waypoints"] = [
            NavWaypoint(
                ident=_decode(nd.wpid[i]),
                lat_deg=float(nd.wplat[i]),
                lon_deg=float(nd.wplon[i]),
                wptype=_decode(nd.wptype[i]) if nd.wptype else None,
            )
            for i in _rank_matches((_decode(x) for x in nd.wpid), q, limit)
        ]

    if "airport" in kinds:
        names = nd.aptname or []
        idents = [_decode(x) for x in nd.aptid]
        # match against ICAO or airport name
        haystack = [
            f"{idents[i]} {_decode(names[i]) if i < len(names) else ''}"
            for i in range(len(idents))
        ]
        out["airports"] = [
            NavAirport(
                icao=idents[i],
                lat_deg=float(nd.aptlat[i]),
                lon_deg=float(nd.aptlon[i]),
                name=_decode(names[i]) if i < len(names) else None,
            )
            for i in _rank_matches(iter(haystack), q, limit)
        ]
    return out


def _rank_matches(values, q: str, limit: int) -> list[int]:
    """Return indices whose upper-cased value matches ``q``, prefix hits first."""
    prefix: list[int] = []
    substring: list[int] = []
    for i, value in enumerate(values):
        v = value.upper()
        if v.startswith(q):
            prefix.append(i)
        elif q in v:
            substring.append(i)
        if len(prefix) >= limit:
            break
    return (prefix + substring)[:limit]


def features_in_bounds(
    bounds: Bounds,
    *,
    margin_frac: float = 0.1,
    waypoint_limit: int | None = 2000,
    airport_limit: int | None = 500,
    airway_limit: int | None = 4000,
) -> dict[str, list]:
    """One-shot map payload: waypoints + airports + airways within the window.

    Returns ``{"window": [...], "waypoints": [...], "airports": [...],
    "airways": [...]}`` ready to serialise for the map tab.
    """
    window = window_from_bounds(bounds, margin_frac=margin_frac)
    return {
        "window": list(window),
        "waypoints": waypoints_in_window(window, limit=waypoint_limit),
        "airports": airports_in_window(window, limit=airport_limit),
        "airways": airways_in_window(window, limit=airway_limit),
    }


__all__ = [
    "LatLonWindow",
    "NavAirport",
    "NavAirwayLeg",
    "NavRunwayThreshold",
    "NavWaypoint",
    "airports_in_window",
    "airways_in_window",
    "features_in_bounds",
    "resolve_airport",
    "resolve_waypoint",
    "runway_thresholds",
    "search",
    "waypoints_in_window",
    "window_from_bounds",
]
