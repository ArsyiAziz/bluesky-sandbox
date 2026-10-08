"""Geometric transforms for airspace-configuration randomization.

The designer can randomize the *whole design as a group* per episode - e.g.
rotate the airspace, its queryable regions/waypoints, and its spawn regions
together about a pivot by an angle drawn from a distribution. This is applied at
``Scenario.sample`` time, so each episode sees a transformed copy while the
schema-stable ``support`` episode stays in the canonical (unrotated) frame.

Rotation is computed in a local tangent plane about the pivot (via
:class:`LocalFrame`), so it is accurate over airspace-sized regions. A footprint
keeps its primitive through it - a circle stays a circle, a sector a sector at
its turned bearing (``Footprint.mapped``) - so the episode's geometry is what
the design draws; the design document is untouched.
"""

from __future__ import annotations

import math
from dataclasses import replace

from bluesky_sandbox.sim.bounds import (
    ConstantAltitudeBand,
    LatLon,
    LinearAltitudeBand,
    LocalFrame,
    RadialAltitudeBand,
    RegionBounds,
    VertexAltitudeBand,
)
from bluesky_sandbox.sim.bounds.base import AltitudeBand, Bounds
from bluesky_sandbox.sim.queryables import Queryable, QueryRegion, Waypoint
from bluesky_sandbox.sim.sampling.distributions import (
    sample_scalar,  # re-export: it lived here historically
)
from bluesky_sandbox.sim.spawn import SpawnConfig

# A point map takes (lat_deg, lon_deg) and returns the transformed (lat, lon).
PointMap = "callable"


def rotator(pivot: tuple[float, float], angle_deg: float):
    """Return a point map rotating about ``pivot`` by ``angle_deg``.

    Positive angle is counter-clockwise in the local east/north plane.
    """
    frame = LocalFrame(LatLon(pivot[0], pivot[1]))
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    def rotate(lat_deg: float, lon_deg: float) -> tuple[float, float]:
        x, y = frame.to_xy_nm(LatLon(lat_deg, lon_deg))  # x=east, y=north (nm)
        xr = x * cos_t - y * sin_t
        yr = x * sin_t + y * cos_t
        p = frame.from_xy_nm(xr, yr)
        return p.lat_deg, p.lon_deg

    return rotate


def translator(east_nm: float, north_nm: float):
    """Return a point map shifting every point by ``east_nm`` / ``north_nm``.

    Translation is pivot-independent: each point moves the same local
    east/north offset (degrees scaled by latitude so the metric offset holds).
    """

    def translate(lat_deg: float, lon_deg: float) -> tuple[float, float]:
        dlat = north_nm / 60.0
        cos_lat = max(0.01, math.cos(math.radians(lat_deg)))
        dlon = east_nm / (60.0 * cos_lat)
        return lat_deg + dlat, lon_deg + dlon

    return translate


def scaler(pivot: tuple[float, float], factor: float):
    """Return a point map scaling distance from ``pivot`` by ``factor``."""
    frame = LocalFrame(LatLon(pivot[0], pivot[1]))

    def scale(lat_deg: float, lon_deg: float) -> tuple[float, float]:
        x, y = frame.to_xy_nm(LatLon(lat_deg, lon_deg))
        p = frame.from_xy_nm(x * factor, y * factor)
        return p.lat_deg, p.lon_deg

    return scale


def compose(*maps):
    """Compose point maps, applied left-to-right (inner-most first)."""
    fns = [m for m in maps if m is not None]

    def mapped(lat_deg: float, lon_deg: float) -> tuple[float, float]:
        for fn in fns:
            lat_deg, lon_deg = fn(lat_deg, lon_deg)
        return lat_deg, lon_deg

    return mapped


def _map_altitude(band: AltitudeBand | None, point) -> AltitudeBand | None:
    if band is None or isinstance(band, ConstantAltitudeBand):
        return band
    if isinstance(band, LinearAltitudeBand):
        s = point(band.start.lat_deg, band.start.lon_deg)
        e = point(band.end.lat_deg, band.end.lon_deg)
        return LinearAltitudeBand(LatLon(*s), LatLon(*e), band.start_band_ft, band.end_band_ft)
    if isinstance(band, RadialAltitudeBand):
        c = point(band.center.lat_deg, band.center.lon_deg)
        return RadialAltitudeBand(LatLon(*c), band.radius_nm, band.inner_band_ft, band.outer_band_ft)
    if isinstance(band, VertexAltitudeBand):
        verts = [point(lat, lon) for lat, lon in band.vertices]
        return VertexAltitudeBand(verts, band.min_values_ft, band.max_values_ft)
    return band


def transform_bounds(bounds: Bounds, point) -> Bounds:
    """Apply a point map to a :class:`RegionBounds` (footprint + altitude anchors).

    The footprint keeps its primitive (``Footprint.mapped``); the turn the map
    gives is added to the region's orientation, which a shape with no axis of
    its own is oriented by.
    """
    if not isinstance(bounds, RegionBounds):
        return bounds
    center = bounds.center
    moved = LatLon(*point(center.lat_deg, center.lon_deg))
    north = LocalFrame(center).offset(0.0, 1.0)
    x, y = LocalFrame(moved).to_xy_nm(LatLon(*point(north.lat_deg, north.lon_deg)))
    out = RegionBounds(bounds.footprint.mapped(point), _map_altitude(bounds.altitude, point))
    out._orientation_deg = (
        getattr(bounds, "_orientation_deg", 0.0) + math.degrees(math.atan2(x, y))
    ) % 360.0
    return out


def transform_queryable(q: Queryable, point) -> Queryable:
    """Apply a point map to a queryable (a region's shape or a waypoint's position)."""
    if isinstance(q, QueryRegion):
        return QueryRegion(
            transform_bounds(q.shape, point),
            color=q.color,
            render_shape=q.render_shape,
            render_label=q.render_label,
            track_temporal_state=q.track_temporal_state,
        )
    if isinstance(q, Waypoint):
        if q.at is not None:
            # Its point, carried: the waypoint is where the point goes (and
            # keeps a fix's name only where the point stays on it).
            at = q.at
            if isinstance(at, LatLon):
                moved = LatLon(*point(at.lat_deg, at.lon_deg))
            elif isinstance(at, Bounds):
                moved = transform_bounds(at, point)
            else:
                moved = at.mapped(point)
            return replace(q, at=moved)
        lat, lon = point(q.lat, q.lon)
        return replace(q, lat=lat, lon=lon, waypoint=None)
    return q


def rotate_bounds(bounds: Bounds, pivot: tuple[float, float], angle_deg: float) -> Bounds:
    """Rotate a :class:`RegionBounds` (footprint + altitude anchors) about a pivot."""
    return transform_bounds(bounds, rotator(pivot, angle_deg))


def rotate_queryable(q: Queryable, pivot: tuple[float, float], angle_deg: float) -> Queryable:
    return transform_queryable(q, rotator(pivot, angle_deg))


def rotate_spawn(spawn: SpawnConfig, pivot: tuple[float, float], angle_deg: float) -> SpawnConfig:
    def rotate_route_step(step):
        if not isinstance(step, dict):
            return step
        out = dict(step)
        sample = out.get("sample")
        if isinstance(sample, Bounds):
            out["sample"] = rotate_bounds(sample, pivot, angle_deg)
        if "choice" in out:
            out["choice"] = [
                [rotate_route_step(s) for s in branch]
                if isinstance(branch, list)
                else rotate_route_step(branch)
                for branch in out["choice"]
            ]
        return out

    def rotate_route(route):
        if not isinstance(route, list):
            return route
        return [rotate_route_step(step) for step in route]

    # dataclasses.replace, NOT field-by-field reconstruction: only the
    # transformed fields change, everything else (conflict_free_* flags,
    # maintain/controlled, and any future field) carries over.
    # The old explicit constructors silently dropped conflict-free spawning
    # from every rotated episode.
    regions = [
        replace(r, shape=rotate_bounds(r.shape, pivot, angle_deg), route=rotate_route(r.route))
        for r in spawn.regions
    ]
    return replace(
        spawn,
        regions=regions,
        route=rotate_route(spawn.route),
        routes={name: rotate_route(route) for name, route in spawn.routes.items()},
    )


def bbox_center(bounds: Bounds) -> tuple[float, float]:
    (lat_min, lat_max), (lon_min, lon_max) = bounds.bounding_box
    return ((lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0)
