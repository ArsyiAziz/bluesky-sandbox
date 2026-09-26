"""Render-preview extraction: a spec -> JSON-able geometry for the map tab.

Turns the materialized resources of a :class:`DesignSpec` into plain polygons,
points, and sampled aircraft so the map can draw an environment without running
the simulator. Reuses the primitives' own ``vertices`` / ``bounding_box`` /
``sample_point`` accessors, so what the map shows is exactly what the env builds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from bluesky.tools.geo import qdrpos

from bluesky_sandbox.sim.bounds import Bounds
from bluesky_sandbox.sim.queryables import QueryRegion, Waypoint
from bluesky_sandbox.sim.scenario.geometry_json import (
    bounds_geometry,
    queryable_geometry,
    spawn_region_geometry,
)
from bluesky_sandbox.sim.sampling.distributions import Categorical
from bluesky_sandbox.sim.scenario import transforms as _t
from bluesky_sandbox.sim.spawn import SpawnConfig

from .builder import build_scenario
from .spec import DesignSpec


def _resolve_spawn_types(spawn: SpawnConfig, allowed_aircraft: list[str]) -> None:
    """Fill in ``aircraft_type`` the way the env would, for standalone preview.

    ``iter_spawns`` needs a non-``None`` global ``aircraft_type``; the env
    resolves it every ``reset()`` via
    :func:`~bluesky_sandbox.config.resolve_spawn_aircraft_types`. Preview builds
    the scenario without an ``EnvConfig`` (so the map renders even when code
    refs are incomplete), so we replicate just that resolution here.
    """
    allowed = [a.upper() for a in allowed_aircraft] or ["B744"]

    def resolve(t):
        if isinstance(t, Categorical):
            return t
        if isinstance(t, str):
            return Categorical({t.upper(): 1.0})
        return Categorical({a: 1.0 for a in allowed})

    spawn.aircraft_type = resolve(spawn.aircraft_type)
    for region in spawn.regions:
        if region.aircraft_type is not None:
            region.aircraft_type = resolve(region.aircraft_type)


def scenario_preview(spec: DesignSpec, *, seed: int = 0) -> dict[str, Any]:
    """Return renderable geometry for a spec: airspace, queryables, spawn + samples.

    A single seeded ``iter_spawns`` draw gives a representative set of aircraft
    so the map can show where traffic appears, without stepping the simulator.
    """
    scenario = build_scenario(spec)
    rng = np.random.default_rng(seed)
    # Sample (not support) so per-episode randomization - rotation and, below,
    # spawn locations - is what the map shows; reseeding varies it.
    episode = scenario.sample(rng)
    _resolve_spawn_types(episode.spawn, spec.env.allowed_aircraft)

    airspace = (
        bounds_geometry(episode.airspace_bounds)
        if episode.airspace_bounds is not None
        else None
    )

    queryables = [
        queryable_geometry(
            name,
            q,
            per_aircraft=(
                isinstance(spec.queryables.get(name), dict)
                and spec.queryables[name].get("sample_per") == "aircraft"
            ),
        )
        for name, q in episode.queryables.items()
    ]
    # A sampled waypoint whose altitude resolves per aircraft (envelope) has no
    # single alt_ft - which would strand its marker at ground level, invisible.
    # Give the map a representative display altitude: the midpoint of its
    # sample region's altitude band (the band the per-aircraft draws land in).
    for entry in queryables:
        if entry.get("kind") != "waypoint" or entry.get("alt_ft") is not None:
            continue
        region = scenario.sampled_waypoints.get(entry["name"])
        lo = getattr(region, "alt_min_ft", None)
        hi = getattr(region, "alt_max_ft", None)
        if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
            entry["display_alt_ft"] = float(lo + hi) / 2.0

    spawn_regions = [
        spawn_region_geometry(i, region) for i, region in enumerate(episode.spawn.regions)
    ]

    def _target(route, key: str) -> dict[str, Any] | None:
        """Resolve the aircraft's final route waypoint (its goal) to a point.

        A per-aircraft sampled waypoint is drawn for *this* aircraft so the map
        shows the per-aircraft spread (reseed to vary).
        """
        if not route:
            return None
        last = route[-1]
        step = last if isinstance(last, dict) else {"waypoint": last}
        name = step.get("waypoint")
        wp = episode.queryables.get(name) if isinstance(name, str) else None
        if wp is None:
            return None
        sample_bounds = step.get("sample")
        if sample_bounds is not None:
            lat, lon = sample_bounds.sample_point(rng)
            alt = step.get("alt_ft", getattr(wp, "alt_ft", None))
            band = getattr(sample_bounds, "alt_band_at", None)
            if band is not None and alt is not None:
                lo, hi = band(lat, lon)
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    alt = float(rng.uniform(lo, hi))
        else:
            lat, lon = getattr(wp, "lat", None), getattr(wp, "lon", None)
            alt = step.get("alt_ft", getattr(wp, "alt_ft", None))
        if lat is None or lon is None:
            return None
        # Presentation fields from the waypoint template so the map can draw
        # the per-aircraft target as a real waypoint (reach-radius tolerance
        # disc in its color), not just a screen-space dot.
        return {
            "lat": float(lat),
            "lon": float(lon),
            "alt_ft": None if alt is None else float(alt),
            "name": name,
            "color": getattr(wp, "color", None),
            "reach_radius_nm": getattr(wp, "reach_radius_nm", None),
            "alt_tolerance_ft": getattr(wp, "alt_tolerance_ft", None),
            "speed_tolerance_kts": getattr(wp, "speed_tolerance_kts", None),
        }

    sampled_aircraft = []
    for i, (_region_index, spawn_time, actype, pos, prefix, route) in enumerate(
        episode.spawn.iter_spawns(
            rng, limit=episode.max_aircraft, include_maintain=True
        )
    ):
        sampled_aircraft.append(
            {
                "lat": float(pos["lat_deg"]),
                "lon": float(pos["lon_deg"]),
                "alt_ft": float(pos.get("alt_ft", float("nan"))),
                "spd_kts": float(pos.get("spd_kts", float("nan"))),
                "actype": actype,
                "spawn_time": float(spawn_time),
                "callsign_prefix": prefix,
                "route": list(route) if route else None,
                "target": _target(route, f"_pv{i}"),
            }
        )

    # Named regions in the *episode* frame: the sampled shape draw (via the
    # builder's region sink) carried into the episode's transform - the
    # whole-geometry rotation, or the region's group chain under groups. This
    # is what lets the map render e.g. a sampled exit corridor at its drawn
    # width and bearing, instead of the canonical unrotated design shape.
    regions: dict[str, Any] = {}
    sink = getattr(scenario, "design_regions", None)
    rot = getattr(scenario, "last_rotation", None)
    group_maps = getattr(scenario, "last_group_maps", None)
    group_chains = getattr(scenario, "design_region_group_chains", None) or {}
    if isinstance(sink, dict):
        for name, bounds in sink.items():
            episode_bounds = bounds
            if rot:
                episode_bounds = _t.rotate_bounds(bounds, rot["pivot"], rot["angle"])
            elif group_maps and group_chains.get(name):
                chain = [group_maps[g] for g in group_chains[name] if g in group_maps]
                if chain:
                    episode_bounds = _t.transform_bounds(bounds, _t.compose(*chain))
            regions[name] = {"name": name, **bounds_geometry(episode_bounds)}

    return {
        "airspace": airspace,
        "queryables": queryables,
        "spawn_regions": spawn_regions,
        "regions": regions,
        "sampled_aircraft": sampled_aircraft,
        "max_aircraft": int(episode.max_aircraft),
        "seed": seed,
        "airspace_warnings": airspace_warnings(episode),
    }


def airspace_warnings(episode) -> list[str]:
    """Flag query/spawn regions or waypoints not subsumed by the airspace.

    The airspace is meant to enclose the whole design (it is the operational
    boundary and the observation-normalization range). Any content whose
    footprint or finite altitude extent falls outside the airspace is reported
    by name so the user can enlarge the airspace or move the content in.
    """
    asp = episode.airspace_bounds
    if asp is None:
        return []
    out: list[str] = []

    # Content that reuses the airspace bounds sits exactly on its edge, where
    # contains() is strict (boundary excluded). Nudge each tested point a hair
    # toward the airspace center so coincident/shared geometry counts as
    # subsumed, while genuinely-outside points stay flagged.
    (a_lat_min, a_lat_max), (a_lon_min, a_lon_max) = asp.bounding_box
    _cen_lat = (a_lat_min + a_lat_max) / 2.0
    _cen_lon = (a_lon_min + a_lon_max) / 2.0
    _EDGE_EPS = 1e-6

    def toward_center(lat: float, lon: float) -> tuple[float, float]:
        return lat + (_cen_lat - lat) * _EDGE_EPS, lon + (_cen_lon - lon) * _EDGE_EPS

    def finite(value: float | None) -> bool:
        return value is not None and np.isfinite(float(value))

    def bounds_alt_samples(bounds: Bounds, lat: float, lon: float) -> tuple[float, ...]:
        try:
            lo, hi = bounds.alt_band_at(lat, lon)
        except Exception:
            lo = getattr(bounds, "alt_min_ft", None)
            hi = getattr(bounds, "alt_max_ft", None)
        return tuple(float(v) for v in (lo, hi) if finite(v))

    def point_outside(lat: float, lon: float, alt_samples: tuple[float, ...] = ()) -> bool:
        lat, lon = toward_center(lat, lon)
        if not asp.contains(lat, lon):
            return True
        return any(not asp.contains(lat, lon, alt) for alt in alt_samples)

    def bounds_outside(bounds: Bounds) -> bool:
        return any(
            point_outside(lat, lon, bounds_alt_samples(bounds, lat, lon))
            for lat, lon in bounds.vertices
        )

    def waypoint_points(q: Waypoint) -> list[tuple[float, float]]:
        points = [(float(q.lat), float(q.lon))]
        radius = q.reach_radius_nm
        if finite(radius) and float(radius) > 0.0:
            points.extend(
                (float(lat), float(lon))
                for lat, lon in (
                    qdrpos(float(q.lat), float(q.lon), bearing, float(radius))
                    for bearing in range(0, 360, 30)
                )
            )
        return points

    def waypoint_alt_samples(q: Waypoint) -> tuple[float, ...]:
        if not finite(q.alt_ft):
            return ()
        alt = float(q.alt_ft)
        tol = q.alt_tolerance_ft
        if finite(tol) and float(tol) > 0.0:
            return alt - float(tol), alt, alt + float(tol)
        return (alt,)

    for name, q in episode.queryables.items():
        if isinstance(q, QueryRegion):
            if bounds_outside(q.bounds):
                out.append(f"queryable '{name}'")
        elif isinstance(q, Waypoint):
            alt_samples = waypoint_alt_samples(q)
            if any(point_outside(lat, lon, alt_samples) for lat, lon in waypoint_points(q)):
                out.append(f"waypoint '{name}'")
    for i, region in enumerate(episode.spawn.regions):
        if bounds_outside(region.bounds):
            out.append(f"spawn '{region.name or i}'")
    return out
