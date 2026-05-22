"""Render-preview extraction: a spec -> JSON-able geometry for the map tab.

Turns the materialised resources of a :class:`DesignSpec` into plain polygons,
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
from bluesky_sandbox.sim.sampling.distributions import Categorical
from bluesky_sandbox.sim.spawn import SpawnConfig

from .builder import build_scenario
from .spec import DesignSpec


def _normalize_spawn_types(spawn: SpawnConfig, allowed_aircraft: list[str]) -> None:
    """Fill in ``aircraft_type`` the way ``EnvConfig`` would, for standalone preview.

    ``iter_spawns`` needs a non-``None`` global ``aircraft_type``; the env
    normally sets this in ``EnvConfig.__post_init__``. Preview builds the
    scenario without an ``EnvConfig`` (so the map renders even when code refs
    are incomplete), so we replicate just that normalisation here.
    """
    allowed = [a.upper() for a in allowed_aircraft] or ["B744"]

    def norm(t):
        if isinstance(t, Categorical):
            return t
        if isinstance(t, str):
            return Categorical({t.upper(): 1.0})
        return Categorical({a: 1.0 for a in allowed})

    spawn.aircraft_type = norm(spawn.aircraft_type)
    for region in spawn.regions:
        if region.aircraft_type is not None:
            region.aircraft_type = norm(region.aircraft_type)


def _bounds_geometry(bounds: Bounds) -> dict[str, Any]:
    (lat_min, lat_max), (lon_min, lon_max) = bounds.bounding_box
    out: dict[str, Any] = {
        "vertices": [[float(a), float(b)] for a, b in bounds.vertices],
        "bounding_box": {
            "lat_min": float(lat_min),
            "lat_max": float(lat_max),
            "lon_min": float(lon_min),
            "lon_max": float(lon_max),
        },
    }
    alt_min = getattr(bounds, "alt_min_ft", None)
    alt_max = getattr(bounds, "alt_max_ft", None)
    if alt_min is not None and alt_max is not None:
        out["alt_min_ft"] = None if alt_min == float("-inf") else float(alt_min)
        out["alt_max_ft"] = None if alt_max == float("inf") else float(alt_max)
    # Per-vertex altitude band (for varying bands: linear/radial/vertex), aligned
    # to `vertices`, so a 3D wireframe can slope its top/sides to follow the band.
    per_vertex = None
    try:
        per_vertex = bounds.per_vertex_alt_range()
    except Exception:
        per_vertex = None
    if per_vertex:
        out["per_vertex_alt_ft"] = [[float(lo), float(hi)] for lo, hi in per_vertex]
    return out


def _heading_range(hdg: Any) -> list[float] | None:
    """``[low, high]`` heading range for a spawn region, or ``None`` (uniform).

    A fixed scalar becomes a degenerate ``[h, h]``; a range passes through; a
    distribution uses its finite support when available, else ``None``.
    """
    if hdg is None:
        return None
    if isinstance(hdg, (int, float)):
        return [float(hdg), float(hdg)]
    if isinstance(hdg, tuple) and len(hdg) == 2:
        return [float(hdg[0]), float(hdg[1])]
    support = getattr(hdg, "support", None)
    if callable(support):
        try:
            lo, hi = support()
            if np.isfinite(lo) and np.isfinite(hi):
                return [float(lo), float(hi)]
        except Exception:
            pass
    return None


def _queryable_geometry(name: str, q: Any, *, per_aircraft: bool = False) -> dict[str, Any]:
    if isinstance(q, QueryRegion):
        return {
            "name": name,
            "kind": "region",
            "color": q.color,
            "render_shape": q.render_shape,
            "render_label": q.render_label,
            **_bounds_geometry(q.bounds),
        }
    if isinstance(q, Waypoint):
        return {
            "name": name,
            "kind": "waypoint",
            "color": q.color,
            "ident": q.waypoint,
            "lat": float(q.lat),
            "lon": float(q.lon),
            "alt_ft": q.alt_ft,
            "speed_kts": q.speed_kts,
            "reach_radius_nm": q.reach_radius_nm,
            "alt_tolerance_ft": q.alt_tolerance_ft,
            "speed_tolerance_kts": q.speed_tolerance_kts,
            "render_shape": q.render_shape,
            "render_label": q.render_label,
            # The runtime Waypoint no longer knows how it was sampled (the
            # builder moves per-aircraft sampling onto route steps), so the
            # flag comes from the spec. A per-aircraft waypoint's template
            # lat/lon is meaningless as a point - the map draws per-aircraft
            # target rings instead of a static marker.
            "sample_per_aircraft": per_aircraft,
        }
    return {"name": name, "kind": "custom", "repr": repr(q)}


def scenario_preview(spec: DesignSpec, *, seed: int = 0) -> dict[str, Any]:
    """Return renderable geometry for a spec: airspace, queryables, spawn + samples.

    A single seeded ``iter_spawns`` draw gives a representative set of aircraft
    so the map can show where traffic appears, without stepping the simulator.
    """
    scenario = build_scenario(spec)
    rng = np.random.default_rng(seed)
    # Sample (not support) so per-episode randomisation - rotation and, below,
    # spawn locations - is what the map shows; reseeding varies it.
    episode = scenario.sample(rng)
    _normalize_spawn_types(episode.spawn, spec.env.allowed_aircraft)

    airspace = (
        _bounds_geometry(episode.airspace_bounds)
        if episode.airspace_bounds is not None
        else None
    )

    queryables = [
        _queryable_geometry(
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

    spawn_regions = []
    for i, region in enumerate(episode.spawn.regions):
        # A spawn region's altitude is its bounds' altitude band - the same band
        # the designer derives the spawn `alt_ft` range from - so the box renders
        # at the altitudes aircraft spawn into.
        spawn_regions.append(
            {
                "name": region.name or f"SPAWN {i}",
                "render_shape": region.render_shape,
                "render_name": region.render_name,
                "max_aircraft": region.max_n(),
                # Initial-heading range (deg) for the spawn-direction arrows, or
                # None when heading is unconstrained (uniform 0-360).
                "heading": _heading_range(region.params.get("hdg_deg")),
                **_bounds_geometry(region.bounds),
            }
        )

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
        # disc in its colour), not just a screen-space dot.
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
        from bluesky_sandbox.sim.scenario import transforms as _t

        for name, bounds in sink.items():
            episode_bounds = bounds
            if rot:
                episode_bounds = _t.rotate_bounds(bounds, rot["pivot"], rot["angle"])
            elif group_maps and group_chains.get(name):
                chain = [group_maps[g] for g in group_chains[name] if g in group_maps]
                if chain:
                    episode_bounds = _t.transform_bounds(bounds, _t.compose(*chain))
            regions[name] = {"name": name, **_bounds_geometry(episode_bounds)}

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
    boundary and the observation-normalisation range). Any content whose
    footprint or finite altitude extent falls outside the airspace is reported
    by name so the user can enlarge the airspace or move the content in.
    """
    asp = episode.airspace_bounds
    if asp is None:
        return []
    out: list[str] = []

    # Content that reuses the airspace bounds sits exactly on its edge, where
    # contains() is strict (boundary excluded). Nudge each tested point a hair
    # toward the airspace centre so coincident/shared geometry counts as
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
