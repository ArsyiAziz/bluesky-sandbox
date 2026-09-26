"""An episode's geometry as JSON-able data, for anything that draws it.

Airspace, queryables and spawn regions become plain polygons and points: each
bound's vertices, bounding box and altitude band (per vertex, where the band
varies); each region and waypoint with how it is drawn. Built from the
primitives' own accessors, so what is drawn is what the env builds. The
designer's preview and a live view of a running env share it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bluesky_sandbox.sim.bounds import Bounds
from bluesky_sandbox.sim.queryables import QueryRegion, Waypoint

__all__ = [
    "bounds_geometry",
    "episode_geometry",
    "heading_range",
    "queryable_geometry",
    "spawn_region_geometry",
]


def bounds_geometry(bounds: Bounds) -> dict[str, Any]:
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


def heading_range(hdg: Any) -> list[float] | None:
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


def queryable_geometry(name: str, q: Any, *, per_aircraft: bool = False) -> dict[str, Any]:
    if isinstance(q, QueryRegion):
        return {
            "name": name,
            "kind": "region",
            "color": q.color,
            "render_shape": q.render_shape,
            "render_label": q.render_label,
            **bounds_geometry(q.bounds),
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


def spawn_region_geometry(index: int, region: Any) -> dict[str, Any]:
    """A spawn region: its bounds, drawn at the altitude band aircraft spawn
    into, with its name and the initial-heading range for direction arrows."""
    return {
        "name": region.name or f"SPAWN {index}",
        "render_shape": region.render_shape,
        "render_name": region.render_name,
        "max_aircraft": region.max_n(),
        # Initial-heading range (deg), or None when unconstrained (uniform 0-360).
        "heading": heading_range(region.params.get("hdg_deg")),
        **bounds_geometry(region.bounds),
    }


def episode_geometry(episode: Any, per_aircraft: frozenset[str] = frozenset()) -> dict[str, Any]:
    """An episode's airspace, queryables and spawn regions. ``per_aircraft``
    names the waypoints sampled per aircraft, whose template point is not one."""
    return {
        "airspace": (
            bounds_geometry(episode.airspace_bounds)
            if episode.airspace_bounds is not None
            else None
        ),
        "queryables": [
            queryable_geometry(name, q, per_aircraft=name in per_aircraft)
            for name, q in episode.queryables.items()
        ],
        "spawn_regions": [
            spawn_region_geometry(i, region) for i, region in enumerate(episode.spawn.regions)
        ],
    }
