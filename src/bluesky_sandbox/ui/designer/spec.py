"""Serialisation layer: simulation primitives <-> JSON-able spec dicts.

Every serialised primitive is a dict tagged with a ``"type"`` field. The two
public entry points are :func:`dump` (object -> dict) and :func:`load`
(dict -> object); :func:`dumps` / :func:`loads` add the JSON hop.

Coverage matches the structured-data primitives:

* footprints - box / disk / polygon / sector / annular_sector / boolean
  (and a polygon fallback for arbitrary shapely footprints),
* altitude bands - constant / linear / radial / vertex,
* bounds - ``RegionBounds``,
* queryables - ``QueryRegion`` / ``Waypoint``,
* scalar-or-distribution values - fixed scalars, ``(low, high)`` ranges,
  frozen ``scipy.stats`` distributions, and :class:`Categorical`,
* spawn - ``SpawnRegion`` / ``SpawnConfig``.

Logic (reward/termination/field classes) is *not* serialised here; the
top-level :class:`DesignSpec` references it by ``"module:attr"`` import
string. See :mod:`bluesky_sandbox.ui.designer.builder`.

Two deliberate fidelity choices:

* ``Waypoint`` round-trips by *identifier* when it has one - a name-resolved
  waypoint dumps its ``waypoint`` name (not the resolved lat/lon) so it
  re-resolves against the live navdb on load. Ad-hoc lat/lon waypoints dump
  their coordinates. This is the "reference by identifier, resolve at build"
  principle applied to serialisation.
* Infinities (open altitude bands default to ``+/-inf``) are encoded as the
  JSON-safe strings ``"inf"`` / ``"-inf"`` so the document parses in a
  browser's ``JSON.parse`` without ``Infinity`` tokens.
"""

from __future__ import annotations

import ast
import json
import math
import textwrap
from dataclasses import dataclass, field
from typing import Any

import scipy.stats as _scipy_stats

from bluesky_sandbox.sim.bounds import (
    AnnularSectorFootprint,
    BooleanFootprint,
    Bounds,
    BoxFootprint,
    ConstantAltitudeBand,
    DiskFootprint,
    Footprint,
    LatLon,
    LinearAltitudeBand,
    PolygonFootprint,
    RadialAltitudeBand,
    RegionBounds,
    SectorFootprint,
    VertexAltitudeBand,
)
from bluesky_sandbox.sim.bounds.altitude import AltitudeBand
from bluesky_sandbox.sim.bounds.footprints import ShapelyFootprint
from bluesky_sandbox.sim.performance.envelope import EnvelopeSample
from bluesky_sandbox.sim.queryables import Queryable, QueryRegion, Waypoint
from bluesky_sandbox.sim.sampling.distributions import Bounded, Categorical
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion

from .transforms import bbox_center, rotate_bounds

# reward/terminated/truncated are always-present env hooks (default 0.0 / False).
DEFAULT_HOOKS = ("reward", "terminated", "truncated")

# Scenario-side hooks: ``name -> (args, one-line purpose)``. These run on the
# generated ``Scenario``, not the env, and are the escape hatch for episode
# sampling the structured spec cannot express. ``episode_geometry`` receives the
# geometry dict the structured design just built - keys ``airspace_bounds`` /
# ``spawn`` / ``queryables`` / ``sampled_waypoints`` - plus the episode ``rng``,
# and returns the dict to actually use. Returning it unchanged is the no-op.
SCENARIO_HOOKS: dict[str, tuple[tuple[str, ...], str]] = {
    "episode_geometry": (
        ("geometry", "rng"),
        "post-process this episode's geometry dict; must return it",
    ),
}


def _validated_scenario_hooks(raw: Any) -> dict[str, str]:
    """Coerce a ``scenario_hooks`` mapping, rejecting unknown hook names.

    Unknown names are an error rather than a silent drop: a body under a
    misspelled key would round-trip through the designer looking live while
    never being emitted.
    """
    hooks = {str(name): str(body) for name, body in dict(raw).items()}
    unknown = sorted(set(hooks) - set(SCENARIO_HOOKS))
    if unknown:
        raise SpecError(
            f"unknown scenario hook(s) {unknown}; "
            f"expected one of {sorted(SCENARIO_HOOKS)}"
        )
    return hooks


def _func_body_source(source: str, name: str) -> str | None:
    """Extract a top-level function's body source (dedented), or None."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name and node.body:
            start = node.body[0].lineno - 1
            return textwrap.dedent("\n".join(lines[start : node.end_lineno])).rstrip()
    return None



class SpecError(ValueError):
    """Raised when a spec dict is malformed or references an unknown type."""


# --------------------------------------------------------------------------- #
# Non-finite-safe float helpers                                               #
# --------------------------------------------------------------------------- #
def _f(x: float) -> float | str:
    """Dump a float, encoding infinities as JSON-safe strings."""
    x = float(x)
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if math.isnan(x):
        return "nan"
    return x


def _pf(x: Any) -> float:
    """Parse a float that may have been encoded as ``"inf"`` / ``"-inf"``."""
    if isinstance(x, str):
        low = x.strip().lower()
        if low in ("inf", "+inf", "infinity"):
            return math.inf
        if low in ("-inf", "-infinity"):
            return -math.inf
        if low == "nan":
            return math.nan
    return float(x)


def _latlon_dump(p: LatLon) -> dict[str, float]:
    return {"lat_deg": float(p.lat_deg), "lon_deg": float(p.lon_deg)}


def _latlon_load(d: Any) -> LatLon:
    if isinstance(d, LatLon):
        return d
    if isinstance(d, (list, tuple)) and len(d) == 2:
        return LatLon(float(d[0]), float(d[1]))
    if isinstance(d, dict):
        return LatLon(float(d["lat_deg"]), float(d["lon_deg"]))
    raise SpecError(f"cannot parse LatLon from {d!r}")


def _band_dump(band: tuple[float, float]) -> list[float | str]:
    return [_f(band[0]), _f(band[1])]


def _band_load(b: Any) -> tuple[float, float]:
    return (_pf(b[0]), _pf(b[1]))


# --------------------------------------------------------------------------- #
# Scalar-or-distribution union                                                #
# --------------------------------------------------------------------------- #
def _is_frozen_scipy(v: Any) -> bool:
    return hasattr(v, "dist") and hasattr(v.dist, "name") and hasattr(v, "rvs")


def dump_value(v: Any) -> Any:
    """Dump a scalar-or-distribution union value.

    Handles the value types accepted across ``SpawnRegion`` / ``SpawnConfig``
    fields: fixed scalars, ``(low, high)`` ranges, frozen ``scipy.stats``
    distributions, :class:`Categorical`, lists (callsign pools / routes), and
    ``None``. Plain scalars/strings/lists pass through unchanged; objects
    become ``{"type": ...}`` dicts.
    """
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, EnvelopeSample):
        out: dict[str, Any] = {"type": "envelope"}
        if v.alt_floor_ft != 1000.0:
            out["alt_floor_ft"] = float(v.alt_floor_ft)
        return out
    if isinstance(v, tuple) and len(v) == 2:
        return {"type": "range", "low": _f(v[0]), "high": _f(v[1])}
    if isinstance(v, Categorical):
        return {"type": "categorical", "weights": {k: float(w) for k, w in v.weights.items()}}
    if isinstance(v, Bounded):
        inner = v.dist
        return {
            "type": "scipy",
            "name": inner.dist.name,
            "args": [_f(a) for a in inner.args],
            "kwds": {k: _f(w) for k, w in inner.kwds.items()},
            "bounds": [_f(v.lo), _f(v.hi)],
            "mode": v.mode,
        }
    if _is_frozen_scipy(v):
        return {
            "type": "scipy",
            "name": v.dist.name,
            "args": [_f(a) for a in v.args],
            "kwds": {k: _f(w) for k, w in v.kwds.items()},
        }
    if isinstance(v, dict) and set(v) == {"route"}:
        return {"route": v["route"]}  # subroute reference step
    if isinstance(v, list):
        return [dump_value(x) for x in v]
    raise SpecError(f"cannot serialise value {v!r} of type {type(v).__name__}")


def load_value(d: Any) -> Any:
    """Inverse of :func:`dump_value`."""
    if isinstance(d, dict) and "type" in d:
        t = d["type"]
        if t == "range":
            return (_pf(d["low"]), _pf(d["high"]))
        if t == "categorical":
            return Categorical({k: float(w) for k, w in d["weights"].items()})
        if t == "scipy":
            name = d["name"]
            factory = getattr(_scipy_stats, name, None)
            if factory is None:
                raise SpecError(f"unknown scipy.stats distribution {name!r}")
            # Preserve integral params as ints: JSON round-trips (and JS
            # clients) blur int/float, but discrete distributions reject float
            # shape params - scipy's betabinom/binom rvs raises "Cannot cast
            # scalar from dtype('float64') to dtype('int64')" for n=6.0. An
            # integral float is numerically identical for continuous dists, so
            # the coercion is always safe.
            def _pnum(x: Any) -> float | int:
                v = _pf(x)
                if isinstance(v, float) and math.isfinite(v) and v.is_integer():
                    return int(v)
                return v

            args = [_pnum(a) for a in d.get("args", [])]
            kwds = {k: _pnum(w) for k, w in d.get("kwds", {}).items()}
            dist = factory(*args, **kwds)
            if "bounds" in d:
                lo, hi = _pf(d["bounds"][0]), _pf(d["bounds"][1])
                return Bounded(dist, lo, hi, mode=d.get("mode", "truncate"))
            return dist
        if t == "envelope":
            return EnvelopeSample(alt_floor_ft=float(d.get("alt_floor_ft", 1000.0)))
        raise SpecError(f"unknown value type {t!r}")
    if isinstance(d, list):
        return [load_value(x) for x in d]
    return d


# Waypoint constraint/target fields that may be per-episode sampled values
# (fixed number, range, or scipy distribution) rather than plain scalars.
SAMPLEABLE_WAYPOINT_FIELDS = (
    "alt_ft",
    "speed_kts",
    "reach_radius_nm",
    "alt_tolerance_ft",
    "speed_tolerance_kts",
)


def is_value_distribution(v: Any) -> bool:
    """True when ``v`` is a tagged sampled-value dict (range / scipy / categorical)."""
    return isinstance(v, dict) and v.get("type") in ("range", "scipy", "categorical")


# Marker for a waypoint alt/speed drawn per-aircraft from the flight envelope.
# Unlike range/scipy, it cannot resolve at build time (no aircraft yet), so the
# builder copies it onto route-step sampling metadata rather than a queryable.
ENVELOPE_VALUE = {"type": "envelope"}


def is_envelope_value(v: Any) -> bool:
    """True when ``v`` marks a per-aircraft flight-envelope sample (alt/speed)."""
    return isinstance(v, EnvelopeSample) or (
        isinstance(v, dict) and v.get("type") == "envelope"
    )


def representative_value(v: Any) -> Any:
    """A deterministic scalar standing in for a sampled value (for the support frame).

    Fixed → itself; range → midpoint; scipy → mean (falling back to median, then
    the low/finite bound) so the schema-support episode stays stable.
    """
    if not is_value_distribution(v):
        return v
    if v["type"] == "range":
        return (float(v["low"]) + float(v["high"])) / 2.0
    if v["type"] == "scipy":
        dist = load_value(v)
        for stat in (dist.mean, dist.median):
            try:
                m = float(stat())
            except Exception:
                continue
            if math.isfinite(m):
                return m
        return 0.0
    return 0.0


def extract_waypoint_field_dists(qdict: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a waypoint spec dict into (concrete dict, {field: encoded dist}).

    Any constraint field carrying a distribution is replaced by its
    representative scalar in the returned dict (so a :class:`Waypoint` can be
    constructed), and recorded in the dist map for per-episode resampling.
    """
    dists: dict[str, Any] = {}
    cleaned = dict(qdict)
    for field_name in SAMPLEABLE_WAYPOINT_FIELDS:
        value = cleaned.get(field_name)
        if is_value_distribution(value):
            dists[field_name] = value
            cleaned[field_name] = representative_value(value)
    return cleaned, dists


# --------------------------------------------------------------------------- #
# Footprints                                                                   #
# --------------------------------------------------------------------------- #
# Keys of a footprint dict that are structure, not sampleable scalar geometry.
_FOOTPRINT_STRUCTURAL_KEYS = frozenset(
    {"type", "center", "coords", "n_vertices", "op", "left", "right"}
)


def _fpv(v: Any) -> Any:
    """Footprint param value: representative scalar when sampled, else as-is.

    Lets ``_footprint_load`` build a concrete (representative) shape from a
    spec whose scalar params are range/scipy dicts; the per-episode sampling
    of those params is the builder's job (see ``footprint_param_dists``).
    """
    return representative_value(v) if is_value_distribution(v) else v


def footprint_param_dists(fp: dict[str, Any]) -> dict[str, Any]:
    """Sampled scalar params of a footprint dict, keyed by (dotted) param path.

    Introspects rather than enumerating per-shape params: any non-structural
    key holding a tagged sampled-value dict counts. Boolean footprints recurse
    as ``left.<param>`` / ``right.<param>``. Categorical values are rejected -
    geometry params must be scalar (range / scipy).
    """
    out: dict[str, Any] = {}
    for key, value in fp.items():
        if key in ("left", "right") and isinstance(value, dict):
            for sub, v in footprint_param_dists(value).items():
                out[f"{key}.{sub}"] = v
            continue
        if key in _FOOTPRINT_STRUCTURAL_KEYS:
            continue
        if is_value_distribution(value):
            if value.get("type") == "categorical":
                raise SpecError(
                    f"footprint param {key!r} cannot be categorical; "
                    "use a range or scipy distribution"
                )
            out[key] = value
    return out


def set_footprint_param(fp: dict[str, Any], path: str, value: float) -> None:
    """Set a (possibly dotted, for boolean footprints) footprint param in place."""
    node = fp
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = float(value)


def _footprint_dump(fp: Footprint) -> dict[str, Any]:
    if isinstance(fp, BoxFootprint):
        return {
            "type": "box",
            "lat_min_deg": float(fp.lat_min_deg),
            "lat_max_deg": float(fp.lat_max_deg),
            "lon_min_deg": float(fp.lon_min_deg),
            "lon_max_deg": float(fp.lon_max_deg),
        }
    if isinstance(fp, DiskFootprint):
        return {
            "type": "disk",
            "center": _latlon_dump(fp.center),
            "radius_nm": float(fp.radius_nm),
            "n_vertices": int(fp.n_vertices),
        }
    if isinstance(fp, PolygonFootprint):
        return {"type": "polygon", "coords": [[float(a), float(b)] for a, b in fp.coords]}
    if isinstance(fp, SectorFootprint):
        return {
            "type": "sector",
            "center": _latlon_dump(fp.center),
            "radius_nm": float(fp.radius_nm),
            "bearing_deg": float(fp.bearing_deg),
            "half_angle_deg": float(fp.half_angle_deg),
            "n_vertices": int(fp.n_vertices),
        }
    if isinstance(fp, AnnularSectorFootprint):
        return {
            "type": "annular_sector",
            "center": _latlon_dump(fp.center),
            "inner_radius_nm": float(fp.inner_radius_nm),
            "outer_radius_nm": float(fp.outer_radius_nm),
            "bearing_deg": float(fp.bearing_deg),
            "half_angle_deg": float(fp.half_angle_deg),
            "n_vertices": int(fp.n_vertices),
        }
    if isinstance(fp, BooleanFootprint):
        return {
            "type": "boolean",
            "op": fp.op,
            "left": _footprint_dump(fp.left),
            "right": _footprint_dump(fp.right),
        }
    if isinstance(fp, ShapelyFootprint):
        # No parametric handles survive a shapely result; degrade to polygon.
        return {"type": "polygon", "coords": [[a, b] for a, b in fp.vertices]}
    raise SpecError(f"cannot serialise footprint of type {type(fp).__name__}")


def _footprint_load(d: dict[str, Any]) -> Footprint:
    # Scalar params go through _fpv: a range/scipy dict loads as its
    # representative value here; per-episode sampling is layered on by the
    # builder (footprint_param_dists + the scenario's episode-geometry hook).
    t = d.get("type")
    if t == "box":
        return BoxFootprint(
            _fpv(d["lat_min_deg"]),
            _fpv(d["lat_max_deg"]),
            _fpv(d["lon_min_deg"]),
            _fpv(d["lon_max_deg"]),
        )
    if t == "disk":
        return DiskFootprint(
            _latlon_load(d["center"]), _fpv(d["radius_nm"]), d.get("n_vertices", 72)
        )
    if t == "polygon":
        return PolygonFootprint([(float(a), float(b)) for a, b in d["coords"]])
    if t == "sector":
        return SectorFootprint(
            _latlon_load(d["center"]),
            _fpv(d["radius_nm"]),
            _fpv(d["bearing_deg"]),
            _fpv(d["half_angle_deg"]),
            d.get("n_vertices", 24),
        )
    if t == "annular_sector":
        return AnnularSectorFootprint(
            _latlon_load(d["center"]),
            _fpv(d["inner_radius_nm"]),
            _fpv(d["outer_radius_nm"]),
            _fpv(d["bearing_deg"]),
            _fpv(d["half_angle_deg"]),
            d.get("n_vertices", 48),
        )
    if t == "boolean":
        return BooleanFootprint(
            d["op"], _footprint_load(d["left"]), _footprint_load(d["right"])
        )
    raise SpecError(f"unknown footprint type {t!r}")


# --------------------------------------------------------------------------- #
# Altitude bands                                                              #
# --------------------------------------------------------------------------- #
def _altitude_dump(band: AltitudeBand) -> dict[str, Any]:
    if isinstance(band, ConstantAltitudeBand):
        return {"type": "constant", "min_ft": _f(band.min_ft), "max_ft": _f(band.max_ft)}
    if isinstance(band, LinearAltitudeBand):
        return {
            "type": "linear",
            "start": _latlon_dump(band.start),
            "end": _latlon_dump(band.end),
            "start_band_ft": _band_dump(band.start_band_ft),
            "end_band_ft": _band_dump(band.end_band_ft),
        }
    if isinstance(band, RadialAltitudeBand):
        return {
            "type": "radial",
            "center": _latlon_dump(band.center),
            "radius_nm": float(band.radius_nm),
            "inner_band_ft": _band_dump(band.inner_band_ft),
            "outer_band_ft": _band_dump(band.outer_band_ft),
        }
    if isinstance(band, VertexAltitudeBand):
        return {
            "type": "vertex",
            "vertices": [[float(a), float(b)] for a, b in band.vertices],
            "min_values_ft": _vertex_values_dump(band.min_values_ft),
            "max_values_ft": _vertex_values_dump(band.max_values_ft),
        }
    raise SpecError(f"cannot serialise altitude band of type {type(band).__name__}")


def _vertex_values_dump(v: Any) -> Any:
    if isinstance(v, (int, float)):
        return _f(v)
    return [_f(x) for x in v]


def _vertex_values_load(v: Any) -> Any:
    if isinstance(v, list):
        return [_pf(x) for x in v]
    return _pf(v)


def _altitude_load(d: Any) -> AltitudeBand | None:
    if d is None:
        return None
    t = d.get("type")
    if t == "constant":
        return ConstantAltitudeBand(_pf(d.get("min_ft", "-inf")), _pf(d.get("max_ft", "inf")))
    if t == "linear":
        return LinearAltitudeBand(
            _latlon_load(d["start"]),
            _latlon_load(d["end"]),
            _band_load(d["start_band_ft"]),
            _band_load(d["end_band_ft"]),
        )
    if t == "radial":
        return RadialAltitudeBand(
            _latlon_load(d["center"]),
            d["radius_nm"],
            _band_load(d["inner_band_ft"]),
            _band_load(d["outer_band_ft"]),
        )
    if t == "vertex":
        return VertexAltitudeBand(
            [(float(a), float(b)) for a, b in d["vertices"]],
            _vertex_values_load(d["min_values_ft"]),
            _vertex_values_load(d["max_values_ft"]),
        )
    raise SpecError(f"unknown altitude band type {t!r}")


# --------------------------------------------------------------------------- #
# Bounds                                                                       #
# --------------------------------------------------------------------------- #
def _bounds_dump(b: Bounds) -> dict[str, Any]:
    if isinstance(b, RegionBounds):
        return {
            "type": "region",
            "footprint": _footprint_dump(b.footprint),
            "altitude": _altitude_dump(b.altitude) if b.altitude is not None else None,
        }
    raise SpecError(f"cannot serialise bounds of type {type(b).__name__}")


def _bounds_load(d: dict[str, Any]) -> Bounds:
    t = d.get("type")
    if t == "region":
        bounds = RegionBounds(
            _footprint_load(d["footprint"]), _altitude_load(d.get("altitude"))
        )
        # Optional static rotation of this bounds about its own centre (degrees,
        # CCW). Baked into the geometry here; the editor keeps the original
        # footprint + `rotation_deg` so the shape stays parametric to edit.
        rotation = d.get("rotation_deg")
        if rotation:
            bounds = rotate_bounds(bounds, bbox_center(bounds), float(rotation))
        return bounds
    raise SpecError(f"unknown bounds type {t!r}")


# --------------------------------------------------------------------------- #
# Queryables                                                                   #
# --------------------------------------------------------------------------- #
def _queryable_dump(q: Queryable) -> dict[str, Any]:
    if isinstance(q, QueryRegion):
        return {
            "type": "query_region",
            "bounds": _bounds_dump(q.bounds),
            "color": q.color,
            "render_shape": q.render_shape,
            "render_label": q.render_label,
            "track_temporal_state": q.track_temporal_state,
        }
    if isinstance(q, Waypoint):
        out: dict[str, Any] = {"type": "waypoint"}
        # Reference by identifier when we have one; otherwise store coords.
        if q.waypoint is not None:
            out["waypoint"] = q.waypoint
        else:
            out["lat"] = float(q.lat)
            out["lon"] = float(q.lon)
        for key in (
            "alt_ft",
            "speed_kts",
            "alt_tolerance_ft",
            "speed_tolerance_kts",
            "speed_tolerance_mach",
            "tsas_region",
            "render_tsas",
        ):
            val = getattr(q, key)
            if val is not None:
                out[key] = val
        if q.reach_radius_nm is None:
            out["reach_radius_nm"] = None
        else:
            out["reach_radius_nm"] = q.reach_radius_nm
        out["color"] = q.color
        out["render_shape"] = q.render_shape
        out["render_label"] = q.render_label
        out["track_temporal_state"] = q.track_temporal_state
        return out
    raise SpecError(
        f"cannot serialise queryable of type {type(q).__name__}; custom "
        "queryables must be referenced by code import string, not dumped."
    )


def _queryable_load(d: dict[str, Any]) -> Queryable:
    t = d.get("type")
    if t == "query_region":
        return QueryRegion(
            _bounds_load(d["bounds"]),
            color=d.get("color", "orange"),
            render_shape=d.get("render_shape", True),
            render_label=d.get("render_label", True),
            track_temporal_state=d.get("track_temporal_state", False),
        )
    if t == "waypoint":
        lat = d.get("lat")
        lon = d.get("lon")
        # A sampled waypoint draws its position from a region per episode (in
        # DesignScenario.sample). The static/support position is the region
        # centre; the web keeps lat/lon in sync, but derive it if absent. The
        # sample may be a footprint or a full bounds (region refs are inlined by
        # the builder before load).
        sample = d.get("sample")
        if sample is not None and "ref" not in sample:
            sampled_bounds = load(sample)
            if isinstance(sampled_bounds, Footprint):
                sampled_bounds = RegionBounds(sampled_bounds)
            if lat is None or lon is None:
                (lat_min, lat_max), (lon_min, lon_max) = sampled_bounds.bounding_box
                lat = (lat_min + lat_max) / 2.0
                lon = (lon_min + lon_max) / 2.0
        # Envelope markers are route-generation metadata. The queryable keeps no
        # per-aircraft sampling state, so its static target leaves that axis unset.
        alt_ft = d.get("alt_ft")
        speed_kts = d.get("speed_kts")
        sample_alt_env = is_envelope_value(alt_ft)
        sample_spd_env = is_envelope_value(speed_kts)
        return Waypoint(
            lat=lat,
            lon=lon,
            waypoint=d.get("waypoint"),
            alt_ft=None if sample_alt_env else alt_ft,
            speed_kts=None if sample_spd_env else speed_kts,
            reach_radius_nm=d.get("reach_radius_nm", 1.0),
            alt_tolerance_ft=d.get("alt_tolerance_ft"),
            speed_tolerance_kts=d.get("speed_tolerance_kts"),
            speed_tolerance_mach=d.get("speed_tolerance_mach"),
            color=d.get("color", "cyan"),
            tsas_region=d.get("tsas_region"),
            render_shape=d.get("render_shape", True),
            render_tsas=d.get("render_tsas"),
            render_label=d.get("render_label", True),
            track_temporal_state=d.get("track_temporal_state", False),
        )
    raise SpecError(f"unknown queryable type {t!r}")


# --------------------------------------------------------------------------- #
# Spawn                                                                        #
# --------------------------------------------------------------------------- #
def _spawn_region_dump(r: SpawnRegion) -> dict[str, Any]:
    return {
        "type": "spawn_region",
        "bounds": _bounds_dump(r.bounds),
        "n_aircraft": dump_value(r.n_aircraft),
        "params": {k: dump_value(v) for k, v in r.params.items()},
        "aircraft_type": dump_value(r.aircraft_type),
        "callsign_prefixes": dump_value(r.callsign_prefixes),
        "spawn_time": dump_value(r.spawn_time),
        "route": dump_value(r.route),
        "name": r.name,
        "render_shape": r.render_shape,
        "render_name": r.render_name,
        "maintain": r.maintain,
        "controlled": r.controlled,
        "conflict_free_spawn": r.conflict_free_spawn,
        "conflict_free_margin_nm": r.conflict_free_margin_nm,
        "conflict_free_margin_ft": r.conflict_free_margin_ft,
        "conflict_free_margin_s": r.conflict_free_margin_s,
    }


def _spawn_region_load(d: dict[str, Any]) -> SpawnRegion:
    return SpawnRegion(
        bounds=_bounds_load(d["bounds"]),
        n_aircraft=load_value(d["n_aircraft"]),
        params={k: load_value(v) for k, v in d.get("params", {}).items()},
        aircraft_type=load_value(d.get("aircraft_type")),
        callsign_prefixes=load_value(d.get("callsign_prefixes")),
        spawn_time=load_value(d.get("spawn_time", 0.0)),
        route=load_value(d.get("route")),
        name=d.get("name"),
        render_shape=d.get("render_shape", True),
        render_name=d.get("render_name", True),
        maintain=d.get("maintain", False),
        controlled=d.get("controlled", True),
        conflict_free_spawn=d.get("conflict_free_spawn"),
        conflict_free_margin_nm=d.get("conflict_free_margin_nm"),
        conflict_free_margin_ft=d.get("conflict_free_margin_ft"),
        conflict_free_margin_s=d.get("conflict_free_margin_s"),
    )


def _spawn_config_dump(c: SpawnConfig) -> dict[str, Any]:
    return {
        "type": "spawn_config",
        "regions": [_spawn_region_dump(r) for r in c.regions],
        "aircraft_type": dump_value(c.aircraft_type),
        "route": dump_value(c.route),
        "routes": {k: list(v) for k, v in c.routes.items()},
        "conflict_free_spawn": c.conflict_free_spawn,
        "conflict_free_margin_nm": c.conflict_free_margin_nm,
        "conflict_free_margin_ft": c.conflict_free_margin_ft,
        "conflict_free_margin_s": c.conflict_free_margin_s,
    }


def _spawn_config_load(d: dict[str, Any]) -> SpawnConfig:
    return SpawnConfig(
        regions=[_spawn_region_load(r) for r in d["regions"]],
        aircraft_type=load_value(d.get("aircraft_type")),
        route=load_value(d.get("route")),
        routes={k: list(v) for k, v in d.get("routes", {}).items()},
        conflict_free_spawn=d.get("conflict_free_spawn", False),
        conflict_free_margin_nm=d.get("conflict_free_margin_nm", 0.0),
        conflict_free_margin_ft=d.get("conflict_free_margin_ft", 0.0),
        conflict_free_margin_s=d.get("conflict_free_margin_s", 0.0),
    )


# --------------------------------------------------------------------------- #
# Public dump / load dispatch                                                 #
# --------------------------------------------------------------------------- #
_LOADERS = {
    "box": _footprint_load,
    "disk": _footprint_load,
    "polygon": _footprint_load,
    "sector": _footprint_load,
    "annular_sector": _footprint_load,
    "boolean": _footprint_load,
    "constant": _altitude_load,
    "linear": _altitude_load,
    "radial": _altitude_load,
    "vertex": _altitude_load,
    "region": _bounds_load,
    "query_region": _queryable_load,
    "waypoint": _queryable_load,
    "spawn_region": _spawn_region_load,
    "spawn_config": _spawn_config_load,
}


def dump(obj: Any) -> dict[str, Any]:
    """Serialise a supported primitive to a tagged spec dict.

    Accepts footprints, altitude bands, ``Bounds``, queryables, ``SpawnRegion``,
    and ``SpawnConfig``. For scalar-or-distribution union values use
    :func:`dump_value` instead.
    """
    if isinstance(obj, Footprint):
        return _footprint_dump(obj)
    if isinstance(obj, AltitudeBand):
        return _altitude_dump(obj)
    if isinstance(obj, Bounds):
        return _bounds_dump(obj)
    if isinstance(obj, (QueryRegion, Waypoint)):
        return _queryable_dump(obj)
    if isinstance(obj, SpawnRegion):
        return _spawn_region_dump(obj)
    if isinstance(obj, SpawnConfig):
        return _spawn_config_dump(obj)
    raise SpecError(f"cannot dump object of type {type(obj).__name__}")


def load(d: dict[str, Any]) -> Any:
    """Reconstruct a primitive from a tagged spec dict produced by :func:`dump`."""
    if not isinstance(d, dict) or "type" not in d:
        raise SpecError(f"expected a tagged dict with a 'type' key, got {d!r}")
    loader = _LOADERS.get(d["type"])
    if loader is None:
        raise SpecError(f"unknown spec type {d['type']!r}")
    return loader(d)


def dumps(obj: Any, **json_kwargs: Any) -> str:
    """Serialise a primitive straight to a JSON string."""
    return json.dumps(dump(obj), **json_kwargs)


def loads(s: str) -> Any:
    """Reconstruct a primitive from a JSON string produced by :func:`dumps`."""
    return load(json.loads(s))


# --------------------------------------------------------------------------- #
# Field references (code tab) and the top-level design document               #
# --------------------------------------------------------------------------- #
@dataclass
class FieldRef:
    """Reference to an observation/action field class plus constructor kwargs.

    ``name`` is a class name resolved against
    ``bluesky_sandbox.interface.fields.observations`` (and ``...actions`` for action
    fields) at build time. ``kwargs`` are passed to the constructor, so bound
    overrides (e.g. custom ``low`` / ``high``) survive the round trip.

    ``transform`` optionally names a no-/kw-arg method to call on the
    constructed field, with ``transform_kwargs``. This supports the documented
    intruder pattern ``obs.AltFt().relative_to_own()`` -> a ``PairObsField``.
    """

    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    transform: str | None = None
    transform_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"field": self.name}
        if self.kwargs:
            out["kwargs"] = dict(self.kwargs)
        if self.transform:
            out["transform"] = self.transform
            if self.transform_kwargs:
                out["transform_kwargs"] = dict(self.transform_kwargs)
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FieldRef:
        return cls(
            name=d["field"],
            kwargs=dict(d.get("kwargs", {})),
            transform=d.get("transform"),
            transform_kwargs=dict(d.get("transform_kwargs", {})),
        )


@dataclass
class TaskInfoSpec:
    """Inline task-info provider edited in the designer.

    ``body`` is the function body for a provider with signature
    ``(obs, action, info, context, rng) -> None``.
    """

    name: str
    body: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "body": self.body}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskInfoSpec:
        return cls(name=str(d["name"]), body=str(d.get("body", "")))


@dataclass
class EnvSpec:
    """Non-geometry environment settings plus code references.

    Reward / termination / truncation are ``@overridable`` env hooks (in
    :attr:`hooks`, always present), not config functions. Custom field/code
    references resolve at build time - the code-tab half of the design seam.
    """

    obs_fields: list[FieldRef] = field(default_factory=list)
    intruder_obs_fields: list[FieldRef] | None = None
    # Privileged, critic-only observation fields (asymmetric actor-critic / CTDE):
    # seen by the value function at training time but never by the actor, so the
    # deployed policy stays a function of the actor-side lists only. Mirror the
    # ownship / intruder lists; both default to None (symmetric).
    critic_obs_fields: list[FieldRef] | None = None
    critic_intruder_obs_fields: list[FieldRef] | None = None
    action_fields: list[FieldRef] = field(default_factory=list)
    # Module-level Python emitted before inline task-info providers. Use this
    # for imports, constants, and small helper functions shared by providers.
    task_info_setup: str = ""
    task_info: list[TaskInfoSpec] = field(default_factory=list)
    # Advanced/compatibility path for externally-defined task-info providers.
    # New designer-authored providers should use ``task_info`` above.
    task_info_providers: list[str] = field(default_factory=list)
    # Env hook bodies: name -> method body source. reward/terminated/truncated
    # are always present (seeded by the GUI); other hooks are opt-in.
    # Module-level Python emitted before the generated env class. Use this for
    # imports, constants, and helpers shared by hook methods.
    hook_setup: str = ""
    hooks: dict[str, str] = field(default_factory=dict)
    allowed_aircraft: list[str] = field(default_factory=lambda: ["B744"])
    dt: float = 1.0
    simdt: float = 0.05
    cd_method: str = "CSTATEBASED"
    reso_method: str | None = None
    pz_radius_nm: float | None = None
    pz_height_ft: float | None = None
    lookahead_s: float | None = None
    performance_model: str | None = "openap"
    # Uniform wind field. ``wind_dir_deg`` is aviation-standard (direction the
    # wind blows FROM, degrees true clockwise from north); ``wind_kts`` its mean
    # speed; ``turbulence_kts`` an Ornstein-Uhlenbeck gust RMS decorrelating over
    # ``gust_tau_s`` seconds. Both speeds 0 = no wind.
    wind_dir_deg: float = 270.0
    wind_kts: float = 0.0
    turbulence_kts: float = 0.0
    gust_tau_s: float = 30.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "obs_fields": [f.to_dict() for f in self.obs_fields],
            "intruder_obs_fields": (
                None
                if self.intruder_obs_fields is None
                else [f.to_dict() for f in self.intruder_obs_fields]
            ),
            "critic_obs_fields": (
                None
                if self.critic_obs_fields is None
                else [f.to_dict() for f in self.critic_obs_fields]
            ),
            "critic_intruder_obs_fields": (
                None
                if self.critic_intruder_obs_fields is None
                else [f.to_dict() for f in self.critic_intruder_obs_fields]
            ),
            "action_fields": [f.to_dict() for f in self.action_fields],
            "task_info_setup": self.task_info_setup,
            "task_info": [p.to_dict() for p in self.task_info],
            "task_info_providers": list(self.task_info_providers),
            "hook_setup": self.hook_setup,
            "hooks": dict(self.hooks),
            "allowed_aircraft": list(self.allowed_aircraft),
            "dt": self.dt,
            "simdt": self.simdt,
            "cd_method": self.cd_method,
            "reso_method": self.reso_method,
            "pz_radius_nm": self.pz_radius_nm,
            "pz_height_ft": self.pz_height_ft,
            "lookahead_s": self.lookahead_s,
            "performance_model": self.performance_model,
            "wind_dir_deg": self.wind_dir_deg,
            "wind_kts": self.wind_kts,
            "turbulence_kts": self.turbulence_kts,
            "gust_tau_s": self.gust_tau_s,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EnvSpec:
        intruders = d.get("intruder_obs_fields")
        critic_own = d.get("critic_obs_fields")
        critic_intr = d.get("critic_intruder_obs_fields")
        return cls(
            obs_fields=[FieldRef.from_dict(f) for f in d.get("obs_fields", [])],
            intruder_obs_fields=(
                None if intruders is None else [FieldRef.from_dict(f) for f in intruders]
            ),
            critic_obs_fields=(
                None if critic_own is None else [FieldRef.from_dict(f) for f in critic_own]
            ),
            critic_intruder_obs_fields=(
                None
                if critic_intr is None
                else [FieldRef.from_dict(f) for f in critic_intr]
            ),
            action_fields=[FieldRef.from_dict(f) for f in d.get("action_fields", [])],
            task_info_setup=str(d.get("task_info_setup", "")),
            task_info=[
                TaskInfoSpec.from_dict(p) for p in d.get("task_info", [])
            ],
            task_info_providers=list(d.get("task_info_providers", [])),
            hook_setup=str(d.get("hook_setup", "")),
            hooks=dict(d.get("hooks", {})),
            allowed_aircraft=list(d.get("allowed_aircraft", ["B744"])),
            dt=d.get("dt", 1.0),
            simdt=d.get("simdt", 0.05),
            cd_method=d.get("cd_method", "CSTATEBASED"),
            reso_method=d.get("reso_method"),
            pz_radius_nm=d.get("pz_radius_nm"),
            pz_height_ft=d.get("pz_height_ft"),
            lookahead_s=d.get("lookahead_s"),
            performance_model=d.get("performance_model", "openap"),
            wind_dir_deg=d.get("wind_dir_deg", 270.0),
            wind_kts=d.get("wind_kts", 0.0),
            turbulence_kts=d.get("turbulence_kts", 0.0),
            gust_tau_s=d.get("gust_tau_s", 30.0),
        )


_SPEC_VERSION = 1


@dataclass
class DesignSpec:
    """Top-level design document - the single source of truth for an environment.

    Geometry/spawn/queryables are held as already-dumped spec dicts (materialise
    them with :func:`load`); ``env`` carries field refs and code import strings.
    The airspace is a singleton ``Bounds`` reused by query/spawn regions, exactly
    as the runtime scenarios do.
    """

    env: EnvSpec
    spawn: dict[str, Any]
    airspace: dict[str, Any] | None = None
    queryables: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Named, reusable bounds. Any consumer that takes a bounds dict (airspace,
    # query-region, spawn-region, waypoint sample, spawn destination) may use
    # ``{"ref": "<name>"}`` to reference one of these instead of inlining it.
    regions: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Named waypoint stacks: several waypoint queryables that share ONE
    # per-episode position draw and take correlated altitudes, so a set of
    # streams can be routed onto a single merge/diverge point at either the same
    # level (forcing in-trail sequencing) or stacked levels. Each entry:
    #
    #   {"members": [<queryable name>, ...],      # ordered; member i is level i
    #    "sample": <bounds-or-ref>,               # one lat/lon drawn per episode
    #    "alt_base_ft": <value|dist>,             # level of member 0; omitted =
    #                                             #   drawn from sample's alt band
    #    "alt_step_ft": <value|dist>,             # member i = base + i * step
    #    "co_altitude_prob": <float in [0, 1]>}   # P(step forced to 0)
    #
    # Members must be per-episode waypoints (not ``sample_per == "aircraft"``);
    # the stack owns their position and altitude, so a member is excluded from
    transform: dict[str, Any] | None = None
    code: dict[str, str] = field(default_factory=dict)
    # Module-level Python prepended to the generated ``scenario.py``, the
    # scenario-side twin of ``env.hook_setup``. Helpers defined here are in
    # scope for ``scenario_hooks`` below, so a design can carry a sampler the
    # structured spec cannot express (a randomised merge topology, a spawn
    # config built per episode) without dropping out of the designer.
    scenario_setup: str = ""
    # name -> body, for the scenario hooks in :data:`SCENARIO_HOOKS`. Emitted as
    # ``@staticmethod`` on the generated Scenario, so the body sees only its
    # declared arguments plus ``scenario_setup``'s module scope.
    scenario_hooks: dict[str, str] = field(default_factory=dict)
    nav_cycle: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = _SPEC_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "nav_cycle": self.nav_cycle,
            "metadata": dict(self.metadata),
            "airspace": self.airspace,
            "queryables": dict(self.queryables),
            "regions": dict(self.regions),
            "spawn": self.spawn,
            "transform": self.transform,
            "env": self.env.to_dict(),
            "code": dict(self.code),
            "scenario_setup": self.scenario_setup,
            "scenario_hooks": dict(self.scenario_hooks),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DesignSpec:
        version = d.get("version", _SPEC_VERSION)
        if version != _SPEC_VERSION:
            raise SpecError(
                f"unsupported DesignSpec version {version}; this build expects "
                f"{_SPEC_VERSION}."
            )
        env = EnvSpec.from_dict(d["env"])
        code = dict(d.get("code", {}))
        # Migrate the old model (reward/termination/truncation as task.py
        # functions referenced by env.reward_fn) into reward/terminated/truncated
        # hooks, then drop the now-unused task.py reward file.
        if d["env"].get("reward_fn") is not None:
            task_src = code.get("task.py", "")
            for hook in DEFAULT_HOOKS:
                if hook not in env.hooks:
                    body = _func_body_source(task_src, hook)
                    if body:
                        env.hooks[hook] = body
            code.pop("task.py", None)
        return cls(
            env=env,
            spawn=d["spawn"],
            airspace=d.get("airspace"),
            queryables=dict(d.get("queryables", {})),
            regions=dict(d.get("regions", {})),
            # Absent in pre-stack designs: an empty mapping is the no-op.
            transform=d.get("transform"),
            code=code,
            # Absent in pre-scenario-hook designs: empty is the no-op.
            scenario_setup=str(d.get("scenario_setup") or ""),
            scenario_hooks=_validated_scenario_hooks(d.get("scenario_hooks") or {}),
            nav_cycle=d.get("nav_cycle"),
            metadata=dict(d.get("metadata", {})),
            version=version,
        )

    def to_json(self, **json_kwargs: Any) -> str:
        json_kwargs.setdefault("indent", 2)
        return json.dumps(self.to_dict(), **json_kwargs)

    @classmethod
    def from_json(cls, s: str) -> DesignSpec:
        return cls.from_dict(json.loads(s))
