"""A configurable, randomised :class:`Scenario` over materialised resources.

:class:`RandomizedScenario` turns a fixed set of runtime resources - an airspace
:class:`~bluesky_sandbox.sim.bounds.Bounds`, a :class:`~bluesky_sandbox.sim.spawn.SpawnConfig`,
and a mapping of named :class:`~bluesky_sandbox.sim.queryables.Queryable` - into a
per-episode sampler with two forms of domain randomisation:

* ``sampled_waypoints`` redraws a waypoint's position (lat/lon, and altitude
  from the region band) from a :class:`Bounds` each episode;
* ``rotation`` rotates the *whole geometry as a group* (airspace + queryables +
  spawn) about a pivot by a sampled angle;
* ``groups`` generalises ``rotation`` to **several** rotation groups that each
  rotate a chosen subset of elements, and that **nest**: a group inside another
  is rotated locally first and then carried by its parent's rotation
  ("rotation in rotation"). ``rotation`` is the single-group special case.

``support()`` stays in the canonical (unrotated, region-centre) frame so the
observation/action schemas are stable. This lives in the core API (not the
designer) so generated task packages depend only on the main library.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from bluesky_sandbox.sim.bounds import Bounds
from bluesky_sandbox.sim.queryables import Queryable, Waypoint
from bluesky_sandbox.sim.spawn import SpawnConfig

from . import transforms as _t
from .base import EpisodeSpec, Scenario


def _mean_point(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Midpoint of the lat/lon bounding box of ``points``."""
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    return ((min(lats) + max(lats)) / 2.0, (min(lons) + max(lons)) / 2.0)


def _episode_spec(
    scenario: RandomizedScenario,
    airspace: Bounds | None,
    spawn: SpawnConfig,
    queryables: dict[str, Queryable],
) -> EpisodeSpec:
    return EpisodeSpec(
        airspace_bounds=airspace,
        spawn=spawn,
        queryables=dict(queryables),
        max_aircraft=scenario.spawn.max_aircraft(),
    )


def _default_pivot(scenario: RandomizedScenario) -> tuple[float, float]:
    if scenario.airspace_bounds is not None:
        return _t.bbox_center(scenario.airspace_bounds)
    if scenario.spawn.regions:
        return _t.bbox_center(scenario.spawn.regions[0].bounds)
    return (0.0, 0.0)


@dataclass(frozen=True)
class RandomizedScenario(Scenario):
    """Scenario over materialised resources with optional per-episode randomisation."""

    airspace_bounds: Bounds | None
    spawn: SpawnConfig
    queryables: dict[str, Queryable]
    rotation: dict[str, Any] | None = None  # {"angle": value|dist, "pivot": (lat,lon)|None}
    # Nestable rotation groups. Each: {"id", "angle": value|dist, "pivot": (lat,lon)|None,
    # "members": [element-id], "parent": id|None}. Element ids: "airspace", "q:<name>"
    # (queryable), "s:<name>" (spawn region). Supersedes ``rotation`` when set.
    groups: tuple[dict[str, Any], ...] | None = None
    # name -> Bounds a waypoint's position (lat/lon, and altitude) is drawn from per episode.
    sampled_waypoints: dict[str, Bounds] = field(default_factory=dict)
    # name -> {field: value|range|dist} for waypoint constraint/target fields that
    # are resampled each episode (reach radius, tolerances, target alt/speed).
    waypoint_fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Optional per-episode geometry hook, called with the episode rng at the top
    # of ``sample()``. Returns replacements for any of ``airspace_bounds`` /
    # ``spawn`` / ``queryables`` / ``sampled_waypoints`` (absent keys keep their
    # static value). This is what makes region *shape* randomisation - sampled
    # footprint parameters - expressible: the hook rebuilds the affected bounds
    # and everything referencing them each episode. The static fields must be
    # constructed to *cover* everything the hook can return (their union
    # support), because ``support()`` reports the static geometry.
    episode_geometry_fn: (
        Callable[[np.random.Generator], dict[str, Any]] | None
    ) = None

    _GEOMETRY_FIELDS = ("airspace_bounds", "spawn", "queryables", "sampled_waypoints")

    def __post_init__(self) -> None:
        # Stash the pristine (support-covering) geometry so ``support()`` can
        # report it even after ``sample()`` has swapped in episode geometry.
        if self.episode_geometry_fn is not None:
            object.__setattr__(
                self,
                "_pristine_geometry",
                {name: getattr(self, name) for name in self._GEOMETRY_FIELDS},
            )

    def _apply_episode_geometry(self, rng: np.random.Generator) -> None:
        if self.episode_geometry_fn is None:
            return
        overrides = self.episode_geometry_fn(rng) or {}
        unknown = set(overrides) - set(self._GEOMETRY_FIELDS)
        if unknown:
            raise ValueError(
                f"episode_geometry_fn returned unknown fields {sorted(unknown)}; "
                f"allowed: {list(self._GEOMETRY_FIELDS)}"
            )
        # Frozen dataclass: the swap is deliberate, contained here and restored
        # by ``support()`` from the pristine stash.
        for name, value in overrides.items():
            object.__setattr__(self, name, value)

    def _sample_queryables(self, rng: np.random.Generator) -> dict[str, Queryable]:
        """Return queryables with per-episode waypoint randomisation applied."""
        if not self.sampled_waypoints and not self.waypoint_fields:
            return self.queryables
        out: dict[str, Queryable] = {}
        for name, q in self.queryables.items():
            region = self.sampled_waypoints.get(name)
            if region is not None and isinstance(q, Waypoint):
                lat, lon = region.sample_point(rng)
                updates: dict[str, Any] = {"lat": lat, "lon": lon, "waypoint": None}
                # Draw altitude from the region's band too, but only when the
                # waypoint actually has an altitude target and the band varies.
                band = getattr(region, "alt_band_at", None)
                if band is not None and q.alt_ft is not None:
                    lo, hi = band(lat, lon)
                    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                        updates["alt_ft"] = float(rng.uniform(lo, hi))
                q = replace(q, **updates)
            fields = self.waypoint_fields.get(name)
            if fields and isinstance(q, Waypoint):
                updates = {f: _t.sample_scalar(v, rng) for f, v in fields.items()}
                # A named waypoint resolves lat/lon at construction but keeps its
                # name, so a plain replace() would trip the "not both" guard;
                # null lat/lon to let it re-resolve from the navdb name.
                if q.waypoint is not None:
                    updates = {"lat": None, "lon": None, **updates}
                q = replace(q, **updates)
            out[name] = q
        return out

    # ------------------------------------------------------------------ groups
    def _bbox_points(
        self,
        member_ids: set[str],
        airspace: Bounds | None,
        queryables: dict[str, Queryable],
        spawn: SpawnConfig,
    ) -> list[tuple[float, float]]:
        """Corner/anchor points of every element in ``member_ids`` (canonical)."""
        pts: list[tuple[float, float]] = []

        def add_bounds(b: Bounds | None) -> None:
            if b is None:
                return
            (lat_min, lat_max), (lon_min, lon_max) = b.bounding_box
            pts.extend([(lat_min, lon_min), (lat_max, lon_max)])

        if "airspace" in member_ids:
            add_bounds(airspace)
        for name, q in queryables.items():
            if f"q:{name}" not in member_ids:
                continue
            if isinstance(q, Waypoint):
                pts.append((q.lat, q.lon))
            else:
                add_bounds(getattr(q, "bounds", None))
        for region in spawn.regions:
            if f"s:{region.name}" in member_ids:
                add_bounds(region.bounds)
        return pts

    def _group_geometry(self):
        """Resolve groups to (chains-by-element, pivots-by-id, angles-sampler).

        Returns ``(children, subtree_ids, order)`` helpers as plain dicts keyed
        by group id.
        """
        groups = {g["id"]: g for g in self.groups or ()}
        children: dict[str, list[str]] = {gid: [] for gid in groups}
        for gid, g in groups.items():
            parent = g.get("parent")
            if parent in children:
                children[parent].append(gid)

        def subtree(gid: str) -> set[str]:
            ids = set(groups[gid].get("members", ()))
            for child in children[gid]:
                ids |= subtree(child)
            return ids

        def chain(gid: str | None) -> list[str]:
            out: list[str] = []
            while gid is not None and gid in groups:
                out.append(gid)
                gid = groups[gid].get("parent")
            return out

        return groups, {gid: subtree(gid) for gid in groups}, chain

    def _apply_groups(
        self,
        queryables: dict[str, Queryable],
        rng: np.random.Generator,
    ) -> EpisodeSpec:
        airspace = self.airspace_bounds
        spawn = self.spawn
        groups, subtree, chain = self._group_geometry()

        # Sample one transform per group (rotation angle, east/north translation,
        # uniform scale); resolve each group's pivot from the *canonical*
        # (pre-transform) geometry so nesting is rigid and stable.
        angles = {gid: _t.sample_scalar(g.get("angle", 0.0), rng) for gid, g in groups.items()}
        scales = {gid: _t.sample_scalar(g.get("scale", 1.0), rng) for gid, g in groups.items()}
        offsets: dict[str, tuple[float, float]] = {}
        for gid, g in groups.items():
            t = g.get("translation")
            offsets[gid] = (
                (_t.sample_scalar(t.get("east", 0.0), rng), _t.sample_scalar(t.get("north", 0.0), rng))
                if t
                else (0.0, 0.0)
            )
        pivots: dict[str, tuple[float, float]] = {}
        for gid, g in groups.items():
            pivot = g.get("pivot")
            if pivot:
                pivots[gid] = (pivot[0], pivot[1])
            else:
                pts = self._bbox_points(subtree[gid], airspace, queryables, spawn)
                pivots[gid] = _mean_point(pts) if pts else _default_pivot(self)

        # Each group's whole transform is a single point map: scale, then rotate
        # (both about the pivot), then translate. Composing point maps is exact,
        # so applying the composition once equals applying each step in turn.
        group_map = {
            gid: _t.compose(
                _t.scaler(pivots[gid], scales[gid]),
                _t.rotator(pivots[gid], angles[gid]),
                _t.translator(*offsets[gid]),
            )
            for gid in groups
        }

        # An element's transform chain = its group then that group's ancestors
        # (inner-most first), so nested groups compose: local first, then carried.
        elem_chain: dict[str, list[str]] = {}
        for gid, g in groups.items():
            for eid in g.get("members", ()):
                elem_chain[eid] = chain(gid)

        def spin(geom, eid, apply):
            ch = elem_chain.get(eid)
            if not ch:
                return geom
            return apply(geom, _t.compose(*(group_map[gid] for gid in ch)))

        # Expose this episode's composed group maps for tooling (the designer's
        # map preview transforms named-region geometry with them).
        object.__setattr__(self, "last_group_maps", dict(group_map))

        # Route steps carry per-aircraft waypoint sample bounds (e.g. an exit
        # corridor). They must move with their *waypoint's* group - mirroring
        # the single-rotation path's rotate_spawn - or grouped designs leave
        # per-aircraft sampled waypoints untransformed.
        def transform_step(step):
            if not isinstance(step, dict):
                return step
            out = dict(step)
            sample = out.get("sample")
            name = out.get("waypoint")
            ch = elem_chain.get(f"q:{name}") if name else None
            if isinstance(sample, Bounds) and ch:
                out["sample"] = _t.transform_bounds(
                    sample, _t.compose(*(group_map[gid] for gid in ch))
                )
            if "choice" in out:
                out["choice"] = [
                    [transform_step(s) for s in branch]
                    if isinstance(branch, list)
                    else transform_step(branch)
                    for branch in out["choice"]
                ]
            return out

        def transform_route(route):
            if not isinstance(route, list):
                return route
            return [transform_step(step) for step in route]

        if airspace is not None:
            airspace = spin(airspace, "airspace", _t.transform_bounds)
        queryables = {
            name: spin(q, f"q:{name}", _t.transform_queryable) for name, q in queryables.items()
        }
        regions = [
            replace(
                r,
                bounds=spin(r.bounds, f"s:{r.name}", _t.transform_bounds),
                route=transform_route(r.route),
            )
            for r in spawn.regions
        ]
        # replace(), NOT field-by-field reconstruction: only the transformed
        # fields change, so conflict_free_* flags and any future SpawnConfig
        # field carry into the episode (the old constructor
        # silently dropped conflict-free spawning from grouped episodes).
        spawn = replace(
            spawn,
            regions=regions,
            route=transform_route(spawn.route),
            routes={name: transform_route(rt) for name, rt in spawn.routes.items()},
        )
        return _episode_spec(self, airspace, spawn, queryables)

    def sample(self, rng: np.random.Generator) -> EpisodeSpec:
        self._apply_episode_geometry(rng)
        queryables = self._sample_queryables(rng)
        # Record the episode's whole-geometry rotation draw so tooling (the
        # designer's map preview) can present other geometry - e.g. named
        # regions - in the same episode frame. None when rotation doesn't
        # apply; the groups path records per-group maps instead.
        object.__setattr__(self, "last_rotation", None)
        object.__setattr__(self, "last_group_maps", None)
        if self.groups:
            return self._apply_groups(dict(queryables), rng)
        if self.rotation is None:
            return _episode_spec(self, self.airspace_bounds, self.spawn, queryables)
        angle = _t.sample_scalar(self.rotation["angle"], rng)
        pivot = self.rotation.get("pivot") or _default_pivot(self)
        object.__setattr__(self, "last_rotation", {"pivot": pivot, "angle": angle})
        airspace = (
            _t.rotate_bounds(self.airspace_bounds, pivot, angle)
            if self.airspace_bounds is not None
            else None
        )
        queryables = {
            name: _t.rotate_queryable(q, pivot, angle) for name, q in queryables.items()
        }
        spawn = _t.rotate_spawn(self.spawn, pivot, angle)
        return _episode_spec(self, airspace, spawn, queryables)

    def support(self) -> EpisodeSpec:
        # Restore pristine geometry first: after a sample(), the fields hold
        # the last episode's draw, but support must be episode-independent.
        if self.episode_geometry_fn is not None:
            for name, value in self._pristine_geometry.items():
                object.__setattr__(self, name, value)
        return _episode_spec(self, self.airspace_bounds, self.spawn, self.queryables)


class RegionParamSampler:
    """Per-episode region-shape sampling for scenarios with sampled footprint
    parameters (a radius, half-angle, ... given as a range or distribution).

    Wiring for a scenario (generated task packages emit exactly this shape):

    * ``regions_fn(draw)`` builds the named-region dict for one flat draw,
      keyed ``"<region>.<param>"`` -> value;
    * ``dists`` maps those same keys to samplers - a ``(low, high)`` tuple or
      a scipy distribution with finite ``support()``;
    * ``geometry_fn(regions)`` builds the episode-geometry field dict
      (``airspace_bounds`` / ``spawn`` / ``queryables`` / ``sampled_waypoints``)
      from a regions dict.

    ``episode_geometry`` plugs into :class:`RandomizedScenario`'s
    ``episode_geometry_fn``; ``support_regions()`` supplies the static
    geometry, widening each sampled region to the shapely union of its shapes
    at every parameter-endpoint combination - covering every episode for
    parameters that grow/shrink the shape monotonically (radii, half-angles,
    box edges). Positional parameters (a sampled bearing) are only covered at
    their endpoints; orient with ``rotation`` instead.

    """

    def __init__(
        self,
        regions_fn: Callable[[dict[str, float]], dict[str, Any]],
        dists: dict[str, Any],
        geometry_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        self._regions_fn = regions_fn
        self._dists = dict(dists)
        self._geometry_fn = geometry_fn

    @staticmethod
    def _endpoints(value: Any) -> tuple[float, float]:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return float(value[0]), float(value[1])
        lo, hi = value.support()
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(
                "sampled region params require finite support; use a range or "
                "a bounded distribution"
            )
        return float(lo), float(hi)

    def _representative_draw(self) -> dict[str, float]:
        return {
            key: (lambda e: (e[0] + e[1]) / 2.0)(self._endpoints(value))
            for key, value in self._dists.items()
        }

    def support_regions(self) -> dict[str, Any]:
        """Named regions with every sampled region widened to its union support."""
        from itertools import product

        from bluesky_sandbox.sim.bounds import RegionBounds, union_footprints

        rep = self._representative_draw()
        regions = self._regions_fn(rep)
        by_region: dict[str, list[str]] = {}
        for key in self._dists:
            by_region.setdefault(key.split(".", 1)[0], []).append(key)
        for name, keys in by_region.items():
            keys = sorted(keys)
            if len(keys) > 8:
                raise ValueError(
                    f"region {name!r} samples {len(keys)} footprint params; "
                    "the endpoint-union support caps at 8"
                )
            variants = []
            for combo in product(*(self._endpoints(self._dists[k]) for k in keys)):
                draw = {**rep, **dict(zip(keys, combo))}
                variants.append(self._regions_fn(draw)[name])
            regions[name] = RegionBounds(
                union_footprints([v.footprint for v in variants]),
                variants[0].altitude,
            )
        return regions

    def episode_geometry(self, rng: np.random.Generator) -> dict[str, Any]:
        draw = {k: _t.sample_scalar(v, rng) for k, v in self._dists.items()}
        regions = self._regions_fn(draw)
        return self._geometry_fn(regions)


def _first_not_none(values) -> float | None:
    """First non-``None`` entry of ``values`` as a float, else ``None``."""
    for value in values:
        if value is not None:
            return float(value)
    return None


def _fit_ladder_to_fleet(
    base_ft: float, step_ft: float, rungs: int, cap_ft: float, floor_ft: float
) -> tuple[float, float]:
    """Slide (then, only if forced, compress) a level ladder under ``cap_ft``.

    Lowers ``base`` first, keeping ``step`` intact: the spacing between streams
    is the curriculum variable (``co_altitude_prob`` decides how often it is
    zero), so squashing it to fit would silently turn high draws into extra
    co-altitude episodes and skew that ratio. ``step`` is compressed only when
    the ladder will not fit between ``floor_ft`` and ``cap_ft`` even resting on
    the floor.
    """
    if rungs <= 1:
        return max(min(base_ft, cap_ft), floor_ft), step_ft
    span = (rungs - 1) * step_ft
    if base_ft + span <= cap_ft:
        return base_ft, step_ft
    if cap_ft - span >= floor_ft:
        return cap_ft - span, step_ft
    # Even on the floor the ladder overshoots: flatten it to what fits.
    return floor_ft, max(0.0, (cap_ft - floor_ft) / (rungs - 1))


