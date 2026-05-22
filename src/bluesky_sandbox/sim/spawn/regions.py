"""Spawn regions and the spawn configuration built from them.

:class:`SpawnRegion` draws one aircraft's initial state inside a bounded
volume; :class:`SpawnConfig` collects the regions of an episode and yields
the whole spawn queue. Route sampling for those aircraft lives in
:mod:`.routes`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from bluesky_sandbox.sim.bounds import Bounds
from bluesky_sandbox.sim.performance.envelope import (
    EnvelopeSample,
    feasible_alt_for_type,
)
from bluesky_sandbox.sim.sampling.distributions import (
    CountDistribution,
    ParamDistribution,
    TypeDistribution,
)

from .routes import RouteSpec, RouteStep, sample_route_path

_SPAWN_BOUND_KEYS = ("lat_deg", "lon_deg", "alt_ft", "spd_kts")
# ``alt_ft`` is optional: when omitted, the spawn altitude is sampled from the
# region's bounds altitude band instead. Use ``EnvelopeSample()`` to draw a
# feasible altitude for the selected aircraft type.
# ``hdg_deg`` is optional: when omitted, the initial heading is uniform 0-360.
_SPAWN_PARAM_KEYS = ("alt_ft", "spd_kts", "hdg_deg")
_SPAWN_REQUIRED_PARAM_KEYS = ("spd_kts",)
# Placeholder CAS used to create an envelope-speed aircraft before its true
# feasible CAS is drawn post-creation (BlueSky clamps it if out of envelope).
_ENVELOPE_PROVISIONAL_SPD_KTS = 250.0
# Param keys that aren't spatial/state ranges and so don't contribute to the
# region's ``resolved_bounds`` scalar union.
_NON_BOUND_PARAM_KEYS = ("hdg_deg",)


def _is_envelope_sample(value: object) -> bool:
    return isinstance(value, EnvelopeSample)


def _is_range(value) -> bool:
    return isinstance(value, tuple) and len(value) == 2


def _validate_range(value, *, label: str) -> tuple[float, float]:
    if not _is_range(value):
        raise ValueError(f"{label} must be a (low, high) tuple, got {value!r}.")
    lo, hi = float(value[0]), float(value[1])
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"{label} must be finite, got {value!r}.")
    if lo > hi:
        raise ValueError(f"{label} must satisfy low <= high, got {value!r}.")
    return lo, hi


def _distribution_support(value, *, label: str) -> tuple[float, float]:
    try:
        lo, hi = value.support()
    except Exception as e:
        raise ValueError(
            f"{label} must expose finite support() bounds, got {value!r}."
        ) from e
    lo, hi = float(lo), float(hi)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"{label} has unbounded support: {value!r}.")
    if lo > hi:
        raise ValueError(f"{label} support has low > high: {(lo, hi)!r}.")
    return lo, hi


@dataclass
class SpawnRegion:
    """A spatial region with its own aircraft count and scalar spawn distributions.

    Parameters
    ----------
    bounds:
        Spatial region used to sample ``(lat, lon)`` spawn positions.
    n_aircraft:
        Number of aircraft to spawn in this region each episode.  Either a
        fixed ``int`` or any :class:`CountDistribution` - any frozen
        ``scipy.stats`` integer distribution qualifies.
    params:
        Scalar spawn parameters for this region (``"alt_ft"``, ``"spd_kts"``).
        Both keys are required. Each value is either:

        * a ``(low, high)`` tuple - samples uniformly over ``[low, high]``,
        * or a frozen ``scipy.stats`` continuous distribution - calls
          ``dist.rvs(random_state=rng)`` each spawn.
    aircraft_type:
        Per-region aircraft type override.  Either a fixed ICAO string, any
        :class:`TypeDistribution`, or ``None`` to inherit from
        :class:`SpawnConfig`.
    callsign_prefixes:
        Optional prefix pool for generated callsigns.  Each spawned aircraft
        will receive a callsign of the form ``{prefix}{NNN}`` where ``NNN`` is
        a random three-digit number (e.g. ``"KL042"``).  Accepted values:

        * ``list[str]`` - sample uniformly from the list each spawn.
        * :class:`TypeDistribution` (e.g. :class:`Categorical`) - weighted
          sampling via ``dist.rvs(random_state=rng)``.
        * ``None`` - three random uppercase letters are used (default behaviour).
    spawn_time:
        Per-aircraft scheduled spawn time in seconds since episode reset
        (must be >= 0). The aircraft becomes part of ``bs_traf`` on the first
        ``step()`` whose ``bs.sim.simt`` has reached this value. Accepted values:

        * ``float`` - fixed time for every aircraft in this region.
        * ``(low, high)`` tuple - sample uniformly over ``[low, high]``.
        * frozen ``scipy.stats`` continuous distribution - calls
          ``dist.rvs(random_state=rng)`` each spawn.

        Defaults to ``0.0`` (all aircraft materialise at episode start, the
        previous behaviour).
    route:
        Per-region route override. Use a fixed list of waypoint queryable names,
        a named route key from :class:`SpawnConfig.routes`, or a distribution that
        samples a named route key. ``None`` inherits :class:`SpawnConfig.route`.
    name:
        Optional human-readable label for this region.  Used as the polygon
        label drawn on the radar; falls back to ``"SPAWN i"`` when ``None``.
    render_shape:
        Whether the region polygon is drawn on the map.  ``True`` by default;
        set ``False`` to keep the region as a spawn-only construct with no
        visual outline.  When ``False`` the name is suppressed too (it has
        no shape to anchor to).
    render_name:
        Whether the region's ``name`` is drawn next to its outline.  ``True``
        by default; set ``False`` to draw the polygon without text.
    maintain:
        Steady-state density mode. When ``False`` (default) this region spawns
        its sampled ``n_aircraft`` once and the episode ends when they leave.
        When ``True`` ``n_aircraft`` is reinterpreted as a *target live count*:
        the environment continuously respawns aircraft into this region (each
        clear of existing traffic) so its live occupancy stays at the target,
        and the episode runs until the ``truncated`` hook ends it. Use an
        ``int`` ``n_aircraft`` for a fixed target.
    controlled:
        Whether this region's aircraft are controlled by the policy. ``True``
        (default) spawns controllable agents. ``False`` spawns *background*
        (uncooperative) traffic: they fly their route on autopilot and are still
        seen as intruders by controlled agents, but the policy never commands
        them - for testing avoidance against traffic that does not cooperate.

    Example
    -------
    ::

        from scipy.stats import poisson, truncnorm
        from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
        from bluesky_sandbox.sim.sampling.distributions import Categorical

        SpawnRegion(
            RegionBounds(
                BoxFootprint(
                    lat_min_deg=51.9,
                    lat_max_deg=52.7,
                    lon_min_deg=4.5,
                    lon_max_deg=5.2,
                )
            ),
            n_aircraft=poisson(mu=3),
            params={
                "alt_ft": (5_000, 15_000),
                "spd_kts": truncnorm(a=-2, b=2, loc=250, scale=30),
            },
            callsign_prefixes=Categorical({"KL": 3.0, "BA": 1.0, "DL": 1.0}),
        )
    """
    bounds:            Bounds
    n_aircraft:        int | CountDistribution
    params:            dict[str, tuple[float, float] | ParamDistribution] = field(default_factory=dict)
    aircraft_type:     str | TypeDistribution | None = field(default=None)
    callsign_prefixes: list[str] | TypeDistribution | None = field(default=None)
    spawn_time:        float | tuple[float, float] | ParamDistribution = 0.0
    route:             RouteSpec | str | TypeDistribution | None = None
    name:              str | None = field(default=None)
    render_shape:      bool = True
    render_name:       bool = True
    maintain:          bool = False
    controlled:        bool = True
    # Per-region override of conflict-free spawning. ``None`` inherits the
    # ``SpawnConfig.conflict_free_spawn`` default (like ``aircraft_type``/``route``
    # fall back to their config-level globals); ``True``/``False`` force it for
    # this region's aircraft regardless of the global.
    conflict_free_spawn: bool | None = None
    # Per-region conflict-free spawn buffer, added on top of the protected zone
    # for the spawn-clear check: a candidate is rejected if its predicted CPA
    # comes within ``PZ + margin`` in space (horizontal nm / vertical ft) or
    # within ``lookahead + margin`` in time (s), giving the episode headroom that
    # survives the first steps as aircraft begin to maneuver. Each ``None``
    # inherits the ``SpawnConfig`` default.
    conflict_free_margin_nm: float | None = None
    conflict_free_margin_ft: float | None = None
    conflict_free_margin_s: float | None = None

    def __post_init__(self) -> None:
        invalid = [k for k in self.params if k not in _SPAWN_PARAM_KEYS]
        if invalid:
            raise ValueError(
                f"SpawnRegion.params contains invalid keys: {invalid}. "
                f"Allowed: {_SPAWN_PARAM_KEYS}"
            )
        missing = [k for k in _SPAWN_REQUIRED_PARAM_KEYS if k not in self.params]
        if missing:
            raise ValueError(
                f"SpawnRegion.params must define required keys: {missing}."
            )
        # Spawn altitude comes from params['alt_ft'] if given, else from the
        # bounds altitude band - one of the two must provide it.
        if "alt_ft" not in self.params and self._finite_alt_band() is None:
            raise ValueError(
                "SpawnRegion needs a spawn altitude: set params['alt_ft'] or "
                "give the bounds a finite altitude band."
            )
        if isinstance(self.n_aircraft, int) and self.n_aircraft < 0:
            raise ValueError(
                f"SpawnRegion.n_aircraft must be >= 0, got {self.n_aircraft}."
            )
        for key, value in self.params.items():
            label = f"SpawnRegion.params[{key!r}]"
            if _is_envelope_sample(value):
                if key not in ("alt_ft", "spd_kts"):
                    raise ValueError(
                        f"{label} can only use EnvelopeSample for 'alt_ft' or "
                        "'spd_kts'."
                    )
                continue
            if isinstance(value, (int, float)):
                if not math.isfinite(float(value)):
                    raise ValueError(f"{label} must be finite, got {value!r}.")
            elif isinstance(value, tuple):
                # Heading wraps through north, so a (low, high) with low > high
                # (e.g. 350 -> 10) is a valid circular arc.
                if key == "hdg_deg":
                    lo, hi = float(value[0]), float(value[1])
                    if not (math.isfinite(lo) and math.isfinite(hi)):
                        raise ValueError(f"{label} must be finite, got {value!r}.")
                else:
                    _validate_range(value, label=label)
            else:
                _distribution_support(value, label=label)
        if isinstance(self.callsign_prefixes, list) and not self.callsign_prefixes:
            raise ValueError("SpawnRegion.callsign_prefixes must not be empty.")
        if isinstance(self.spawn_time, (int, float)):
            if self.spawn_time < 0:
                raise ValueError(
                    f"SpawnRegion.spawn_time must be >= 0, got {self.spawn_time}."
                )
        elif isinstance(self.spawn_time, tuple):
            lo, _ = _validate_range(
                self.spawn_time, label="SpawnRegion.spawn_time"
            )
            if lo < 0:
                raise ValueError(
                    f"SpawnRegion.spawn_time tuple must have low >= 0, got {self.spawn_time!r}."
                )
        else:
            _distribution_support(self.spawn_time, label="SpawnRegion.spawn_time")

    def sample_n(self, rng: np.random.Generator) -> int:
        """Return the number of aircraft to spawn in this region."""
        if isinstance(self.n_aircraft, int):
            return self.n_aircraft
        n = int(self.n_aircraft.rvs(random_state=rng))
        if n < 0:
            raise ValueError(
                f"SpawnRegion.n_aircraft sampled a negative count: {n}."
            )
        return n

    def max_n(self) -> int:
        """Deterministic upper bound on ``sample_n``.

        For an ``int`` ``n_aircraft`` this is the int itself. For a frozen
        scipy distribution it's ``support()[1]``; this fails loudly on
        unbounded distributions (e.g. ``poisson``) since callers depend on
        a finite cap to size observation spaces.
        """
        if isinstance(self.n_aircraft, int):
            return self.n_aircraft
        lo, hi_float = _distribution_support(
            self.n_aircraft, label="SpawnRegion.n_aircraft"
        )
        if lo < 0:
            raise ValueError(
                f"SpawnRegion.n_aircraft support must be non-negative, got {(lo, hi_float)!r}."
            )
        hi = int(hi_float)
        return hi

    def sample_type(self, rng: np.random.Generator) -> str:
        """Return a sampled aircraft type for this region."""
        if isinstance(self.aircraft_type, str):
            return self.aircraft_type
        return self.aircraft_type.rvs(random_state=rng)

    def sample_callsign_prefix(self, rng: np.random.Generator) -> str | None:
        """Return a callsign prefix string, or ``None`` to use random letters.

        * ``list[str]`` - samples uniformly from the pool.
        * :class:`TypeDistribution` (e.g. :class:`Categorical`) - calls
          ``dist.rvs(random_state=rng)`` for weighted sampling.
        * ``None`` - returns ``None``; the environment generates random letters.
        """
        if self.callsign_prefixes is None:
            return None
        if isinstance(self.callsign_prefixes, list):
            idx = int(rng.integers(len(self.callsign_prefixes)))
            return self.callsign_prefixes[idx]
        return self.callsign_prefixes.rvs(random_state=rng)

    def sample_spawn_time(self, rng: np.random.Generator) -> float:
        """Sample one aircraft's scheduled spawn time (seconds since reset).

        Negative samples (possible with unbounded distributions) are clamped
        to ``0.0`` so the aircraft simply materialises at episode start.
        """
        if isinstance(self.spawn_time, (int, float)):
            t = float(self.spawn_time)
        elif isinstance(self.spawn_time, tuple):
            t = float(rng.uniform(self.spawn_time[0], self.spawn_time[1]))
        else:
            t = float(self.spawn_time.rvs(random_state=rng))
        if not math.isfinite(t):
            raise ValueError(
                f"SpawnRegion.spawn_time sampled a non-finite value: {t!r}."
            )
        return max(0.0, t)

    def _finite_alt_band(self) -> tuple[float, float] | None:
        """The bounds' altitude band, when it is finite; otherwise ``None``."""
        lo = getattr(self.bounds, "alt_min_ft", None)
        hi = getattr(self.bounds, "alt_max_ft", None)
        if lo is None or hi is None or not (math.isfinite(lo) and math.isfinite(hi)):
            return None
        return float(lo), float(hi)

    def sample_pos(
        self,
        rng: np.random.Generator,
        actype: str | None = None,
    ) -> dict[str, float]:
        """Sample spawn parameters for one aircraft in this region."""
        pos = {}
        for k, v in self.params.items():
            if _is_envelope_sample(v):
                if actype is None:
                    raise ValueError(
                        "SpawnRegion.sample_pos needs an aircraft type when a "
                        "param uses EnvelopeSample()."
                    )
                if k == "spd_kts":
                    # The feasible CAS depends on the live aircraft's vmin (only
                    # known post-creation) and the spawn altitude, so defer it:
                    # flag it and store a feasible provisional for the probe cre.
                    pos["_spd_from_envelope"] = True
                    pos[k] = _ENVELOPE_PROVISIONAL_SPD_KTS
                else:  # alt_ft
                    band = getattr(self.bounds, "alt_band_at", None)
                    alt_min_ft = alt_max_ft = None
                    if band is not None:
                        lat_for_band, lon_for_band = self.bounds.sample_point(rng)
                        alt_min_ft, alt_max_ft = band(lat_for_band, lon_for_band)
                        pos["lat_deg"], pos["lon_deg"] = lat_for_band, lon_for_band
                    pos[k] = feasible_alt_for_type(
                        actype,
                        rng,
                        v.alt_floor_ft,
                        alt_min_ft=alt_min_ft,
                        alt_max_ft=alt_max_ft,
                    )
            elif isinstance(v, (int, float)):
                pos[k] = float(v)
            elif isinstance(v, tuple):
                lo, hi = float(v[0]), float(v[1])
                # Heading: a low > high range wraps through north (sample over
                # [low, high+360) then fold back below).
                if k == "hdg_deg" and lo > hi:
                    hi += 360.0
                pos[k] = float(rng.uniform(lo, hi))
            else:
                pos[k] = float(v.rvs(random_state=rng))
            if k == "hdg_deg":
                pos[k] %= 360.0
            if not math.isfinite(pos[k]):
                raise ValueError(
                    f"SpawnRegion.params[{k!r}] sampled a non-finite value: {pos[k]!r}."
                )
        if "lat_deg" not in pos or "lon_deg" not in pos:
            pos["lat_deg"], pos["lon_deg"] = self.bounds.sample_point(rng)
        # When params doesn't pin the altitude, draw it from the bounds band so
        # spawn altitude follows the region geometry (no duplicated alt range).
        if "alt_ft" not in pos:
            band = getattr(self.bounds, "alt_band_at", None)
            lo, hi = band(pos["lat_deg"], pos["lon_deg"]) if band else self._finite_alt_band()
            pos["alt_ft"] = float(rng.uniform(lo, hi)) if hi > lo else float(lo)
        return pos


@dataclass
class SpawnConfig:
    """Defines how aircraft are spawned at each ``reset()``.

    All spawn behaviour is driven by :class:`SpawnRegion` objects.  Each
    region spawns independently with its own aircraft count, spatial bounds,
    scalar parameter distributions, and optionally its own aircraft type.

    Parameters
    ----------
    regions:
        Zero or more :class:`SpawnRegion` objects.  Each region is sampled
        independently every episode. An empty list is useful while designing
        an environment and produces no spawned aircraft.
    aircraft_type:
        Global fallback aircraft type.  Used for any region that does not
        specify its own ``aircraft_type``.  Either a fixed ICAO string, any
        :class:`TypeDistribution` (e.g. :class:`Categorical`), or ``None``
        for uniform sampling across the environment's allowed aircraft.
    route:
        Global fallback route. Use a fixed list of waypoint queryable names,
        a named route key from ``routes``, or a distribution that samples one
        of those route keys.
    routes:
        Optional named route library used by string/distribution route specs.
    conflict_free_spawn:
        Global default for conflict-free spawning, applied to any region that
        leaves its own ``SpawnRegion.conflict_free_spawn`` unset (``None``). When
        effective, an aircraft is spawned clear of a predicted conflict: the
        environment rejects a candidate spawn state whose straight-line closest
        approach with any existing aircraft would breach the protected zone
        within BlueSky's lookahead (matching state-based CD), resampling until
        clear. Guarantees the intrinsic conflict cost starts at zero, so a
        conflict-free policy is achievable; conflicts still *develop* as aircraft
        converge. Default ``False`` (spawn as sampled - e.g. the conflict tests).
        Set it per region to mix conflict-free and as-sampled spawn areas.
    """

    regions:       list[SpawnRegion]
    aircraft_type: str | TypeDistribution | None = None
    route:         RouteSpec | str | TypeDistribution | None = None
    routes:        dict[str, RouteSpec] = field(default_factory=dict)
    conflict_free_spawn: bool = False
    # Conflict-free spawn buffer added on top of the protected zone: horizontal
    # (nm) and vertical (ft) separation plus time (s) on the prediction horizon
    # (``lookahead + margin_s``). A positive margin makes the "conflict-free"
    # guarantee hold with headroom, so it does not decay the instant aircraft
    # start turning toward their routes; ``0`` on all three reproduces the
    # bare-PZ check. Applied to any region leaving its own override unset.
    conflict_free_margin_nm: float = 0.0
    conflict_free_margin_ft: float = 0.0
    conflict_free_margin_s: float = 0.0
    def __post_init__(self) -> None:
        self.regions = list(self.regions)

    def region_conflict_free(self, index: int) -> bool:
        """Effective conflict-free-spawn flag for region ``index``.

        The region's own ``conflict_free_spawn`` wins when set; otherwise the
        config-level ``conflict_free_spawn`` default applies (mirroring how
        ``aircraft_type``/``route`` fall back to their globals).
        """
        override = self.regions[index].conflict_free_spawn
        return self.conflict_free_spawn if override is None else bool(override)

    def region_conflict_margins(self, index: int) -> tuple[float, float, float]:
        """Effective conflict-free spawn buffer for region ``index``.

        Returns ``(margin_nm, margin_ft, margin_s)`` - spatial (nm/ft) and
        temporal (s) buffers over the protected zone. Each region field wins when
        set, else the config-level default applies (mirroring
        ``region_conflict_free``).
        """
        r = self.regions[index]
        margin_nm = (
            self.conflict_free_margin_nm
            if r.conflict_free_margin_nm is None
            else r.conflict_free_margin_nm
        )
        margin_ft = (
            self.conflict_free_margin_ft
            if r.conflict_free_margin_ft is None
            else r.conflict_free_margin_ft
        )
        margin_s = (
            self.conflict_free_margin_s
            if r.conflict_free_margin_s is None
            else r.conflict_free_margin_s
        )
        return float(margin_nm), float(margin_ft), float(margin_s)

    @property
    def resolved_bounds(self) -> dict[str, tuple[float, float]]:
        """Union bounding box across all regions.

        - ``lat_deg`` / ``lon_deg``: enclosing box of all region spatial bounds.
        - Scalar keys (``alt_ft``, ``spd_kts``): union of all region ``params``
          ranges.
        """
        if not self.regions:
            return {
                "lat_deg": (0.0, 0.0),
                "lon_deg": (0.0, 0.0),
                "alt_ft": (0.0, 0.0),
                "spd_kts": (0.0, 0.0),
            }
        lat_mins, lat_maxs, lon_mins, lon_maxs = [], [], [], []
        scalar_lows: dict[str, list[float]] = {}
        scalar_highs: dict[str, list[float]] = {}
        for r in self.regions:
            (r_lat_min, r_lat_max), (r_lon_min, r_lon_max) = r.bounds.bounding_box
            lat_mins.append(r_lat_min)
            lat_maxs.append(r_lat_max)
            lon_mins.append(r_lon_min)
            lon_maxs.append(r_lon_max)
            for k, v in r.params.items():
                if k in _NON_BOUND_PARAM_KEYS:
                    continue
                if _is_envelope_sample(v):
                    lo = float(v.alt_floor_ft)
                    hi = 60_000.0
                elif isinstance(v, (int, float)):
                    lo = hi = float(v)
                elif isinstance(v, tuple):
                    lo, hi = _validate_range(v, label=f"SpawnRegion.params[{k!r}]")
                else:
                    lo, hi = _distribution_support(
                        v, label=f"SpawnRegion.params[{k!r}]"
                    )
                scalar_lows.setdefault(k, []).append(lo)
                scalar_highs.setdefault(k, []).append(hi)
            # Spawn altitude follows the bounds band when params has no alt_ft.
            if "alt_ft" not in r.params:
                band = r._finite_alt_band()
                if band is not None:
                    scalar_lows.setdefault("alt_ft", []).append(band[0])
                    scalar_highs.setdefault("alt_ft", []).append(band[1])
        result = {
            "lat_deg": (min(lat_mins), max(lat_maxs)),
            "lon_deg": (min(lon_mins), max(lon_maxs)),
        }
        for k in scalar_lows:
            result[k] = (min(scalar_lows[k]), max(scalar_highs[k]))
        return result

    def sample_type(self, rng: np.random.Generator) -> str:
        """Return a sampled aircraft type string (global fallback)."""
        if isinstance(self.aircraft_type, str):
            return self.aircraft_type
        if self.aircraft_type is not None:
            return self.aircraft_type.rvs(random_state=rng)
        raise ValueError(
            "aircraft_type is None; normalize SpawnConfig before use."
        )

    def sample_route(
        self,
        rng: np.random.Generator,
        region: SpawnRegion,
    ) -> list[RouteStep] | None:
        """Return the route sampled for one aircraft, or ``None``.

        Subroute references are expanded inline and each ``{"choice": ...}``
        branch is sampled (see :func:`sample_route_path`), yielding one concrete
        ordered list of steps (bare waypoint names, plus ``{"waypoint": ...}``
        steps for any per-step crossing restrictions).
        """
        spec = region.route if region.route is not None else self.route
        if spec is None:
            return None
        if isinstance(spec, (list, tuple)):
            return sample_route_path(spec, self.routes, rng)
        key = spec if isinstance(spec, str) else spec.rvs(random_state=rng)
        if isinstance(key, (list, tuple)):
            return sample_route_path(key, self.routes, rng)
        if key not in self.routes:
            raise ValueError(
                f"Spawn route sampled unknown route key {key!r}; "
                f"available routes: {list(self.routes)}"
            )
        return sample_route_path(self.routes[key], self.routes, rng)

    def max_aircraft(self) -> int:
        """Deterministic upper bound on the total aircraft count per episode.

        Sums :meth:`SpawnRegion.max_n` across all regions. Used by the base env
        to size observation spaces with intruder padding.
        """
        return sum(r.max_n() for r in self.regions)

    def iter_spawns(
        self,
        rng: np.random.Generator,
        *,
        limit: int | None = None,
        include_maintain: bool = False,
    ) -> Iterator[tuple[int, float, str, dict[str, float], str | None, list[RouteStep] | None]]:
        """Yield ``(spawn_time, actype, pos, callsign_prefix, route)`` per aircraft.

        ``spawn_time`` is seconds since episode reset; the base env materialises
        each aircraft on the first ``step()`` whose ``bs.sim.simt`` has reached
        this value. Defaults to ``0.0`` per region (immediate spawn).

        ``callsign_prefix`` is a string prefix (e.g. ``"KL"``) when the region
        specifies :attr:`SpawnRegion.callsign_prefixes`, otherwise ``None``
        (the environment will generate random letters).

        ``route`` is an optional list of waypoint queryable names sampled from
        the spawn config for this aircraft.
        """
        counts: dict[int, int] = {}
        for region_index, region in enumerate(self.regions):
            # Maintain regions are filled and topped-up by the runtime's
            # steady-state path (separation-guarded), not the one-shot queue.
            # ``include_maintain`` lets the designer preview still show their
            # target aircraft as a representative snapshot.
            if region.maintain and not include_maintain:
                continue
            counts[region_index] = max(region.sample_n(rng), 0)
        # Never an empty episode: a region's count draw may legitimately come
        # out 0, so floor the total at 1 aircraft.
        if counts and sum(counts.values()) == 0:
            counts[int(rng.choice(list(counts)))] = 1
        candidates = []
        for region_index, n in counts.items():
            for _ in range(n):
                candidates.append((region_index, *self.sample_region_spawn(region_index, rng)))
        if limit is not None:
            limit = int(limit)
            if limit < 0:
                raise ValueError(f"SpawnConfig.iter_spawns limit must be >= 0, got {limit}")
            order = rng.permutation(len(candidates))
            candidates = [candidates[int(i)] for i in order[:limit]]
        yield from candidates

    def sample_region_spawn(
        self,
        region_index: int,
        rng: np.random.Generator,
    ) -> tuple[float, str, dict[str, float], str | None, list[RouteStep] | None]:
        """Sample one ``(spawn_time, actype, pos, prefix, route)`` for a region.

        Shared by :meth:`iter_spawns` (episode-start fill) and the runtime's
        steady-state ``maintain`` top-up, so a respawn is drawn exactly like an
        initial spawn for that region.
        """
        region = self.regions[region_index]
        actype = (
            region.sample_type(rng)
            if region.aircraft_type is not None
            else self.sample_type(rng)
        )
        prefix = region.sample_callsign_prefix(rng)
        route = self.sample_route(rng, region)
        t = region.sample_spawn_time(rng)
        return (t, actype, region.sample_pos(rng, actype), prefix, route)

    @property
    def has_maintain_regions(self) -> bool:
        """True when any region runs in steady-state ``maintain`` mode."""
        return any(region.maintain for region in self.regions)
