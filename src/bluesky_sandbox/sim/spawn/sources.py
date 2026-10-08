"""Spawn sources: where an episode's aircraft come from, beyond spawn regions.

A source *plans* the episode at reset: it returns the aircraft to spawn, each a
time-stamped :class:`SpawnRequest`. The environment queues every source's plan
with the regions' spawns and creates each aircraft when its time comes, the
same way it creates a region's - callsign, route, hooks, control state, log.
What the data is, and how it is read, is the source's own business: a file of
ADS-B tracks, a fitted distribution, a script.

Each source carries what applies to its aircraft whatever their origin:

- ``conflict_free`` - only spawn one clear of live traffic over CD's lookahead;
- ``when_blocked`` - what to do with one that is not: ``"resample"`` (draw
  another, where the source can; else wait), ``"defer"`` (wait until clear),
  ``"skip"`` (drop it) or ``"allow"`` (spawn it anyway - the same as not
  conflict-free: a faithful replay of data that had conflicts);
- ``route`` - the route of a request that names none (steps, or a key into the
  spawn config's ``routes``); ``assign_route`` - or a function choosing one,
  such as :func:`nearest_entry`;
- ``max_aircraft`` - the most it plans, counted into the episode's max
  aircraft (``None``: not counted - as for code spawns, ``aircraft_cap`` sizes
  the episode then).

Built in: :class:`Replay` (requests your code made, scheduled), :class:`Mixture`
(each aircraft from one of several sources, by weight) and
:class:`PlannedSource` (a function returning the plan). A source of your own
subclasses :class:`SpawnSource` and defines :meth:`~SpawnSource.plan`.

Reacting to the running episode instead - spawning when something happens -
is ``env.spawn`` from a hook.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

from bluesky_sandbox.sim.bounds import Bounds, LatLon

from .routes import expand_route_paths

__all__ = [
    "Mixture",
    "PlanContext",
    "PlannedSource",
    "PlannedSpawn",
    "RegionSource",
    "Replay",
    "SpawnRequest",
    "SpawnSource",
    "nearest_entry",
]

WhenBlocked = Literal["resample", "defer", "skip", "allow"]


@dataclass(frozen=True)
class SpawnRequest:
    """One aircraft to spawn: where, at what state, and when.

    ``spd_kts`` - CAS; ``None`` draws one its flight envelope allows at
    ``alt_ft``. ``hdg_deg`` - ``None`` draws one. ``actype`` - ``None`` draws
    the spawn config's aircraft type. ``route`` - steps (waypoint names, step
    dicts) or a key into the spawn config's ``routes``; ``None`` leaves it to
    the source. ``time_s`` - seconds into the episode. ``controlled`` - ``None``
    leaves it to ``define_initial_aircraft_control_state``. ``callsign_prefix``
    - the airline its callsign is issued under (``"KLM"``), when it names no
    ``callsign``; ``None`` issues random letters."""

    at: LatLon
    alt_ft: float
    spd_kts: float | None = None
    hdg_deg: float | None = None
    actype: str | None = None
    callsign: str | None = None
    route: Any = None
    time_s: float = 0.0
    controlled: bool | None = None
    callsign_prefix: str | None = None


@dataclass(frozen=True)
class PlannedSpawn:
    """One aircraft of an episode's plan: its request - type drawn, route
    resolved - and where it came from: a spawn region (``region_index``) or a
    source (``source_index``); -1 for neither."""

    request: SpawnRequest
    region_index: int = -1
    source_index: int = -1


@dataclass(frozen=True)
class PlanContext:
    """What a source plans from: the episode's geometry, as it is this episode."""

    #: The design's shapes - areas and points - by name.
    shapes: Mapping[str, Bounds] = field(default_factory=dict)
    #: The queryables (waypoints, regions), by name.
    queryables: Mapping[str, Any] = field(default_factory=dict)
    #: The airspace's shape, or None.
    airspace: Bounds | None = None
    #: The spawn config: its route library, aircraft types, regions.
    spawn: Any = None


class SpawnSource(ABC):
    """Plans an episode's aircraft at reset. See the module docstring."""

    name: str = ""
    conflict_free: bool = False
    when_blocked: WhenBlocked = "defer"
    route: Any = None
    assign_route: Callable[[SpawnRequest, PlanContext], Any] | None = None
    max_aircraft: int | None = None

    @abstractmethod
    def plan(self, rng: np.random.Generator, ctx: PlanContext) -> list[SpawnRequest]:
        """This episode's aircraft, each with its time."""

    def redraw(
        self, request: SpawnRequest, rng: np.random.Generator, ctx: PlanContext
    ) -> SpawnRequest | None:
        """Another request in place of ``request`` (blocked; ``"resample"``),
        or ``None`` when the source cannot draw again - it is deferred then."""
        return None

    def draw(self, rng: np.random.Generator, ctx: PlanContext) -> SpawnRequest | None:
        """One request, as from this source's distribution - for a
        :class:`Mixture`. By default one of its plan's, at random."""
        planned = self.plan(rng, ctx)
        return planned[int(rng.integers(len(planned)))] if planned else None

    def route_for(self, request: SpawnRequest, ctx: PlanContext) -> Any:
        """The route ``request`` flies: its own, else the source's, else the
        one ``assign_route`` chooses."""
        if request.route is not None:
            return request.route
        if self.route is not None:
            return self.route
        if self.assign_route is not None:
            return self.assign_route(request, ctx)
        return None

    def _policies(self, **kwargs: Any) -> None:
        for name in ("name", "conflict_free", "when_blocked", "route", "assign_route", "max_aircraft"):
            if name in kwargs:
                setattr(self, name, kwargs.pop(name))
        if kwargs:
            raise TypeError(f"{type(self).__name__}() got unexpected arguments {sorted(kwargs)}")
        if self.when_blocked not in ("resample", "defer", "skip", "allow"):
            raise ValueError(
                f"when_blocked must be 'resample', 'defer', 'skip' or 'allow', got {self.when_blocked!r}"
            )


class Replay(SpawnSource):
    """Requests your code made - from any file, any format - spawned at their
    times. ``window`` - ``(start_s, end_s)``: only those within it, its start
    becoming the episode's 0. ``time_scale`` - times multiplied by it (0.5
    plays twice as fast). Blocked spawns wait by default: a recorded position
    cannot be drawn again."""

    def __init__(
        self,
        requests: Sequence[SpawnRequest],
        *,
        window: tuple[float, float] | None = None,
        time_scale: float = 1.0,
        **policies: Any,
    ) -> None:
        self.requests = list(requests)
        self.window = window
        self.time_scale = float(time_scale)
        self._policies(**policies)
        if self.max_aircraft is None:
            self.max_aircraft = len(self._in_window())

    def _in_window(self) -> list[SpawnRequest]:
        if self.window is None:
            return list(self.requests)
        start, end = self.window
        return [r for r in self.requests if start <= r.time_s < end]

    def plan(self, rng: np.random.Generator, ctx: PlanContext) -> list[SpawnRequest]:
        start = 0.0 if self.window is None else float(self.window[0])
        return [replace(r, time_s=(r.time_s - start) * self.time_scale) for r in self._in_window()]


class PlannedSource(SpawnSource):
    """A source from a function: ``plan(rng, ctx) -> [SpawnRequest, ...]``."""

    def __init__(self, plan: Callable[[np.random.Generator, PlanContext], Sequence[SpawnRequest]], **policies: Any) -> None:
        self._plan = plan
        self._policies(**policies)

    def plan(self, rng: np.random.Generator, ctx: PlanContext) -> list[SpawnRequest]:
        return list(self._plan(rng, ctx) or [])


class Mixture(SpawnSource):
    """``n`` aircraft (a count, or a ``(low, high)`` range drawn each episode),
    each from one of ``sources`` chosen by ``weights`` - drawn from it as its
    own are (:meth:`SpawnSource.draw`). Blocked spawns are drawn again."""

    def __init__(
        self,
        sources: Sequence[SpawnSource],
        weights: Sequence[float] | None = None,
        n: int | tuple[int, int] = 1,
        **policies: Any,
    ) -> None:
        self.sources = list(sources)
        if not self.sources:
            raise ValueError("Mixture needs at least one source")
        w = np.ones(len(self.sources)) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != (len(self.sources),) or (w < 0).any() or w.sum() <= 0:
            raise ValueError("Mixture weights: one non-negative weight per source, not all 0")
        self.weights = w / w.sum()
        self.n = n
        self.when_blocked = "resample"
        self._policies(**policies)
        if self.max_aircraft is None:
            self.max_aircraft = int(n[1] if isinstance(n, tuple) else n)

    def _one(self, rng: np.random.Generator, ctx: PlanContext) -> SpawnRequest | None:
        source = self.sources[int(rng.choice(len(self.sources), p=self.weights))]
        request = source.draw(rng, ctx)
        if request is not None and request.route is None:
            # A component's own route rules travel with its aircraft.
            request = replace(request, route=source.route_for(request, ctx))
        return request

    def plan(self, rng: np.random.Generator, ctx: PlanContext) -> list[SpawnRequest]:
        n = self.n
        count = int(rng.integers(n[0], n[1] + 1)) if isinstance(n, tuple) else int(n)
        out = [r for r in (self._one(rng, ctx) for _ in range(count)) if r is not None]
        return out

    def redraw(self, request: SpawnRequest, rng: np.random.Generator, ctx: PlanContext) -> SpawnRequest | None:
        again = self._one(rng, ctx)
        return None if again is None else replace(again, time_s=request.time_s)

    def draw(self, rng: np.random.Generator, ctx: PlanContext) -> SpawnRequest | None:
        return self._one(rng, ctx)


class RegionSource(SpawnSource):
    """Aircraft drawn as ``region`` (a :class:`SpawnRegion`) draws its own -
    position, altitude, speed, heading, type, route - its count each episode.
    For a :class:`Mixture` of random traffic and your own: the region need not
    be one of the spawn config's. Conflict-free as the region is; blocked
    spawns are drawn again."""

    def __init__(self, region: Any, **policies: Any) -> None:
        self.region = region
        self.conflict_free = bool(getattr(region, "conflict_free_spawn", None) or False)
        self.when_blocked = "resample"
        self.name = getattr(region, "name", "") or ""
        self._policies(**policies)
        if self.max_aircraft is None:
            self.max_aircraft = int(region.max_n())

    def draw(self, rng: np.random.Generator, ctx: PlanContext) -> SpawnRequest | None:
        if ctx.spawn is None:
            raise ValueError("RegionSource draws with the spawn config: plan with one in the context")
        return ctx.spawn.draw_request(self.region, rng)

    def plan(self, rng: np.random.Generator, ctx: PlanContext) -> list[SpawnRequest]:
        n = max(int(self.region.sample_n(rng)), 0)
        return [r for r in (self.draw(rng, ctx) for _ in range(n)) if r is not None]

    def redraw(self, request: SpawnRequest, rng: np.random.Generator, ctx: PlanContext) -> SpawnRequest | None:
        again = self.draw(rng, ctx)
        return None if again is None else replace(again, time_s=request.time_s)


def nearest_entry(request: SpawnRequest, ctx: PlanContext) -> str | None:
    """The route (a key into the spawn config's ``routes``) whose first fix is
    nearest the aircraft - among those ahead of it, when it has a heading (the
    fix within 90 degrees of where it points). ``None`` with no route to give.

    For data with no routes of its own - ADS-B tracks: each aircraft joins the
    procedure it is entering."""
    routes: Mapping[str, Any] = getattr(ctx.spawn, "routes", None) or {}
    best: tuple[float, str] | None = None
    for key, spec in routes.items():
        for path in expand_route_paths(spec, routes):
            if not path:
                continue
            fix = ctx.queryables.get(path[0])
            lat, lon = getattr(fix, "lat", None), getattr(fix, "lon", None)
            if lat is None or lon is None:
                continue
            north = (lat - request.at.lat_deg) * 60.0
            east = (lon - request.at.lon_deg) * 60.0 * math.cos(math.radians(request.at.lat_deg))
            if request.hdg_deg is not None:
                bearing = math.degrees(math.atan2(east, north)) % 360.0
                if abs((bearing - request.hdg_deg + 180.0) % 360.0 - 180.0) > 90.0:
                    continue
            distance = math.hypot(north, east)
            if best is None or distance < best[0]:
                best = (distance, key)
    return None if best is None else best[1]
