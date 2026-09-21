"""Spawning: sampling, clearing and materialising an episode's aircraft.

The environment owns the episode; this owns *how aircraft enter it*. Four
lifecycle calls, all driven from ``reset``/``step``:

* :meth:`begin_episode` - clear the queue and fix each ``maintain`` region's
  target live count for the episode.
* :meth:`schedule_episode` - sample the one-shot spawn queue, sorted by
  ``spawn_time``, and snapshot its size for the progress denominator.
* :meth:`drain` - materialise every queued aircraft whose time has arrived.
* :meth:`maintain` - top steady-state regions back up to their target.

State split. This class owns the queue, the per-region failure counts and the
maintain targets - nothing outside spawning mutates them, and the three
read-only progress queries the environment exposes (``episode_done``,
``has_future_agents``, ``episode_spawn_progress``) are served from here.
Per-aircraft lifecycle registries (``_aircraft_spawn_time``,
``_aircraft_region``) stay on the environment: they outlive spawning and are
purged on deletion, so this writes them through ``self.env`` the same way it
reaches ``_runtime``, ``_hooks`` and the query-state monitor.
"""

from __future__ import annotations

import string
import warnings
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from bluesky.tools.aero import ft, nm

from bluesky_sandbox.sim.geometry.clearance import (
    inside_separation_zone,
    predicted_conflict,
)
from bluesky_sandbox.sim.geometry.conflict import cd_hpz_m, cd_rpz_m
from bluesky_sandbox.sim.performance.envelope import (
    feasible_alt_cas,
    feasible_cas_at_alt,
    reachable_alt_window,
)
from bluesky_sandbox.sim.queryables import Waypoint, WaypointTarget
from bluesky_sandbox.sim.spawn import route_step_name

from .state import (
    AircraftControlState,
    Callsign,
    ResolvedRoute,
    SpawnPosition,
    SpawnQueueItem,
)

_ALPHABET = list(string.ascii_uppercase)

# A candidate spawn state must clear all live traffic (avoiding an instant loss
# of separation), retried up to this many times before deferring to a later
# step. The zone itself is the live CD protected zone plus the region's
# ``spawn_sep_*``, so it tracks whatever the scenario actually detects
# conflicts against.
_MAINTAIN_SPAWN_MAX_TRIES = 20
# Consecutive fully-failed spawns for one region before warning once. A single
# failure is normal; a sustained failure means the guard is unsatisfiable and
# the region is silently under-populated.
_MAINTAIN_SPAWN_WARN_AFTER = 5


class SpawnGenerator:
    """Sample, clear and materialise the aircraft that enter an episode."""

    def __init__(self, env=None) -> None:
        self.env = env
        self._queue: list = []
        self._maintain_failures: dict[int, int] = {}
        self._maintain_target: dict[int, int] = {}
        self._scheduled_count: int = 0

    def bind_env(self, env) -> None:
        self.env = env

    def _config(self):
        if self.env is None:
            raise RuntimeError("SpawnGenerator env has not been set.")
        return self.env.config

    # ---- lifecycle ------------------------------------------------------- #

    def begin_episode(self, rng: np.random.Generator) -> None:
        """Drop the previous episode's queue and fix this one's maintain targets.

        Runs before ``bs.sim.reset()``: it only reads the spawn config, and the
        environment needs the targets in place before any traffic exists.
        """
        self._queue.clear()
        self._maintain_failures.clear()
        self._sample_maintain_targets(rng)

    def schedule_episode(self, rng: np.random.Generator) -> None:
        """Sample the episode's one-shot spawns and snapshot the total.

        The snapshot is taken before any drain so the HUD's ``spawned X/N``
        denominator is stable for the whole episode.
        """
        self._enqueue_spawns(rng)
        self._scheduled_count = len(self._queue)

    def drain(self, rng: np.random.Generator) -> None:
        """Materialise every queued aircraft whose ``spawn_time`` has arrived."""
        self._drain_spawn_queue(rng)

    def maintain(self, rng: np.random.Generator) -> None:
        """Top up each steady-state region to its target live count."""
        self._maintain_spawns(rng)

    # ---- progress queries ------------------------------------------------ #

    @property
    def pending(self) -> int:
        """Aircraft still queued to spawn."""
        return len(self._queue)

    @property
    def has_maintain(self) -> bool:
        """True when any region replenishes itself, so the queue never empties."""
        return bool(self._maintain_target)

    @property
    def scheduled_count(self) -> int:
        """Total one-shot spawns sampled for this episode (fixed at reset)."""
        return self._scheduled_count

    # ---- internals ------------------------------------------------------- #

    def _generate_callsign(
        self,
        used: set[Callsign],
        rng: np.random.Generator,
        prefix: str | None = None,
    ) -> Callsign:
        while True:
            if prefix is None:
                letters = "".join(rng.choice(_ALPHABET, size=3))
            else:
                letters = prefix
            callsign = f"{letters}{rng.integers(1, 1000):03d}"
            if callsign not in used:
                return callsign

    def _spawn_hdg(self, pos: SpawnPosition, rng: np.random.Generator) -> float:
        """Resolve a candidate's spawn heading (random when unspecified)."""
        if pos.hdg_deg is not None:
            return float(pos.hdg_deg) % 360.0
        return float(rng.uniform(0.0, 360.0))

    def _spawn_position_clear(
        self,
        pos: SpawnPosition,
        hdg_deg: float,
        conflict_free: bool,
        *,
        sep_nm: float | None = None,
        sep_ft: float | None = None,
        lookahead_s: float | None = None,
    ) -> bool:
        """Whether a candidate spawn state is acceptable against live traffic.

        Both branches clear the same separation - the region's ``spawn_sep_nm`` /
        ``spawn_sep_ft``, each defaulting to CD's own zone - and differ only in
        *when* they look:

        * ``conflict_free``: the predicted closest approach over
          ``lookahead_s``, so the aircraft is not on course to conflict either.
        * otherwise: the spawn's present position only, which is what a
          steady-state ``maintain`` top-up needs - materialising on top of live
          traffic is an instant loss of separation the policy could not avoid,
          while a conflict that *develops* later is the task.

        A present-position breach needs both dimensions, so a candidate laterally
        close to traffic but well above it is clear. ``lookahead_s`` is unused
        here: nothing is predicted.

        ``conflict_free`` is resolved per region by
        ``SpawnConfig.region_conflict_free`` and the margins by
        ``SpawnConfig.region_spawn_separation``.
        """
        if conflict_free:
            return not predicted_conflict(
                pos.lat_deg,
                pos.lon_deg,
                pos.alt_ft,
                hdg_deg,
                pos.spd_kts,
                sep_nm=sep_nm,
                sep_ft=sep_ft,
                lookahead_s=lookahead_s,
                # Steady wind only: existing traffic's ``gs`` carries the field,
                # so both sides of the relative velocity must be ground-
                # referenced. Gusts are zero-mean and per-step.
                wind_kts=float(getattr(self._config(), "wind_kts", 0.0)),
                wind_dir_deg=float(getattr(self._config(), "wind_dir_deg", 0.0)),
            )
        return not inside_separation_zone(
            pos.lat_deg,
            pos.lon_deg,
            pos.alt_ft,
            sep_nm=sep_nm,
            sep_ft=sep_ft,
        )

    def _resolve_spawn_speed(
        self,
        actype: str,
        pos: SpawnPosition,
        rng: np.random.Generator,
    ) -> SpawnPosition:
        """Resolve an envelope-sampled spawn speed *before* any clearance check.

        The feasible CAS band needs the live performance model (``vmin`` is only
        known post-creation), so create a short-lived probe aircraft, draw the
        CAS at the spawn altitude, and delete the probe. The returned position
        carries the final speed (``spd_from_envelope`` cleared), so the
        conflict-free check and the materialised aircraft use the *same* speed -
        redrawing it after the check would silently invalidate the clearance.
        """
        if not pos.spd_from_envelope:
            return pos
        used = set(self.env._runtime.agent_ids)
        callsign = self._generate_callsign(used, rng)
        self.env._runtime.create_aircraft(
            callsign,
            actype,
            pos.lat_deg,
            pos.lon_deg,
            0.0,
            pos.alt_ft,
            pos.spd_kts,
        )
        try:
            cas_kts = feasible_cas_at_alt(
                self.env._runtime.index(callsign), pos.alt_ft, rng
            )
        finally:
            self.env._runtime.delete_aircraft(callsign)
        return replace(pos, spd_kts=cas_kts, spd_from_envelope=False)

    def _conflict_free_item(
        self, item: SpawnQueueItem, rng: np.random.Generator
    ) -> SpawnQueueItem | None:
        """Resample a queued spawn until its state is clear.

        Resolves and pins the spawn speed (envelope draw) and heading before
        each check, so the materialised aircraft flies exactly the state that
        was cleared. Returns ``None`` when no clear candidate is found within
        ``_MAINTAIN_SPAWN_MAX_TRIES`` - the caller defers the spawn instead of
        creating an aircraft in (predicted) conflict.
        """
        sep_nm, sep_ft, look_s = self.env.episode_spawn.region_spawn_separation(
            item.region_index
        )
        for _ in range(_MAINTAIN_SPAWN_MAX_TRIES):
            pos = self._resolve_spawn_speed(item.actype, item.position, rng)
            hdg = self._spawn_hdg(pos, rng)
            if self._spawn_position_clear(
                pos,
                hdg,
                conflict_free=True,
                sep_nm=sep_nm,
                sep_ft=sep_ft,
                lookahead_s=look_s,
            ):
                return replace(item, position=replace(pos, hdg_deg=hdg))
            _t, actype, position, prefix, route = (
                self.env.episode_spawn.sample_region_spawn(item.region_index, rng)
            )
            item = replace(
                item,
                actype=actype,
                position=SpawnPosition.from_mapping(position),
                callsign_prefix=prefix,
                route=route,
            )
        return None

    def _sample_clear_spawn(
        self,
        region_index: int,
        rng: np.random.Generator,
    ) -> SpawnQueueItem | None:
        """Sample a maintain-region spawn clear of existing traffic.

        Retries up to ``_MAINTAIN_SPAWN_MAX_TRIES`` (respecting
        ``conflict_free_spawn``), returning ``None`` if none is clear so the
        top-up defers to a later step. Repeated failure warns once per region:
        an unsatisfiable guard would otherwise quietly hold the region below the
        aircraft count ``maintain`` asks for.
        """
        conflict_free = self.env.episode_spawn.region_conflict_free(region_index)
        sep_nm, sep_ft, look_s = self.env.episode_spawn.region_spawn_separation(
            region_index
        )
        for _ in range(_MAINTAIN_SPAWN_MAX_TRIES):
            _spawn_time, actype, position, prefix, route = (
                self.env.episode_spawn.sample_region_spawn(region_index, rng)
            )
            pos = SpawnPosition.from_mapping(position)
            if conflict_free:
                # Pin the envelope speed before the check so the cleared state
                # is the flown state (see ``_resolve_spawn_speed``).
                pos = self._resolve_spawn_speed(actype, pos, rng)
            hdg = self._spawn_hdg(pos, rng)
            if self._spawn_position_clear(
                pos,
                hdg,
                conflict_free,
                sep_nm=sep_nm,
                sep_ft=sep_ft,
                lookahead_s=look_s,
            ):
                self._maintain_failures.pop(region_index, None)
                return SpawnQueueItem(
                    spawn_time=self.env._runtime.sim_time,
                    actype=actype,
                    position=replace(pos, hdg_deg=hdg),
                    callsign_prefix=prefix,
                    route=route,
                    region_index=region_index,
                )
        n = self._maintain_failures.get(region_index, 0) + 1
        self._maintain_failures[region_index] = n
        if n == _MAINTAIN_SPAWN_WARN_AFTER:
            zone_nm = cd_rpz_m() / nm if sep_nm is None else sep_nm
            zone_ft = cd_hpz_m() / ft if sep_ft is None else sep_ft
            what = (
                "conflict-free spawn state"
                if conflict_free
                else f"spawn position {zone_nm:.1f} nm / {zone_ft:.0f} ft "
                "clear of live traffic"
            )
            warnings.warn(
                f"[spawn] maintain region {region_index} has failed to find a "
                f"{what} on {n} consecutive resets "
                f"({_MAINTAIN_SPAWN_MAX_TRIES} tries each); it is running below "
                "its requested aircraft count. Widen the region, lower the "
                "count, or lower SpawnConfig.spawn_sep_nm / _ft.",
                RuntimeWarning,
                stacklevel=2,
            )
        return None

    def _resolve_route_target_for_aircraft(
        self,
        callsign: str,
        step,
        rng: np.random.Generator,
        from_state: tuple[float, float, float | None] | None = None,
    ) -> WaypointTarget:
        """Resolve one route step into the concrete target sent to BlueSky.

        ``from_state`` is the ``(lat, lon, alt_ft)`` the aircraft starts this leg
        from - ``None`` (the spawn state) for the first leg, the previous
        waypoint for a chained leg - so ``reachable_from_spawn`` bounds the
        envelope draw against the leg it actually flies, not the spawn.
        """
        if isinstance(step, str):
            queryable = self.env.episode_queryables[step]
            if not isinstance(queryable, Waypoint):
                raise TypeError(f"route step {step!r} does not reference a Waypoint")
            return queryable.target
        if not isinstance(step, dict) or "waypoint" not in step:
            raise TypeError(f"route step must be a waypoint name or dict, got {step!r}")

        queryable = self.env.episode_queryables[step["waypoint"]]
        if not isinstance(queryable, Waypoint):
            raise TypeError(
                f"route step {step!r} does not reference a Waypoint queryable"
            )

        target = queryable.target
        sample_bounds = step.get("sample")
        sample_alt_from_envelope = bool(step.get("sample_alt_from_envelope", False))
        sample_speed_from_envelope = bool(step.get("sample_speed_from_envelope", False))
        reachable_from_spawn = bool(step.get("reachable_from_spawn", False))
        lat = target.lat
        lon = target.lon
        alt = step.get("alt_ft", target.alt_ft)
        speed = step.get("speed_kts", target.speed_kts)
        region_alt_lo = region_alt_hi = None

        if sample_bounds is not None:
            lat, lon = sample_bounds.sample_point(rng)
            band = getattr(sample_bounds, "alt_band_at", None)
            if band is not None:
                lo, hi = band(lat, lon)
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    region_alt_lo, region_alt_hi = lo, hi
                    if not sample_alt_from_envelope and alt is not None:
                        alt = float(rng.uniform(lo, hi))

        if sample_alt_from_envelope or sample_speed_from_envelope:
            acidx = self.env._runtime.index(callsign)
            alt_min_ft, alt_max_ft = region_alt_lo, region_alt_hi
            if reachable_from_spawn:
                vs_fraction = float(step.get("reachable_vs_fraction", 1.0))
                from_lat, from_lon, from_alt_ft = from_state or (None, None, None)
                reach_lo, reach_hi = reachable_alt_window(
                    acidx,
                    lat,
                    lon,
                    vs_fraction,
                    from_lat=from_lat,
                    from_lon=from_lon,
                    from_alt_ft=from_alt_ft,
                )
                if reach_lo is not None:
                    alt_min_ft = (
                        reach_lo if alt_min_ft is None else max(alt_min_ft, reach_lo)
                    )
                    alt_max_ft = (
                        reach_hi if alt_max_ft is None else min(alt_max_ft, reach_hi)
                    )
            env_alt, env_cas = feasible_alt_cas(
                acidx,
                rng,
                float(step.get("envelope_alt_floor_ft", 1000.0)),
                alt_min_ft=alt_min_ft,
                alt_max_ft=alt_max_ft,
            )
            if sample_alt_from_envelope:
                alt = env_alt
            if sample_speed_from_envelope:
                speed = env_cas

        resolved_target = WaypointTarget(
            lat=float(lat),
            lon=float(lon),
            waypoint=None if sample_bounds is not None else target.waypoint,
            alt_ft=alt,
            speed_kts=speed,
            reach_radius_nm=target.reach_radius_nm,
            alt_tolerance_ft=target.alt_tolerance_ft,
            speed_tolerance_kts=target.speed_tolerance_kts,
        )
        if resolved_target.speed_kts is not None and resolved_target.alt_ft is None:
            raise ValueError(
                "Route waypoint speed constraints require alt_ft or "
                f"sample_alt_from_envelope. Got {step!r}."
            )
        return resolved_target

    def _resolve_route_for_aircraft(
        self,
        callsign: str,
        route: Sequence[Any] | None,
        rng: np.random.Generator,
    ) -> ResolvedRoute | None:
        if not route:
            return None
        # Resolve legs in order, threading each resolved waypoint as the start
        # state of the next - so a chained leg's reachable-altitude window is
        # measured from the previous waypoint (where the aircraft actually
        # begins that leg), not from the spawn.
        targets: list[WaypointTarget] = []
        from_state: tuple[float, float, float | None] | None = None
        for step in route:
            target = self._resolve_route_target_for_aircraft(
                callsign, step, rng, from_state=from_state
            )
            targets.append(target)
            # Next leg starts where this one ends; carry altitude forward when a
            # leg has no altitude constraint (the aircraft holds), so the chain
            # stays continuous rather than snapping back to the spawn altitude.
            carry_alt = (
                target.alt_ft
                if target.alt_ft is not None
                else (from_state[2] if from_state is not None else None)
            )
            from_state = (float(target.lat), float(target.lon), carry_alt)
        return ResolvedRoute(
            names=[route_step_name(step) for step in route],
            targets=targets,
        )

    def _materialize_spawn(
        self,
        item: SpawnQueueItem,
        used: set[str],
        rng: np.random.Generator,
    ) -> bool:
        """Create one aircraft from a spawn item; return whether it is controlled.

        Shared by the one-shot queue drain and the steady-state ``maintain``
        top-up. ``used`` is the set of callsigns already taken this pass and is
        updated in place.
        """
        callsign = self._generate_callsign(used, rng, item.callsign_prefix)
        used.add(callsign)
        hdg = (
            float(item.position.hdg_deg) % 360.0
            if item.position.hdg_deg is not None
            else float(rng.uniform(0.0, 360.0))
        )
        # bs.traf.cre() writes acalt/acspd straight into self.alt (m) and
        # feeds acspd into vcasormach() (m/s).
        # Spawn params use ft / kts, so convert here.
        self.env._runtime.create_aircraft(
            callsign,
            item.actype,
            item.position.lat_deg,
            item.position.lon_deg,
            hdg,
            item.position.alt_ft,
            item.position.spd_kts,
        )
        if item.position.spd_from_envelope:
            # The provisional create initialised the performance model, so
            # vmin is now known; draw a feasible CAS at the spawn altitude
            # and recreate so the initial speed is set consistently by cre.
            acidx = self.env._runtime.index(callsign)
            cas_kts = feasible_cas_at_alt(acidx, item.position.alt_ft, rng)
            self.env._runtime.delete_aircraft(callsign)
            self.env._runtime.create_aircraft(
                callsign,
                item.actype,
                item.position.lat_deg,
                item.position.lon_deg,
                hdg,
                item.position.alt_ft,
                cas_kts,
            )

        route = self._resolve_route_for_aircraft(callsign, item.route, rng)
        if route:
            self.env._runtime.append_aircraft_route(
                callsign,
                route.targets,
            )
            self.env._query_state_monitor.set_aircraft_route(
                callsign,
                route.names,
            )
        # Hooks see a names-only view (tasks may treat the route as a list of
        # waypoint names); the full step list with any per-step crossing
        # restrictions only reaches ADDWPT above.
        route_names = route.names if route else None
        self.env._hooks.on_aircraft_spawned(callsign, route_names)
        state = self.env._hooks.define_initial_aircraft_control_state(
            callsign,
            route_names,
        )
        # A spawn region marked ``controlled=False`` forces its aircraft to
        # background (uncooperative) traffic regardless of the task hook: they
        # fly their route but the policy never commands them.
        regions = self.env.episode_spawn.regions
        if (
            0 <= item.region_index < len(regions)
            and not regions[item.region_index].controlled
        ):
            state = AircraftControlState.BACKGROUND
        self.env.set_aircraft_control_state(callsign, state)
        # Use the actual materialisation time (``bs.sim.simt``) rather than the
        # scheduled ``spawn_time`` so ``time_in_env`` measures real simulator
        # presence - not the gap between schedule and the draining step.
        self.env._aircraft_spawn_time[callsign] = self.env._runtime.sim_time
        self.env._aircraft_region[callsign] = item.region_index
        return state is AircraftControlState.CONTROLLED

    # Naming convention in this section:
    # - "agent" means a controlled PettingZoo participant.
    # - "aircraft" means a live BlueSky traffic object in any control state.

    def _enqueue_spawns(self, rng: np.random.Generator) -> None:
        """Sample one episode's worth of spawns into ``self._queue``.

        Sampled items are sorted ascending by ``spawn_time`` so the drain
        path can stop as soon as it hits an entry that isn't due yet.
        """
        self._maintain_failures.clear()
        self._queue = sorted(
            (
                SpawnQueueItem(
                    spawn_time=spawn_time,
                    actype=actype,
                    position=SpawnPosition.from_mapping(position),
                    callsign_prefix=callsign_prefix,
                    route=route,
                    region_index=region_index,
                )
                for region_index, spawn_time, actype, position, callsign_prefix, route in (
                    self.env.episode_spawn.iter_spawns(
                        rng,
                        limit=self.env.episode_max_aircraft,
                    )
                )
            ),
            key=lambda item: item.spawn_time,
        )

    def _sample_maintain_targets(self, rng: np.random.Generator) -> None:
        """Fix each ``maintain`` region's target live count for this episode."""
        self._maintain_target = {
            region_index: region.sample_n(rng)
            for region_index, region in enumerate(self.env.episode_spawn.regions)
            if region.maintain
        }

    def _drain_spawn_queue(self, rng: np.random.Generator) -> None:
        """Materialise every queued aircraft whose ``spawn_time`` has arrived.

        A conflict-free item with no clear candidate state is *deferred* (its
        ``spawn_time`` pushed one env step) rather than spawned in conflict -
        the conflict-free guarantee is absolute at spawn time; density catches
        up as soon as the airspace admits a clear state.
        """
        if not self._queue:
            return
        now = self.env._runtime.sim_time
        used = set(self.env._runtime.agent_ids)
        # Queue is sorted by spawn_time; pop from the front while due.
        i = 0
        agent_cache_changed = False
        deferred: list[SpawnQueueItem] = []
        for item in self._queue:
            if item.spawn_time > now:
                break
            i += 1
            if self.env.episode_spawn.region_conflict_free(item.region_index):
                cleared = self._conflict_free_item(item, rng)
                if cleared is None:
                    retry_at = now + (float(self._config().dt) or 1.0)
                    print(
                        "[spawn] no conflict-free spawn state found after "
                        f"{_MAINTAIN_SPAWN_MAX_TRIES} tries (region "
                        f"{item.region_index}); deferring to t={retry_at:.0f}s"
                    )
                    deferred.append(replace(item, spawn_time=retry_at))
                    continue
                item = cleared
            agent_cache_changed = (
                self._materialize_spawn(item, used, rng) or agent_cache_changed
            )
        if i:
            del self._queue[:i]
        if deferred:
            self._queue.extend(deferred)
            self._queue.sort(key=lambda queued: queued.spawn_time)
        if agent_cache_changed:
            self.env._invalidate_agent_cache()

    def _maintain_spawns(self, rng: np.random.Generator) -> None:
        """Top up each steady-state region to its target live count.

        Called after spawns/deletions each step (and at reset): counts live
        aircraft per maintain region and materialises separation-guarded
        replacements until the target is met. A region that can't place a
        clear spawn this step simply retries next step.
        """
        if not self._maintain_target:
            return
        live = self.env._live_agent_id_set()
        counts: dict[int, int] = {}
        for acid, region_index in self.env._aircraft_region.items():
            if acid in live:
                counts[region_index] = counts.get(region_index, 0) + 1
        used = set(self.env._runtime.agent_ids)
        changed = False
        for region_index, target in self._maintain_target.items():
            for _ in range(max(0, target - counts.get(region_index, 0))):
                item = self._sample_clear_spawn(region_index, rng)
                if item is None:
                    break  # too crowded to place clear of traffic; retry later
                changed = self._materialize_spawn(item, used, rng) or changed
        if changed:
            self.env._invalidate_agent_cache()
