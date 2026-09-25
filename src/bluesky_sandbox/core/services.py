from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from collections.abc import Sequence as SequenceABC

import bluesky as bs
import numpy as np
from bluesky.tools.aero import ft, kts
from bluesky.tools.geo import qdrdist
from gymnasium.spaces import Box, Dict, Sequence

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.core.aircraft_table import AircraftTable, Column
from bluesky_sandbox.interface.fields.base import (
    PairObsField,
    SwitchActionMixin,
)
from bluesky_sandbox.interface.task import (
    SeparationContext,
    SeparationEvent,
    StepEvent,
    StepTime,
)
from bluesky_sandbox.interface.wrappers.observations.normalizer import Normalizer
from bluesky_sandbox.sim.bounds import contains_many
from bluesky_sandbox.sim.performance.speeds import within_speed_tolerance_many
from bluesky_sandbox.sim.queryables import (
    QueryRegion,
    RegionCurrent,
    RegionResult,
    RegionStep,
    Waypoint,
    WaypointResult,
    WaypointStep,
)
from bluesky_sandbox.sim.spawn import param_alt_range
from bluesky_sandbox.ui.display.overlays import BoundsResource, Renderable


def _field_normalizer(field) -> Normalizer | None:
    return getattr(field, "normalizer", None)


def _field_output_bounds(field, idx: int | None) -> tuple[list[float], list[float]]:
    normalizer = _field_normalizer(field)
    if normalizer is None:
        if idx is None and field.meta.dynamic_bounds:
            # Unbounded per output component (multi-dim dynamic fields must span
            # their full width here, not a single element).
            size = _field_output_size(field)
            return [float("-inf")] * size, [float("inf")] * size
        idx = 0 if idx is None else idx
        lo, hi = field.bounds(idx)
        return _flatten_field_values(lo), _flatten_field_values(hi)
    return normalizer.output_bounds(field)


def _field_output_size(field) -> int:
    normalizer = _field_normalizer(field)
    if normalizer is None:
        output_size = getattr(field, "output_size", None)
        if callable(output_size):
            return int(output_size())
        return 1
    return normalizer.output_size(field)


def _flatten_field_values(value) -> list[float]:
    values = np.asarray(value, dtype=np.float32).reshape(-1)
    return [float(item) for item in values]


def _normalize_field_value(field, value, idx: int) -> list[float]:
    normalizer = _field_normalizer(field)
    if normalizer is None:
        values = _flatten_field_values(value)
        expected = _field_output_size(field)
        if len(values) != expected:
            raise ValueError(
                f"Observation field {field.meta.name!r} expected {expected} "
                f"values, got {len(values)}."
            )
        return values
    values = _flatten_field_values(value)
    if len(values) != 1:
        raise ValueError(
            f"Observation field {field.meta.name!r} uses "
            f"{normalizer.__class__.__name__}, which expects one raw value, "
            f"got {len(values)}."
        )
    value = values[0]
    return normalizer.normalize(field, value, idx)


def _normalize_field_values_batch(field, values, idx: int) -> np.ndarray:
    """Vectorized :func:`_normalize_field_value` over a batch of raw values that
    all share ``idx`` (one ownship's intruders). Returns an
    ``(n, output_size)`` ``float32`` array, byte-identical to normalizing each
    element. ``values`` is 1-D for normalizer fields (Circular expands to 2);
    ``(n, size)`` for no-normalizer fields.
    """
    normalizer = _field_normalizer(field)
    if normalizer is None:
        arr = np.asarray(values, dtype=np.float32)
        expected = _field_output_size(field)
        # An empty batch - an ownship with no intruders - carries no elements to
        # infer the width from, so state it rather than let ``-1`` fail.
        arr = arr.reshape(arr.shape[0], expected if arr.size == 0 else -1)
        if arr.shape[1] != expected:
            raise ValueError(
                f"Observation field {field.meta.name!r} expected {expected} "
                f"values, got {arr.shape[1]}."
            )
        return arr
    return normalizer.normalize_many(field, values, idx)


def _denormalize_action_value(field, values, idx: int):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    normalizer = _field_normalizer(field)
    if normalizer is None:
        expected = _field_output_size(field)
        if values.size != expected:
            raise ValueError(
                f"Action field {field.meta.name!r} expected {expected} "
                f"value(s), got {values.size}."
            )
        if expected == 1:
            return float(values[0])
        return values.copy()
    expected = normalizer.output_size(field)
    if values.size != expected:
        raise ValueError(
            f"Action field {field.meta.name!r} uses "
            f"{normalizer.__class__.__name__}, which expects {expected} "
            f"values, got {values.size}."
        )
    return normalizer.denormalize(field, values.tolist(), idx)


class ActionDispatcher:
    """Apply configured action fields in dependency-aware order."""

    def __init__(self, env=None) -> None:
        self.env = env

    def bind_env(self, env) -> None:
        self.env = env

    def _config(self) -> EnvConfig:
        if self.env is None:
            raise RuntimeError("ActionDispatcher env has not been set.")
        return self.env.config

    def apply(self, idx: int, action) -> None:
        action_fields = self._config().action_fields
        flat_action = np.asarray(action, dtype=np.float32).reshape(-1)
        values = []
        cursor = 0
        for field in action_fields:
            size = _field_output_size(field)
            next_cursor = cursor + size
            if next_cursor > flat_action.size:
                raise ValueError(
                    f"Action for {field.meta.name!r} needs {size} values at "
                    f"offset {cursor}, but action has {flat_action.size} values."
                )
            values.append(
                (
                    field,
                    _denormalize_action_value(
                        field,
                        flat_action[cursor:next_cursor],
                        idx,
                    ),
                )
            )
            cursor = next_cursor
        if cursor != flat_action.size:
            raise ValueError(
                f"Action has {flat_action.size} values but configured fields "
                f"consume {cursor}."
            )
        switch_fields = [
            (field, value)
            for field, value in values
            if isinstance(field, SwitchActionMixin)
        ]
        switch_commands = {}
        switch_on = {}
        for field, value in switch_fields:
            command = field.switch_command(value)
            switch_commands[field.meta.name] = command
            switch_on[field.meta.name] = command is True or (
                command is None and field.current_switch_state(idx)
            )
        switch_on_before_dependencies = dict(switch_on)

        changed = True
        while changed:
            changed = False
            for field, _value in switch_fields:
                if not switch_on.get(field.meta.name, False):
                    continue
                for required in field.meta.requires_on:
                    if not switch_on.get(required, False):
                        switch_on[required] = True
                        changed = True

        suppressed_axes = {
            axis
            for field, _value in switch_fields
            if switch_on.get(field.meta.name, False)
            for axis in field.meta.suppresses_when_on
        }

        for field, value in switch_fields:
            if switch_commands.get(field.meta.name) is False:
                field.set(idx, value)

        for field, value in values:
            if isinstance(field, SwitchActionMixin):
                continue
            if field.meta.control_axis in suppressed_axes:
                continue
            field.set(idx, value)

        for field, value in switch_fields:
            command = switch_commands.get(field.meta.name)
            if command is True:
                field.set(idx, value)
            elif switch_on.get(
                field.meta.name, False
            ) and not switch_on_before_dependencies.get(field.meta.name, False):
                field.set(idx, field.switch_on_value())


class ObservationAssembler:
    """Build observations and spaces from configured field objects."""

    def __init__(self, env=None) -> None:
        self.env = env

    def bind_env(self, env) -> None:
        self.env = env

    def _config(self) -> EnvConfig:
        if self.env is None:
            raise RuntimeError("ObservationAssembler env has not been set.")
        return self.env.config

    def observation_space(self, agent):
        config = self._config()
        idx = None if agent is None else bs.traf.id.index(agent)
        own_low, own_high = self.field_output_bounds(idx, config.obs_fields)
        ownship_space = Box(low=own_low, high=own_high, dtype=np.float32)

        intr_fields = config.intruder_obs_fields
        critic_own_fields = config.critic_obs_fields
        critic_intr_fields = config.critic_intruder_obs_fields
        if not (intr_fields or critic_own_fields or critic_intr_fields):
            return ownship_space

        def _sequence(fields):
            low, high = self.field_output_bounds(idx, fields)
            return Sequence(Box(low=low, high=high, dtype=np.float32), stack=True)

        spaces = {"ownship": ownship_space}
        if intr_fields:
            spaces["intruders"] = _sequence(intr_fields)
        # Privileged critic-only blocks (see EnvConfig.critic_*obs_fields). They
        # ride in the observation Dict so the pad wrapper / batched_obs carry them
        # alongside the actor blocks; the actor's encoder ignores them.
        if critic_own_fields:
            c_low, c_high = self.field_output_bounds(idx, critic_own_fields)
            spaces["critic_ownship"] = Box(low=c_low, high=c_high, dtype=np.float32)
        if critic_intr_fields:
            spaces["critic_intruders"] = _sequence(critic_intr_fields)
        return Dict(spaces)

    def action_space(self, agent):
        config = self._config()
        idx = None if agent is None else bs.traf.id.index(agent)
        low, high = self.field_output_bounds(idx, config.action_fields)
        return Box(low=low, high=high, dtype=np.float32)

    def get_obs(self, agent_ids=None) -> dict:
        config = self._config()
        ntraf = bs.traf.ntraf
        all_indices = tuple(range(ntraf))

        def _ownship_pack(fields):
            specs = tuple((f, _field_output_size(f)) for f in fields)
            dim = sum(size for _f, size in specs)
            raw = [f.get_many(all_indices) for f, _size in specs]
            return specs, dim, raw

        def _intruder_pack(fields):
            specs = tuple(
                (f, _field_output_size(f), isinstance(f, PairObsField)) for f in fields
            )
            dim = sum(size for _f, size, _p in specs)
            # Coerce to an ndarray: intruder batches are fancy-indexed by
            # ``other_arr`` below, which fails on the plain list returned by the
            # default ``ObsField.get_many`` (fields only ever used as ownship
            # observations never hit that path).
            raw = [
                None if is_pair else np.asarray(f.get_many(all_indices))
                for f, _size, is_pair in specs
            ]
            return specs, dim, raw

        obs_specs, ownship_dim, ownship_raw = _ownship_pack(config.obs_fields)
        intr_fields = config.intruder_obs_fields or []
        critic_own_fields = config.critic_obs_fields or []
        critic_intr_fields = config.critic_intruder_obs_fields or []
        intr_specs, intruder_dim, intr_raw = _intruder_pack(intr_fields)
        critic_own_specs, critic_own_dim, critic_own_raw = _ownship_pack(
            critic_own_fields
        )
        critic_intr_specs, critic_intr_dim, critic_intr_raw = _intruder_pack(
            critic_intr_fields
        )

        def _fill_ownship(specs, raw, dim, acidx):
            vec = np.empty(dim, dtype=np.float32)
            cursor = 0
            for spec_idx, (field, _size) in enumerate(specs):
                values = _normalize_field_value(field, raw[spec_idx][acidx], acidx)
                next_cursor = cursor + len(values)
                vec[cursor:next_cursor] = values
                cursor = next_cursor
            return vec

        def _fill_intruders(specs, raw, dim, other_indices, other_arr, acidx):
            block = np.empty((len(other_indices), dim), dtype=np.float32)
            cursor = 0
            for spec_idx, (field, size, is_pair) in enumerate(specs):
                if is_pair:
                    raw_batch = field.get_pairs(acidx, other_indices)
                else:
                    raw_batch = raw[spec_idx][other_arr]
                # Bounds are resolved once at the shared ownship idx (acidx), so
                # the whole intruder batch normalizes in one vectorized op.
                block[:, cursor : cursor + size] = _normalize_field_values_batch(
                    field, raw_batch, acidx
                )
                cursor += size
            return block

        needs_others = bool(intr_fields) or bool(critic_intr_fields)

        obs = {}
        if agent_ids is None:
            agent_items = tuple(enumerate(bs.traf.id))
        else:
            live_index = {acid: idx for idx, acid in enumerate(bs.traf.id)}
            agent_items = tuple(
                (idx, acid)
                for acid in agent_ids
                if (idx := live_index.get(acid)) is not None
            )
        for acidx, acid in agent_items:
            ownship = _fill_ownship(obs_specs, ownship_raw, ownship_dim, acidx)

            if not (intr_fields or critic_own_fields or critic_intr_fields):
                obs[acid] = ownship
                continue

            agent_obs = {"ownship": ownship}
            if needs_others:
                other_indices = tuple(idx for idx in range(ntraf) if idx != acidx)
                # dtype=intp so an *empty* other set (a lone surviving agent)
                # stays an integer index array; np.asarray(()) defaults to
                # float64, which raises on the non-pair fancy-index path below.
                other_arr = np.asarray(other_indices, dtype=np.intp)
            if intr_fields:
                agent_obs["intruders"] = _fill_intruders(
                    intr_specs, intr_raw, intruder_dim, other_indices, other_arr, acidx
                )
            # Privileged critic-only blocks: same geometry, appended to the
            # critic's view only (see EnvConfig.critic_*obs_fields).
            if critic_own_fields:
                agent_obs["critic_ownship"] = _fill_ownship(
                    critic_own_specs, critic_own_raw, critic_own_dim, acidx
                )
            if critic_intr_fields:
                agent_obs["critic_intruders"] = _fill_intruders(
                    critic_intr_specs,
                    critic_intr_raw,
                    critic_intr_dim,
                    other_indices,
                    other_arr,
                    acidx,
                )
            obs[acid] = agent_obs
        return obs

    def field_bounds(self, idx: int, fields) -> tuple[np.ndarray, np.ndarray]:
        resolved = [field.bounds(idx) for field in fields]
        low = np.array([lo for lo, _hi in resolved], dtype=np.float32)
        high = np.array([hi for _lo, hi in resolved], dtype=np.float32)
        return low, high

    def ownship_bounds(self, idx: int, fields) -> tuple[np.ndarray, np.ndarray]:
        return self.field_bounds(idx, fields)

    def field_output_bounds(
        self,
        idx: int | None,
        fields,
    ) -> tuple[np.ndarray, np.ndarray]:
        lows: list[float] = []
        highs: list[float] = []
        for field in fields:
            lo, hi = _field_output_bounds(field, idx)
            lows.extend(lo)
            highs.extend(hi)
        return np.array(lows, dtype=np.float32), np.array(highs, dtype=np.float32)


class RenderableBuilder:
    """Map environment config resources to driver renderables."""

    def __init__(self, env=None) -> None:
        self.env = env

    def bind_env(self, env) -> None:
        self.env = env

    def _config(self) -> EnvConfig:
        if self.env is None:
            raise RuntimeError("RenderableBuilder env has not been set.")
        return self.env.config

    def iter_renderables(self) -> Iterator[Renderable]:
        if self.env is None:
            raise RuntimeError("RenderableBuilder env has not been set.")
        airspace_bounds = self.env.episode_airspace_bounds
        spawn = self.env.episode_spawn
        queryables = self.env.episode_queryables
        if airspace_bounds is not None:
            yield BoundsResource(
                bounds=airspace_bounds,
                color="red",
                label="AIRSPACE",
                kind="airspace",
            )

        for i, region in enumerate(spawn.regions):
            if not region.render_shape:
                continue
            default_name = f"SPAWN {i}" if region.name is None else region.name
            yield BoundsResource(
                bounds=region.bounds,
                color="green",
                label=default_name if region.render_name else "",
                kind="spawn",
                alt_range_override=param_alt_range(region.params.get("alt_ft")),
                extra_meta={"spawn_alt": region.params.get("alt_ft")},
            )

        for name, qable in queryables.items():
            if isinstance(qable, QueryRegion):
                if not qable.render_shape:
                    continue
                yield BoundsResource(
                    bounds=qable.bounds,
                    color=qable.color,
                    label=name if qable.render_label else "",
                    kind="query",
                )
            # Waypoints are intentionally NOT drawn as static markers: they
            # would clutter the view and are shown per-aircraft on the selected
            # aircraft's route instead (see each view's selected-route drawing).


def _aircraft_keys(env) -> tuple[int, ...]:
    """The monitors' row keys: each aircraft's uid, in ``bs.traf.id`` order.

    Keyed by uid, not callsign: a new aircraft can reuse a deleted one's
    callsign, and rows keyed by callsign would hand it the old one's history.
    """
    if env is None:
        raise RuntimeError("Monitor env has not been set.")
    return tuple(env._runtime.aircraft_uids.tolist())


def _row_at(table: AircraftTable, env, acidx: int) -> int | None:
    """The table row of the aircraft at BlueSky index ``acidx``, by its uid.

    Right however traffic changed since the table last synced: the uid array
    is kept current by BlueSky itself.
    """
    uids = env._runtime.aircraft_uids
    if not 0 <= acidx < len(uids):
        return None
    return table.row.get(int(uids[acidx]))


class QueryStateMonitor:
    """Track built-in queryable event state across simulator substeps.

    One :class:`AircraftTable` holds it: a row per aircraft, a column per
    episode queryable. ``route_index`` is where each queryable sits on the
    aircraft's BlueSky route. The rest is written only for queryables that
    ``track_temporal_state``: ``held_*`` count the substeps a queryable's
    predicate held - inside a region, a waypoint's constraints satisfied - and
    the waypoint-only ``reached_*`` / ``min_*`` stay at their fills for regions.
    """

    def __init__(self, env=None) -> None:
        self.env = env
        # (name, queryable) for each queryable that tracks temporal state.
        self._tracked: tuple[tuple[str, object], ...] = ()
        self._table = AircraftTable(
            {
                "route_index": Column(-1, np.int32),
                "held_total_s": Column(0.0, np.float64),
                "held_step_substeps": Column(0, np.int32, per_step=True),
                "reached_step_substeps": Column(0, np.int32, per_step=True),
                "min_distance_nm": Column(math.inf, np.float64, per_step=True),
                "min_abs_alt_diff_ft": Column(math.inf, np.float64, per_step=True),
            },
            keyed_columns=True,
        )
        # Per-step route-target cache for substep dwell tracking:
        # name -> ((aircraft ids, route-index bytes), target arrays).
        self._waypoint_target_cache: dict[str, tuple[tuple, tuple]] = {}

    def bind_env(self, env) -> None:
        self.env = env

    def clear(self) -> None:
        self._table.clear()
        self.begin_step()

    def set_aircraft_route(
        self,
        acid: str,
        route_names: SequenceABC[str] | None,
    ) -> None:
        """Remember which BlueSky route index corresponds to each query name."""
        self._sync_layout()
        row = self._row_of(acid)
        if row is None:
            return
        route_index = self._table["route_index"]
        route_index[row, :] = -1
        if not route_names:
            return
        for index, name in enumerate(route_names):
            col = self._table.col.get(name)
            if col is not None and route_index[row, col] < 0:
                route_index[row, col] = index

    def clear_aircraft_route(self, acid: str) -> None:
        row = self._row_of(acid)
        if row is not None:
            self._table["route_index"][row, :] = -1

    def _row_of(self, acid: str) -> int | None:
        try:
            acidx = bs.traf.id.index(acid)
        except ValueError:
            return None
        return _row_at(self._table, self.env, acidx)

    def _route_index(self, acidx: int, name: str) -> int | None:
        row = _row_at(self._table, self.env, acidx)
        col = self._table.col.get(name)
        if row is None or col is None:
            return None
        route_idx = int(self._table["route_index"][row, col])
        return route_idx if route_idx >= 0 else None

    def begin_step(self) -> None:
        self._sync_layout()
        queryables = {} if self.env is None else self.env.episode_queryables
        self._tracked = tuple(
            (name, queryable)
            for name, queryable in queryables.items()
            if bool(getattr(queryable, "track_temporal_state", False))
        )
        self._table.reset_step()
        # Route targets may be edited by this step's actions (dispatched after
        # begin_step, before the substep loop) - rebuild lazily per step.
        self._waypoint_target_cache.clear()

    def _sync_layout(self) -> None:
        """Columns to the episode's queryables, rows to BlueSky's aircraft."""
        names = () if self.env is None else tuple(self.env.episode_queryables)
        self._table.set_columns(names)
        self._table.sync_rows(_aircraft_keys(self.env))

    def record_substep(self) -> None:
        if self.env is None:
            raise RuntimeError("QueryStateMonitor env has not been set.")
        if not self._tracked:
            return
        table = self._table
        table.sync_rows(_aircraft_keys(self.env))
        simdt = float(self.env.config.simdt)
        n = len(table)
        if n == 0:
            return

        lat_deg = np.asarray(bs.traf.lat, dtype=np.float64)[:n]
        lon_deg = np.asarray(bs.traf.lon, dtype=np.float64)[:n]
        alt_ft = np.asarray(bs.traf.alt, dtype=np.float64)[:n] / ft
        held_step = table["held_step_substeps"]
        held_total = table["held_total_s"]

        waypoints = []
        for name, queryable in self._tracked:
            col = table.col[name]
            if isinstance(queryable, Waypoint):
                waypoints.append((name, queryable, col))
                continue
            if not isinstance(queryable, QueryRegion):
                continue
            inside = contains_many(queryable.bounds, lat_deg, lon_deg, alt_ft)
            if inside is None:
                inside = np.fromiter(
                    (queryable.contains_aircraft(acidx) for acidx in range(n)),
                    dtype=bool,
                    count=n,
                )
            held_step[:, col] += inside.astype(np.int32)
            held_total[:, col] += inside.astype(np.float64) * simdt

        if not waypoints:
            return

        reached_rows = np.zeros(n, dtype=bool)
        try:
            reached_indices = np.asarray(tuple(bs.traf.ap.idxreached), dtype=np.int64)
            reached_indices = reached_indices[
                (0 <= reached_indices) & (reached_indices < n)
            ]
            reached_rows[reached_indices] = True
        except (AttributeError, TypeError, ValueError):
            pass
        # Which route waypoint each aircraft just passed. It only feeds
        # ``reached_rows & ...``, and a waypoint is reached on a tiny fraction
        # of substeps, so skip the per-aircraft route walk when none was.
        just_reached_idx = np.full(n, -1, dtype=np.int32)
        if reached_rows.any():
            active_route_idx = np.full(n, -1, dtype=np.int32)
            routes = bs.traf.ap.route
            for acidx in range(min(n, len(routes))):
                try:
                    active_route_idx[acidx] = (
                        -1
                        if routes[acidx].iactwp is None
                        else int(routes[acidx].iactwp)
                    )
                except (TypeError, ValueError):
                    active_route_idx[acidx] = -1
            swlnav = np.asarray(bs.traf.swlnav, dtype=np.bool_)[:n]
            just_reached_idx = np.where(swlnav, active_route_idx - 1, active_route_idx)

        min_distance = table["min_distance_nm"]
        min_abs_alt = table["min_abs_alt_diff_ft"]
        reached_step = table["reached_step_substeps"]
        for name, queryable, col in waypoints:
            route_indices = table["route_index"][:, col]
            (
                distance_nm,
                alt_diff_ft,
                satisfied,
            ) = self._waypoint_tracking_arrays(
                name,
                queryable,
                route_indices,
                lat_deg,
                lon_deg,
                alt_ft,
            )
            np.minimum(min_distance[:, col], distance_nm, out=min_distance[:, col])
            abs_alt_diff = np.abs(alt_diff_ft)
            finite_alt = np.isfinite(abs_alt_diff)
            np.minimum(
                min_abs_alt[:, col],
                np.where(finite_alt, abs_alt_diff, math.inf),
                out=min_abs_alt[:, col],
            )
            held_step[:, col] += satisfied.astype(np.int32)
            held_total[:, col] += satisfied.astype(np.float64) * simdt
            reached = reached_rows & (route_indices == just_reached_idx)
            reached_step[:, col] += reached.astype(np.int32)

    def _waypoint_target_arrays(
        self,
        name: str,
        queryable: Waypoint,
        route_indices: np.ndarray,
        n: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per-aircraft ``(lat, lon, alt_ft, speed_kts)`` target arrays, cached.

        Route waypoint data only changes between env steps (actions dispatch
        before the substep loop; waypoint *passage* moves ``iactwp`` but not the
        stored waypoint), so the per-aircraft route-target resolution is cached
        for the step and invalidated in :meth:`begin_step`. The key also carries
        the aircraft-id tuple and the route-index column, so mid-step spawns or
        route re-indexing rebuild it.
        """
        key = (self._table.ids, route_indices.tobytes())
        cached = self._waypoint_target_cache.get(name)
        if cached is not None and cached[0] == key:
            return cached[1]

        target = queryable.target
        target_lat = np.full(n, target.lat, dtype=np.float64)
        target_lon = np.full(n, target.lon, dtype=np.float64)
        target_alt_ft = np.full(
            n,
            math.nan if target.alt_ft is None else target.alt_ft,
            dtype=np.float64,
        )
        target_speed_kts = np.full(
            n,
            math.nan if target.speed_kts is None else target.speed_kts,
            dtype=np.float64,
        )
        route_rows = np.flatnonzero(route_indices >= 0)
        for acidx in route_rows:
            try:
                route_target = queryable.target_from_route(
                    int(acidx),
                    int(route_indices[acidx]),
                )
            except (IndexError, TypeError, ValueError):
                continue
            target_lat[acidx] = route_target.lat
            target_lon[acidx] = route_target.lon
            target_alt_ft[acidx] = (
                math.nan if route_target.alt_ft is None else route_target.alt_ft
            )
            target_speed_kts[acidx] = (
                math.nan if route_target.speed_kts is None else route_target.speed_kts
            )
        arrays = (target_lat, target_lon, target_alt_ft, target_speed_kts)
        self._waypoint_target_cache[name] = (key, arrays)
        return arrays

    def _waypoint_tracking_arrays(
        self,
        name: str,
        queryable: Waypoint,
        route_indices: np.ndarray,
        lat_deg: np.ndarray,
        lon_deg: np.ndarray,
        alt_ft: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = lat_deg.size
        target = queryable.target
        (
            target_lat,
            target_lon,
            target_alt_ft,
            target_speed_kts,
        ) = self._waypoint_target_arrays(name, queryable, route_indices, n)

        _qdr_deg, distance_nm = qdrdist(lat_deg, lon_deg, target_lat, target_lon)
        alt_diff_ft = alt_ft - target_alt_ft
        within_lateral = (
            np.ones(n, dtype=bool)
            if target.reach_radius_nm is None
            else distance_nm <= target.reach_radius_nm
        )
        within_altitude = (
            np.ones(n, dtype=bool)
            if target.alt_tolerance_ft is None
            else np.isnan(target_alt_ft)
            | (np.abs(alt_diff_ft) <= target.alt_tolerance_ft)
        )
        # Speed constraint per aircraft, regime-aware (Mach above the CAS/Mach
        # crossover, CAS below) via the shared helper, so the dwell-tracking mask
        # agrees with current_state and stays well-defined for any sampled target.
        within_speed = np.ones(n, dtype=bool)
        if (
            target.speed_tolerance_kts is not None
            or target.speed_tolerance_mach is not None
        ):
            within_speed = within_speed_tolerance_many(
                n,
                target_speed_kts * kts,
                target.speed_tolerance_kts,
                target.speed_tolerance_mach,
            )
        return (
            np.asarray(distance_nm, dtype=np.float64),
            np.asarray(alt_diff_ft, dtype=np.float64),
            within_lateral & within_altitude & within_speed,
        )

    def query(self, acid: str, acidx: int, name: str, queryable):
        track_temporal_state = bool(getattr(queryable, "track_temporal_state", False))
        if isinstance(queryable, Waypoint) and not track_temporal_state:
            route_idx = self._route_index(acidx, name)
            target = (
                queryable.target_from_route(acidx, route_idx)
                if route_idx is not None
                else queryable.target
            )
            return WaypointResult.for_aircraft(
                queryable,
                acidx,
                target=target,
                current=queryable.current_state(acidx, target),
                route=queryable.route_state(acidx, route_idx),
            )
        if not track_temporal_state:
            return queryable.result_type.for_aircraft(queryable, acidx)
        if isinstance(queryable, QueryRegion):
            current = queryable.contains_aircraft(acidx)
            event = self._event(
                current,
                _row_at(self._table, self.env, acidx),
                self._table.col.get(name),
                self._table["held_step_substeps"],
                self._table["held_total_s"],
            )
            return RegionResult.for_aircraft(
                queryable,
                acidx,
                current=RegionCurrent(inside=event.current),
                step=RegionStep(inside=event.during_step),
                time=event.time,
            )
        if isinstance(queryable, Waypoint):
            return self._waypoint_result(acid, acidx, name, queryable)
        return queryable.result_type.for_aircraft(queryable, acidx)

    def _waypoint_result(
        self,
        acid: str,
        acidx: int,
        name: str,
        queryable: Waypoint,
    ) -> WaypointResult:
        table = self._table
        row = _row_at(table, self.env, acidx)
        col = table.col.get(name)
        route_idx = self._route_index(acidx, name)
        target = (
            queryable.target_from_route(acidx, route_idx)
            if route_idx is not None
            else queryable.target
        )
        current = queryable.current_state(acidx, target)
        route = queryable.route_state(acidx, route_idx)
        satisfied = self._event(
            current.satisfied,
            row,
            col,
            table["held_step_substeps"],
            table["held_total_s"],
        )
        reached = self._event(
            queryable.reached_during_substep(acidx, route_idx),
            row,
            col,
            table["reached_step_substeps"],
            None,
        )
        min_distance = (
            float(table["min_distance_nm"][row, col])
            if row is not None
            and col is not None
            and math.isfinite(float(table["min_distance_nm"][row, col]))
            else current.distance_nm
        )
        min_abs_alt = (
            float(table["min_abs_alt_diff_ft"][row, col])
            if row is not None
            and col is not None
            and math.isfinite(float(table["min_abs_alt_diff_ft"][row, col]))
            else abs(current.alt_diff_ft)
        )
        return WaypointResult.for_aircraft(
            queryable,
            acidx,
            target=target,
            current=current,
            route=route,
            step=WaypointStep(
                satisfied=satisfied.during_step,
                reached=reached.during_step,
                min_distance_nm=min_distance,
                min_abs_alt_diff_ft=min_abs_alt,
            ),
            time=satisfied.time,
        )

    def _event(
        self,
        current: bool,
        row: int | None,
        col: int | None,
        step_substeps: np.ndarray,
        total_s: np.ndarray | None,
    ) -> StepEvent:
        if row is None or col is None:
            substeps = 0
            total = 0.0
        else:
            substeps = int(step_substeps[row, col])
            total = 0.0 if total_s is None else float(total_s[row, col])
        during_step_s = substeps * float(self.env.config.simdt) if self.env else 0.0
        return StepEvent(
            current=current,
            during_step=substeps > 0,
            substeps=substeps,
            time=StepTime(
                total_s=total,
                during_step_s=during_step_s,
            ),
        )


class TrafficMonitor:
    """Observe traffic during a step and retain per-agent traffic facts.

    Conflict and LoS state come from BlueSky's detector, which runs only every
    ``asas_dt`` (see :attr:`EnvConfig.asas_dt`), not every physics substep. The
    per-substep counts and durations therefore change only when it runs: each
    detection's result is held for the substeps until the next one. Detection
    replaces ``confpairs`` / ``lospairs`` / ``inconf`` with new objects, so
    :meth:`record_substep` rebuilds partners and increments only when one of
    them is a different object, and otherwise re-adds the cached increments.
    """

    def __init__(self, env=None) -> None:
        self.env = env
        self._table = AircraftTable(
            {
                "conflict_total_s": Column(0.0, np.float64),
                "los_total_s": Column(0.0, np.float64),
                "conflict_step_substeps": Column(0, np.int32, per_step=True),
                "los_step_substeps": Column(0, np.int32, per_step=True),
            }
        )
        # This step's partner callsigns, one entry per table row. Kept beside
        # the table rather than in it: BlueSky reports partners as callsign
        # strings and callers read them back as tuples, so an array would only
        # add conversions both ways.
        self._conflict_step_partners: list[set[str] | None] = []
        self._los_step_partners: list[set[str] | None] = []
        self.substep_count = 0
        self._current_conflict_partners: tuple[tuple[str, ...], ...] | None = None
        self._current_los_partners: tuple[tuple[str, ...], ...] | None = None
        # The detector output and aircraft ids the cached increments below were
        # built from; ``None`` forces a rebuild on the next substep.
        self._cd_seen: tuple[object, ...] | None = None
        self._inc_conf = np.zeros(0, dtype=np.int32)
        self._inc_conf_s = np.zeros(0, dtype=np.float64)
        self._inc_los = np.zeros(0, dtype=np.int32)
        self._inc_los_s = np.zeros(0, dtype=np.float64)

    def bind_env(self, env) -> None:
        self.env = env

    def clear(self) -> None:
        self._table.clear()
        self._conflict_step_partners = []
        self._los_step_partners = []
        self.begin_step()

    def begin_step(self) -> None:
        self.substep_count = 0
        self._sync_rows()
        self._table.reset_step()
        self._conflict_step_partners = [None] * len(self._table)
        self._los_step_partners = [None] * len(self._table)
        self._current_conflict_partners = None
        self._current_los_partners = None
        # Rebuild on the first substep: the step partner sets were just wiped,
        # and BlueSky's own reset empties ``confpairs`` in place - the same list
        # object - so identity alone would miss it.
        self._cd_seen = None

    def record_substep(self) -> None:
        self.substep_count += 1
        simdt = float(self.env.config.simdt) if self.env is not None else 0.0
        self._sync_rows()

        cd = bs.traf.cd
        seen = (cd.confpairs, cd.lospairs, cd.inconf, self._table.ids)
        if self._cd_seen is None or any(
            now is not before for now, before in zip(seen, self._cd_seen)
        ):
            self._refresh_detection(simdt)
            self._cd_seen = seen

        table = self._table
        table["conflict_step_substeps"] += self._inc_conf
        table["conflict_total_s"] += self._inc_conf_s
        table["los_step_substeps"] += self._inc_los
        table["los_total_s"] += self._inc_los_s

    def _refresh_detection(self, simdt: float) -> None:
        """Rebuild partners and per-substep increments from BlueSky's detector.

        Merging this detection's partners into the step's partner sets once is
        the same as merging them every substep it stays current: a set union
        with the same members changes nothing.
        """
        conf_partners, los_partners = self._build_current_partner_sets()
        self._current_conflict_partners = tuple(
            () if partners is None else tuple(sorted(partners))
            for partners in conf_partners
        )
        self._current_los_partners = tuple(
            () if partners is None else tuple(sorted(partners))
            for partners in los_partners
        )

        n = len(self._table)
        inconf = np.asarray(bs.traf.cd.inconf, dtype=np.bool_)[:n]
        if inconf.size < n:
            inconf = np.pad(inconf, (0, n - inconf.size), constant_values=False)
        self._inc_conf = inconf.astype(np.int32)
        self._inc_conf_s = inconf.astype(np.float64) * simdt

        los_mask = np.zeros(n, dtype=bool)
        los_rows = [row for row, partners in enumerate(los_partners) if partners]
        if los_rows:
            los_mask[np.asarray(los_rows, dtype=np.intp)] = True
        self._inc_los = los_mask.astype(np.int32)
        self._inc_los_s = los_mask.astype(np.float64) * simdt

        for row, partners in enumerate(conf_partners):
            if not partners:
                continue
            step_partners = self._conflict_step_partners[row]
            if step_partners is None:
                self._conflict_step_partners[row] = set(partners)
            else:
                step_partners.update(partners)
        for row, partners in enumerate(los_partners):
            if not partners:
                continue
            step_partners = self._los_step_partners[row]
            if step_partners is None:
                self._los_step_partners[row] = set(partners)
            else:
                step_partners.update(partners)

    def build_separation_context(self, acid: str, acidx: int) -> SeparationContext:
        conf_partners, los_partners = self._current_partner_lists()

        simdt = float(self.env.config.simdt) if self.env is not None else 0.0
        table = self._table
        row = _row_at(table, self.env, acidx)
        if row is None:
            conflict_substeps = 0
            los_substeps = 0
            conflict_total_s = 0.0
            los_total_s = 0.0
            current_conflict_partners: tuple[str, ...] = ()
            current_los_partners: tuple[str, ...] = ()
            conflict_step_partners: tuple[str, ...] = ()
            los_step_partners: tuple[str, ...] = ()
        else:
            conflict_substeps = int(table["conflict_step_substeps"][row])
            los_substeps = int(table["los_step_substeps"][row])
            conflict_total_s = float(table["conflict_total_s"][row])
            los_total_s = float(table["los_total_s"][row])
            current_conflict_partners = conf_partners[row]
            current_los_partners = los_partners[row]
            conflict_step = self._conflict_step_partners[row]
            los_step = self._los_step_partners[row]
            conflict_step_partners = (
                () if conflict_step is None else tuple(sorted(conflict_step))
            )
            los_step_partners = () if los_step is None else tuple(sorted(los_step))
        return SeparationContext(
            conflict=SeparationEvent(
                current=bool(bs.traf.cd.inconf[acidx]),
                during_step=conflict_substeps > 0,
                substeps=conflict_substeps,
                partners=current_conflict_partners,
                step_partners=conflict_step_partners,
                time=StepTime(
                    total_s=conflict_total_s,
                    during_step_s=conflict_substeps * simdt,
                ),
            ),
            los=SeparationEvent(
                current=bool(current_los_partners),
                during_step=los_substeps > 0,
                substeps=los_substeps,
                partners=current_los_partners,
                step_partners=los_step_partners,
                time=StepTime(
                    total_s=los_total_s,
                    during_step_s=los_substeps * simdt,
                ),
            ),
        )

    def build_separation_info(self, acid: str, acidx: int) -> dict:
        return self.build_separation_context(acid, acidx).as_info()

    def _sync_rows(self) -> None:
        """Rows to BlueSky's aircraft, carrying the partner lists alongside."""
        take = self._table.sync_rows(_aircraft_keys(self.env))
        if take is None:
            return
        self._conflict_step_partners = [
            self._conflict_step_partners[i] if i >= 0 else None for i in take
        ]
        self._los_step_partners = [
            self._los_step_partners[i] if i >= 0 else None for i in take
        ]

    def _current_partner_lists(
        self,
    ) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]:
        if (
            self._current_conflict_partners is None
            or self._current_los_partners is None
        ):
            conf_partners, los_partners = self._build_current_partner_sets()
            self._current_conflict_partners = tuple(
                () if partners is None else tuple(sorted(partners))
                for partners in conf_partners
            )
            self._current_los_partners = tuple(
                () if partners is None else tuple(sorted(partners))
                for partners in los_partners
            )
        return self._current_conflict_partners, self._current_los_partners

    def _build_current_partner_sets(
        self,
    ) -> tuple[list[set[str] | None], list[set[str] | None]]:
        self._sync_rows()
        # Freshly synced, the rows are in ``bs.traf.id`` order.
        row_of = {acid: row for row, acid in enumerate(bs.traf.id)}
        conf_partners: list[set[str] | None] = [None] * len(self._table)
        los_partners: list[set[str] | None] = [None] * len(self._table)
        for a, b in bs.traf.cd.confpairs:
            row = row_of.get(str(a))
            if row is not None:
                partners = conf_partners[row]
                if partners is None:
                    conf_partners[row] = {str(b)}
                else:
                    partners.add(str(b))
        for a, b in bs.traf.cd.lospairs:
            row = row_of.get(str(a))
            if row is not None:
                partners = los_partners[row]
                if partners is None:
                    los_partners[row] = {str(b)}
                else:
                    partners.add(str(b))
        return conf_partners, los_partners


class AgentInfoBuilder:
    """Build the PettingZoo info dict returned for live agents."""

    def __init__(self, env=None) -> None:
        self.env = env

    def bind_env(self, env) -> None:
        self.env = env

    def build(self, agent_ids: SequenceABC[str] | None = None) -> dict:
        if self.env is None:
            raise RuntimeError("AgentInfoBuilder env has not been set.")
        config = self.env.config
        aircraft_spawn_time = self.env._aircraft_spawn_time
        traffic_monitor = self.env._traffic_monitor
        info = {}
        airspace = self.env.episode_airspace_bounds
        traf = bs.traf
        sim_time = bs.sim.simt
        indexed_agent_ids: Iterable[tuple[int, str]]
        if agent_ids is None:
            indexed_agent_ids = enumerate(traf.id)
        else:
            live_index = {acid: acidx for acidx, acid in enumerate(traf.id)}
            indexed_agent_ids = (
                (live_index[acid], acid) for acid in agent_ids if acid in live_index
            )

        for acidx, acid in indexed_agent_ids:
            lat_deg = float(traf.lat[acidx])
            lon_deg = float(traf.lon[acidx])
            alt_ft = float(traf.alt[acidx] / ft)
            in_airspace = (
                airspace.contains(lat_deg, lon_deg, alt_ft)
                if airspace is not None
                else True
            )
            info[acid] = {
                "acid": acid,
                "acidx": acidx,
                "type": traf.type[acidx],
                "performance_model": config.performance_model,
                "phase": traf.perf.phase[acidx],
                "time_in_env": sim_time
                - aircraft_spawn_time.get(
                    acid,
                    sim_time,
                ),
                "in_airspace": in_airspace,
                "task": {},
                "autopilot": {
                    "lnav": bool(traf.swlnav[acidx]),
                    "vnav": bool(traf.swvnav[acidx]),
                    "lnav_vnav": bool(traf.swlnav[acidx]) and bool(traf.swvnav[acidx]),
                },
                "substeps": traffic_monitor.substep_count,
                "separation": traffic_monitor.build_separation_info(acid, acidx),
            }
        return info
