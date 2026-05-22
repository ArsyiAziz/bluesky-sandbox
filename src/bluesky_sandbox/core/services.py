from __future__ import annotations

import math
from collections.abc import Iterator
from collections.abc import Sequence as SequenceABC

import bluesky as bs
import numpy as np
from bluesky.tools.aero import ft, kts
from bluesky.tools.geo import qdrdist
from gymnasium.spaces import Box, Dict, Sequence

from bluesky_sandbox.config import EnvConfig
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
from bluesky_sandbox.sim.bounds import (
    BoxFootprint,
    ConstantAltitudeBand,
    DiskFootprint,
    RegionBounds,
)
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


def _region_contains_many(
    region: QueryRegion,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    alt_ft: np.ndarray,
) -> np.ndarray | None:
    """Vectorized containment for common region shapes.

    Returns ``None`` for uncommon bounds so callers can fall back to the
    polymorphic scalar API without complicating every footprint class.
    """
    bounds = region.bounds
    if not isinstance(bounds, RegionBounds):
        return None

    footprint = bounds.footprint
    altitude = bounds.altitude
    if isinstance(footprint, BoxFootprint):
        mask = (
            (footprint.lat_min_deg < lat_deg)
            & (lat_deg < footprint.lat_max_deg)
            & (footprint.lon_min_deg < lon_deg)
            & (lon_deg < footprint.lon_max_deg)
        )
    elif isinstance(footprint, DiskFootprint):
        (lat_min, lat_max), (lon_min, lon_max) = footprint.bounding_box
        mask = (
            (lat_min <= lat_deg)
            & (lat_deg <= lat_max)
            & (lon_min <= lon_deg)
            & (lon_deg <= lon_max)
        )
        x_nm = (lon_deg - footprint.center.lon_deg) * 60.0 * footprint._frame._cos_lat
        y_nm = (lat_deg - footprint.center.lat_deg) * 60.0
        mask &= (x_nm * x_nm + y_nm * y_nm) <= footprint.radius_nm * footprint.radius_nm
    else:
        return None

    mask &= (bounds.alt_min_ft <= alt_ft) & (alt_ft <= bounds.alt_max_ft)
    if isinstance(altitude, ConstantAltitudeBand):
        return mask

    candidate_rows = np.flatnonzero(mask)
    for row in candidate_rows:
        if not altitude.contains(
            float(lat_deg[row]),
            float(lon_deg[row]),
            float(alt_ft[row]),
        ):
            mask[row] = False
    return mask


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


def _param_alt_range(alt_p) -> tuple[float, float] | None:
    """Extract a finite ``(lo, hi)`` from a spawn ``params['alt_ft']`` value."""
    if alt_p is None:
        return None
    if isinstance(alt_p, (int, float)):
        lo = hi = float(alt_p)
    elif isinstance(alt_p, tuple):
        lo, hi = alt_p
    else:
        try:
            lo, hi = alt_p.support()
        except Exception:
            return None
    lo, hi = float(lo), float(hi)
    return (lo, hi) if math.isfinite(lo) and math.isfinite(hi) else None


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
        switch_commands = {
            field.meta.name: field.switch_command(value)
            for field, value in switch_fields
        }
        switch_on = {
            field.meta.name: (
                command is True or (command is None and field.current_switch_state(idx))
            )
            for field, _value in switch_fields
            for command in (switch_commands[field.meta.name],)
        }
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
        critic_own_specs, critic_own_dim, critic_own_raw = _ownship_pack(critic_own_fields)
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
                alt_range_override=_param_alt_range(region.params.get("alt_ft")),
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


class QueryStateMonitor:
    """Track built-in queryable event state across simulator substeps."""

    def __init__(self, env=None) -> None:
        self.env = env
        self._aircraft_ids: tuple[str, ...] = ()
        self._aircraft_index: dict[str, int] = {}
        self._queryable_names: tuple[str, ...] = ()
        self._queryable_index: dict[str, int] = {}
        self._tracked_names: tuple[str, ...] = ()
        self._tracked_queryables: tuple[object, ...] = ()
        self._tracked_index: dict[str, int] = {}
        self._tracked_region_cols: tuple[int, ...] = ()
        self._tracked_waypoint_cols: tuple[int, ...] = ()
        self._route_indices = np.full((0, 0), -1, dtype=np.int32)
        self._region_total_s = np.zeros((0, 0), dtype=np.float64)
        self._region_step_substeps = np.zeros((0, 0), dtype=np.int32)
        self._waypoint_satisfied_total_s = np.zeros((0, 0), dtype=np.float64)
        self._waypoint_satisfied_step_substeps = np.zeros((0, 0), dtype=np.int32)
        self._waypoint_reached_step_substeps = np.zeros((0, 0), dtype=np.int32)
        self._waypoint_min_distance_nm = np.full((0, 0), math.inf, dtype=np.float64)
        self._waypoint_min_abs_alt_diff_ft = np.full(
            (0, 0),
            math.inf,
            dtype=np.float64,
        )
        # Per-step route-target cache for substep dwell tracking:
        # col -> ((aircraft ids, route-index bytes), target arrays).
        self._waypoint_target_cache: dict[int, tuple[tuple, tuple]] = {}

    def bind_env(self, env) -> None:
        self.env = env

    def clear(self) -> None:
        self._aircraft_ids = ()
        self._aircraft_index = {}
        self._route_indices = np.full(
            (0, len(self._queryable_names)),
            -1,
            dtype=np.int32,
        )
        self._region_total_s = np.zeros(
            (0, len(self._tracked_names)),
            dtype=np.float64,
        )
        self._waypoint_satisfied_total_s = np.zeros(
            (0, len(self._tracked_names)),
            dtype=np.float64,
        )
        self.begin_step()

    def set_aircraft_route(
        self,
        acid: str,
        route_names: SequenceABC[str] | None,
    ) -> None:
        """Remember which BlueSky route index corresponds to each query name."""
        self._ensure_queryable_index()
        self._sync_aircraft_rows(reset_step=False)
        row = self._aircraft_index.get(acid)
        if row is None:
            return
        self._route_indices[row, :] = -1
        if not route_names:
            return
        for route_idx, name in enumerate(route_names):
            col = self._queryable_index.get(name)
            if col is not None and self._route_indices[row, col] < 0:
                self._route_indices[row, col] = route_idx

    def clear_aircraft_route(self, acid: str) -> None:
        row = self._aircraft_index.get(acid)
        if row is not None and self._route_indices.size:
            self._route_indices[row, :] = -1

    def _route_index(self, acid: str, name: str) -> int | None:
        row = self._aircraft_index.get(acid)
        col = self._queryable_index.get(name)
        if row is None or col is None or self._route_indices.size == 0:
            return None
        route_idx = int(self._route_indices[row, col])
        return route_idx if route_idx >= 0 else None

    def begin_step(self) -> None:
        self._ensure_queryable_index()
        self._ensure_tracked_queryables()
        self._sync_aircraft_rows(reset_step=True)
        # Route targets may be edited by this step's actions (dispatched after
        # begin_step, before the substep loop) - rebuild lazily per step.
        self._waypoint_target_cache.clear()

    def _ensure_queryable_index(self) -> None:
        if self.env is None:
            names: tuple[str, ...] = ()
        else:
            names = tuple(self.env.episode_queryables)
        if names == self._queryable_names:
            return
        old_index = self._queryable_index
        old_route_indices = self._route_indices
        self._queryable_names = names
        self._queryable_index = {name: col for col, name in enumerate(names)}
        new_route_indices = np.full(
            (len(self._aircraft_ids), len(names)),
            -1,
            dtype=np.int32,
        )
        for name, old_col in old_index.items():
            new_col = self._queryable_index.get(name)
            if new_col is not None and old_col < old_route_indices.shape[1]:
                new_route_indices[:, new_col] = old_route_indices[:, old_col]
        self._route_indices = new_route_indices

    def _ensure_tracked_queryables(self) -> None:
        if self.env is None:
            names: tuple[str, ...] = ()
            queryables: tuple[object, ...] = ()
        else:
            items = tuple(
                (name, queryable)
                for name, queryable in self.env.episode_queryables.items()
                if bool(getattr(queryable, "track_temporal_state", False))
            )
            names = tuple(name for name, _queryable in items)
            queryables = tuple(queryable for _name, queryable in items)
        if names == self._tracked_names:
            self._tracked_queryables = queryables
            self._refresh_tracked_type_columns()
            return
        old_index = self._tracked_index
        old_region_total_s = self._region_total_s
        old_waypoint_satisfied_total_s = self._waypoint_satisfied_total_s
        self._tracked_names = names
        self._tracked_queryables = queryables
        self._tracked_index = {name: col for col, name in enumerate(names)}
        self._refresh_tracked_type_columns()
        shape = (len(self._aircraft_ids), len(names))
        self._region_total_s = np.zeros(shape, dtype=np.float64)
        self._waypoint_satisfied_total_s = np.zeros(shape, dtype=np.float64)
        for name, old_col in old_index.items():
            new_col = self._tracked_index.get(name)
            if new_col is None:
                continue
            if old_col < old_region_total_s.shape[1]:
                self._region_total_s[:, new_col] = old_region_total_s[:, old_col]
            if old_col < old_waypoint_satisfied_total_s.shape[1]:
                self._waypoint_satisfied_total_s[:, new_col] = (
                    old_waypoint_satisfied_total_s[:, old_col]
                )
        self._reset_step_arrays()

    def _refresh_tracked_type_columns(self) -> None:
        self._tracked_region_cols = tuple(
            col
            for col, queryable in enumerate(self._tracked_queryables)
            if isinstance(queryable, QueryRegion)
        )
        self._tracked_waypoint_cols = tuple(
            col
            for col, queryable in enumerate(self._tracked_queryables)
            if isinstance(queryable, Waypoint)
        )

    def _sync_aircraft_rows(self, *, reset_step: bool) -> None:
        current_ids = tuple(str(acid) for acid in bs.traf.id)
        if current_ids == self._aircraft_ids:
            if reset_step:
                self._reset_step_arrays()
            return

        old_ids = self._aircraft_ids
        old_index = self._aircraft_index
        old_route_indices = self._route_indices
        old_region_total_s = self._region_total_s
        old_waypoint_satisfied_total_s = self._waypoint_satisfied_total_s
        old_region_step_substeps = self._region_step_substeps
        old_waypoint_satisfied_step_substeps = self._waypoint_satisfied_step_substeps
        old_waypoint_reached_step_substeps = self._waypoint_reached_step_substeps
        old_waypoint_min_distance_nm = self._waypoint_min_distance_nm
        old_waypoint_min_abs_alt_diff_ft = self._waypoint_min_abs_alt_diff_ft

        self._aircraft_ids = current_ids
        self._aircraft_index = {acid: row for row, acid in enumerate(current_ids)}
        route_shape = (len(current_ids), len(self._queryable_names))
        tracked_shape = (len(current_ids), len(self._tracked_names))
        self._route_indices = np.full(route_shape, -1, dtype=np.int32)
        self._region_total_s = np.zeros(tracked_shape, dtype=np.float64)
        self._waypoint_satisfied_total_s = np.zeros(tracked_shape, dtype=np.float64)
        self._region_step_substeps = np.zeros(tracked_shape, dtype=np.int32)
        self._waypoint_satisfied_step_substeps = np.zeros(tracked_shape, dtype=np.int32)
        self._waypoint_reached_step_substeps = np.zeros(tracked_shape, dtype=np.int32)
        self._waypoint_min_distance_nm = np.full(
            tracked_shape,
            math.inf,
            dtype=np.float64,
        )
        self._waypoint_min_abs_alt_diff_ft = np.full(
            tracked_shape,
            math.inf,
            dtype=np.float64,
        )

        for acid in old_ids:
            old_row = old_index[acid]
            new_row = self._aircraft_index.get(acid)
            if new_row is None:
                continue
            if old_row < old_route_indices.shape[0]:
                self._route_indices[new_row, :] = old_route_indices[old_row, :]
            if old_row < old_region_total_s.shape[0]:
                self._region_total_s[new_row, :] = old_region_total_s[old_row, :]
            if old_row < old_waypoint_satisfied_total_s.shape[0]:
                self._waypoint_satisfied_total_s[new_row, :] = (
                    old_waypoint_satisfied_total_s[old_row, :]
                )
            if not reset_step and old_row < old_region_step_substeps.shape[0]:
                self._region_step_substeps[new_row, :] = old_region_step_substeps[
                    old_row,
                    :,
                ]
                self._waypoint_satisfied_step_substeps[new_row, :] = (
                    old_waypoint_satisfied_step_substeps[old_row, :]
                )
                self._waypoint_reached_step_substeps[new_row, :] = (
                    old_waypoint_reached_step_substeps[old_row, :]
                )
                self._waypoint_min_distance_nm[new_row, :] = (
                    old_waypoint_min_distance_nm[old_row, :]
                )
                self._waypoint_min_abs_alt_diff_ft[new_row, :] = (
                    old_waypoint_min_abs_alt_diff_ft[old_row, :]
                )

    def _reset_step_arrays(self) -> None:
        shape = (len(self._aircraft_ids), len(self._tracked_names))
        self._region_step_substeps = np.zeros(shape, dtype=np.int32)
        self._waypoint_satisfied_step_substeps = np.zeros(shape, dtype=np.int32)
        self._waypoint_reached_step_substeps = np.zeros(shape, dtype=np.int32)
        self._waypoint_min_distance_nm = np.full(
            shape,
            math.inf,
            dtype=np.float64,
        )
        self._waypoint_min_abs_alt_diff_ft = np.full(
            shape,
            math.inf,
            dtype=np.float64,
        )

    def record_substep(self) -> None:
        if self.env is None:
            raise RuntimeError("QueryStateMonitor env has not been set.")
        if not self._tracked_queryables:
            return
        self._sync_aircraft_rows(reset_step=False)
        simdt = float(self.env.config.simdt)
        n = len(self._aircraft_ids)
        if n == 0:
            return

        lat_deg = np.asarray(bs.traf.lat, dtype=np.float64)[:n]
        lon_deg = np.asarray(bs.traf.lon, dtype=np.float64)[:n]
        alt_ft = np.asarray(bs.traf.alt, dtype=np.float64)[:n] / ft

        for col in self._tracked_region_cols:
            queryable = self._tracked_queryables[col]
            inside = _region_contains_many(queryable, lat_deg, lon_deg, alt_ft)
            if inside is None:
                inside = np.fromiter(
                    (queryable.contains_aircraft(acidx) for acidx in range(n)),
                    dtype=bool,
                    count=n,
                )
            self._region_step_substeps[:, col] += inside.astype(np.int32)
            self._region_total_s[:, col] += inside.astype(np.float64) * simdt

        if not self._tracked_waypoint_cols:
            return

        cas_kts = np.asarray(bs.traf.cas, dtype=np.float64)[:n] / kts
        trk_deg = np.asarray(bs.traf.trk, dtype=np.float64)[:n]
        reached_rows = np.zeros(n, dtype=bool)
        try:
            reached_indices = np.asarray(tuple(bs.traf.ap.idxreached), dtype=np.int64)
            reached_indices = reached_indices[
                (0 <= reached_indices) & (reached_indices < n)
            ]
            reached_rows[reached_indices] = True
        except (AttributeError, TypeError, ValueError):
            pass
        active_route_idx = np.full(n, -1, dtype=np.int32)
        routes = bs.traf.ap.route
        for acidx in range(min(n, len(routes))):
            try:
                active_route_idx[acidx] = (
                    -1 if routes[acidx].iactwp is None else int(routes[acidx].iactwp)
                )
            except (TypeError, ValueError):
                active_route_idx[acidx] = -1
        swlnav = np.asarray(bs.traf.swlnav, dtype=bool)[:n]
        just_reached_idx = np.where(swlnav, active_route_idx - 1, active_route_idx)

        for col in self._tracked_waypoint_cols:
            queryable = self._tracked_queryables[col]
            name = self._tracked_names[col]
            route_col = self._queryable_index.get(name)
            route_indices = (
                self._route_indices[:, route_col]
                if route_col is not None and self._route_indices.size
                else np.full(n, -1, dtype=np.int32)
            )
            (
                distance_nm,
                alt_diff_ft,
                satisfied,
            ) = self._waypoint_tracking_arrays(
                col,
                queryable,
                route_indices,
                lat_deg,
                lon_deg,
                alt_ft,
                cas_kts,
                trk_deg,
            )
            np.minimum(
                self._waypoint_min_distance_nm[:, col],
                distance_nm,
                out=self._waypoint_min_distance_nm[:, col],
            )
            abs_alt_diff = np.abs(alt_diff_ft)
            finite_alt = np.isfinite(abs_alt_diff)
            np.minimum(
                self._waypoint_min_abs_alt_diff_ft[:, col],
                np.where(finite_alt, abs_alt_diff, math.inf),
                out=self._waypoint_min_abs_alt_diff_ft[:, col],
            )
            self._waypoint_satisfied_step_substeps[:, col] += satisfied.astype(
                np.int32
            )
            self._waypoint_satisfied_total_s[:, col] += (
                satisfied.astype(np.float64) * simdt
            )
            reached = reached_rows & (route_indices == just_reached_idx)
            self._waypoint_reached_step_substeps[:, col] += reached.astype(np.int32)

    def _waypoint_target_arrays(
        self,
        col: int,
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
        key = (self._aircraft_ids, route_indices.tobytes())
        cached = self._waypoint_target_cache.get(col)
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
        self._waypoint_target_cache[col] = (key, arrays)
        return arrays

    def _waypoint_tracking_arrays(
        self,
        col: int,
        queryable: Waypoint,
        route_indices: np.ndarray,
        lat_deg: np.ndarray,
        lon_deg: np.ndarray,
        alt_ft: np.ndarray,
        cas_kts: np.ndarray,
        trk_deg: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = lat_deg.size
        target = queryable.target
        (
            target_lat,
            target_lon,
            target_alt_ft,
            target_speed_kts,
        ) = self._waypoint_target_arrays(col, queryable, route_indices, n)

        qdr_deg, distance_nm = qdrdist(lat_deg, lon_deg, target_lat, target_lon)
        del qdr_deg, trk_deg, cas_kts
        alt_diff_ft = alt_ft - target_alt_ft
        within_lateral = (
            np.ones(n, dtype=bool)
            if target.reach_radius_nm is None
            else distance_nm <= target.reach_radius_nm
        )
        within_altitude = (
            np.ones(n, dtype=bool)
            if target.alt_tolerance_ft is None
            else np.isnan(target_alt_ft) | (np.abs(alt_diff_ft) <= target.alt_tolerance_ft)
        )
        # Speed constraint per aircraft, regime-aware (Mach above the CAS/Mach
        # crossover, CAS below) via the shared helper, so the dwell-tracking mask
        # agrees with current_state and stays well-defined for any sampled target.
        within_speed = np.ones(n, dtype=bool)
        if target.speed_tolerance_kts is not None or target.speed_tolerance_mach is not None:
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
            route_idx = self._route_index(acid, name)
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
                self._aircraft_index.get(acid),
                self._tracked_index.get(name),
                self._region_step_substeps,
                self._region_total_s,
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
        row = self._aircraft_index.get(acid)
        col = self._tracked_index.get(name)
        route_idx = self._route_index(acid, name)
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
            self._waypoint_satisfied_step_substeps,
            self._waypoint_satisfied_total_s,
        )
        reached = self._event(
            queryable.reached_during_substep(acidx, route_idx),
            row,
            col,
            self._waypoint_reached_step_substeps,
            None,
        )
        min_distance = (
            float(self._waypoint_min_distance_nm[row, col])
            if row is not None
            and col is not None
            and math.isfinite(float(self._waypoint_min_distance_nm[row, col]))
            else current.distance_nm
        )
        min_abs_alt = (
            float(self._waypoint_min_abs_alt_diff_ft[row, col])
            if row is not None
            and col is not None
            and math.isfinite(float(self._waypoint_min_abs_alt_diff_ft[row, col]))
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
    """Observe traffic during a step and retain per-agent traffic facts."""

    def __init__(self, env=None) -> None:
        self.env = env
        self._aircraft_ids: tuple[str, ...] = ()
        self._aircraft_index: dict[str, int] = {}
        self._conflict_step_substeps = np.zeros(0, dtype=np.int32)
        self._los_step_substeps = np.zeros(0, dtype=np.int32)
        self._conflict_total_s = np.zeros(0, dtype=np.float64)
        self._los_total_s = np.zeros(0, dtype=np.float64)
        self._conflict_step_partners: list[set[str] | None] = []
        self._los_step_partners: list[set[str] | None] = []
        self.substep_count = 0
        self._current_conflict_partners: list[tuple[str, ...]] | None = None
        self._current_los_partners: list[tuple[str, ...]] | None = None

    def bind_env(self, env) -> None:
        self.env = env

    def clear(self) -> None:
        self._aircraft_ids = ()
        self._aircraft_index = {}
        self._conflict_total_s = np.zeros(0, dtype=np.float64)
        self._los_total_s = np.zeros(0, dtype=np.float64)
        # Substep accumulators must shrink with the ids: an episode that starts
        # with zero live aircraft (deferred spawn_time) hits the empty==empty
        # early-return in _sync_aircraft_rows, so stale-sized arrays from the
        # previous episode would never be resized before use.
        self._conflict_step_substeps = np.zeros(0, dtype=np.int32)
        self._los_step_substeps = np.zeros(0, dtype=np.int32)
        self.begin_step()

    def begin_step(self) -> None:
        self.substep_count = 0
        self._sync_aircraft_rows(reset_step=True)
        self._conflict_step_substeps.fill(0)
        self._los_step_substeps.fill(0)
        self._conflict_step_partners = [None for _acid in self._aircraft_ids]
        self._los_step_partners = [None for _acid in self._aircraft_ids]
        self._current_conflict_partners = None
        self._current_los_partners = None

    def record_substep(self) -> None:
        self.substep_count += 1
        simdt = float(self.env.config.simdt) if self.env is not None else 0.0
        self._sync_aircraft_rows(reset_step=False)

        conf_partners, los_partners = self._build_current_partner_sets()
        self._current_conflict_partners = tuple(
            () if partners is None else tuple(sorted(partners))
            for partners in conf_partners
        )
        self._current_los_partners = tuple(
            () if partners is None else tuple(sorted(partners))
            for partners in los_partners
        )

        n = len(self._aircraft_ids)
        inconf = np.asarray(bs.traf.cd.inconf, dtype=bool)[:n]
        if inconf.size < n:
            inconf = np.pad(inconf, (0, n - inconf.size), constant_values=False)
        self._conflict_step_substeps += inconf.astype(np.int32)
        self._conflict_total_s += inconf.astype(np.float64) * simdt

        los_mask = np.zeros(n, dtype=bool)
        los_rows = [row for row, partners in enumerate(los_partners) if partners]
        if los_rows:
            los_mask[np.asarray(los_rows, dtype=np.intp)] = True
        self._los_step_substeps += los_mask.astype(np.int32)
        self._los_total_s += los_mask.astype(np.float64) * simdt

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
        row = self._aircraft_index.get(acid)
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
            conflict_substeps = int(self._conflict_step_substeps[row])
            los_substeps = int(self._los_step_substeps[row])
            conflict_total_s = float(self._conflict_total_s[row])
            los_total_s = float(self._los_total_s[row])
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

    def _sync_aircraft_rows(self, *, reset_step: bool) -> None:
        current_ids = tuple(str(acid) for acid in bs.traf.id)
        if current_ids == self._aircraft_ids:
            if reset_step:
                self._conflict_step_partners = [
                    None for _acid in self._aircraft_ids
                ]
                self._los_step_partners = [None for _acid in self._aircraft_ids]
            return

        old_ids = self._aircraft_ids
        old_index = self._aircraft_index
        old_conflict_total_s = self._conflict_total_s
        old_los_total_s = self._los_total_s
        old_conflict_step_substeps = self._conflict_step_substeps
        old_los_step_substeps = self._los_step_substeps
        old_conflict_step_partners = self._conflict_step_partners
        old_los_step_partners = self._los_step_partners

        self._aircraft_ids = current_ids
        self._aircraft_index = {acid: row for row, acid in enumerate(current_ids)}
        n = len(current_ids)
        self._conflict_total_s = np.zeros(n, dtype=np.float64)
        self._los_total_s = np.zeros(n, dtype=np.float64)
        self._conflict_step_substeps = np.zeros(n, dtype=np.int32)
        self._los_step_substeps = np.zeros(n, dtype=np.int32)
        self._conflict_step_partners = [None for _acid in current_ids]
        self._los_step_partners = [None for _acid in current_ids]

        for acid in old_ids:
            old_row = old_index[acid]
            new_row = self._aircraft_index.get(acid)
            if new_row is None:
                continue
            self._conflict_total_s[new_row] = old_conflict_total_s[old_row]
            self._los_total_s[new_row] = old_los_total_s[old_row]
            if not reset_step:
                self._conflict_step_substeps[new_row] = old_conflict_step_substeps[
                    old_row
                ]
                self._los_step_substeps[new_row] = old_los_step_substeps[old_row]
                old_conflict = old_conflict_step_partners[old_row]
                old_los = old_los_step_partners[old_row]
                self._conflict_step_partners[new_row] = (
                    None if old_conflict is None else set(old_conflict)
                )
                self._los_step_partners[new_row] = (
                    None if old_los is None else set(old_los)
                )

    def _current_partner_lists(
        self,
    ) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]:
        if self._current_conflict_partners is None or self._current_los_partners is None:
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
        self._sync_aircraft_rows(reset_step=False)
        conf_partners: list[set[str] | None] = [
            None for _acid in self._aircraft_ids
        ]
        los_partners: list[set[str] | None] = [None for _acid in self._aircraft_ids]
        for a, b in bs.traf.cd.confpairs:
            row = self._aircraft_index.get(str(a))
            if row is not None:
                partners = conf_partners[row]
                if partners is None:
                    conf_partners[row] = {str(b)}
                else:
                    partners.add(str(b))
        for a, b in bs.traf.cd.lospairs:
            row = self._aircraft_index.get(str(a))
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
                "time_in_env": sim_time - aircraft_spawn_time.get(
                    acid,
                    sim_time,
                ),
                "in_airspace": in_airspace,
                "task": {},
                "autopilot": {
                    "lnav": bool(traf.swlnav[acidx]),
                    "vnav": bool(traf.swvnav[acidx]),
                    "lnav_vnav": bool(traf.swlnav[acidx])
                    and bool(traf.swvnav[acidx]),
                },
                "substeps": traffic_monitor.substep_count,
                "separation": traffic_monitor.build_separation_info(acid, acidx),
            }
        return info
