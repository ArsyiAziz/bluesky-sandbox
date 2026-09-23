from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
    get_args,
)

import bluesky as bs
import numpy as np
from pettingzoo import ParallelEnv

from bluesky_sandbox.config import (
    EnvConfig,
    normalize_spawn_aircraft_types,
)
from bluesky_sandbox.interface.fields.base import StepContext
from bluesky_sandbox.interface.fields.observations import (
    reset_all_field_state,
    set_action_space_bounds,
)
from bluesky_sandbox.interface.task import (
    AgentStepContext,
    BaseAgentInfo,
    BaseObs,
)
from bluesky_sandbox.sim.queryables import (
    Queryable,
    QueryRegion,
    RegionCurrent,
    RegionResult,
)
from bluesky_sandbox.sim.scenario import EpisodeSpec, Scenario
from bluesky_sandbox.sim.weather import WindField, wind_from_config
from bluesky_sandbox.ui.drivers import RenderMode, get_driver_class

from .runtime import BlueSkyRuntime
from .services import (
    ActionDispatcher,
    AgentInfoBuilder,
    ObservationAssembler,
    QueryStateMonitor,
    RenderableBuilder,
    TrafficMonitor,
)
from .spawning import SpawnGenerator
from .state import (
    AircraftControlState,
    AircraftControlStates,
    Callsign,
    ResolvedRoute,
    SpawnPosition,
    SpawnProgress,
    SpawnQueueItem,
)

# These moved to ``.state`` so ``.spawning`` could construct them without
# importing this module back. Re-exported here because they were part of this
# module's surface before the split.
__all__ = [
    "AircraftControlState",
    "AircraftControlStates",
    "BlueskyBaseEnvironment",
    "Callsign",
    "RenderMode",
    "ResolvedRoute",
    "SpawnPosition",
    "SpawnProgress",
    "SpawnQueueItem",
    "TaskHooks",
    "ViewSpec",
    "overridable",
]

if TYPE_CHECKING:
    from bluesky_sandbox.ui.drivers.panda3d.views.base import Panda3DView
    from bluesky_sandbox.ui.drivers.pygame.layout import _SpecTag
    from bluesky_sandbox.ui.drivers.pygame.views.base import PygameView


# View spec for the ``views=`` kwarg. Two shapes are accepted, validated
# against ``render_mode`` at construction:
#   - pygame:   a view class/instance, a HSplit/VSplit tag, or a nested
#               tuple/list interpreted as a split.
#   - panda3d:  a flat list of :class:`Panda3DView` instances (no
#               nesting - Panda3D has no draggable layout).
if TYPE_CHECKING:
    PygameViewSpec: TypeAlias = (
        type[PygameView] | PygameView | _SpecTag | tuple[Any, ...] | list[Any]
    )
    Panda3DViewSpec: TypeAlias = Sequence[Panda3DView]
    ViewSpec: TypeAlias = PygameViewSpec | Panda3DViewSpec
else:
    ViewSpec = Any
_VIEWS_BY_MODE = {"pygame", "panda3d"}


F = TypeVar("F", bound=Callable[..., Any])
AgentActions: TypeAlias = Mapping[str, Any]
AgentInfos: TypeAlias = dict[str, BaseAgentInfo]
AgentObservations: TypeAlias = dict[str, BaseObs]
AgentRewards: TypeAlias = dict[str, float]
DoneFlags: TypeAlias = dict[str, bool]
EnvOptions: TypeAlias = Mapping[str, Any]


def overridable(func: F) -> F:
    """Mark methods intended for environment subclasses to override.

    Tags the function so tooling (e.g. the designer catalog) can discover the
    available hooks by introspection rather than a hard-coded list.
    """
    func.__overridable__ = True  # type: ignore[attr-defined]
    return func


class TaskHooks(Protocol):
    """Task-author hook contract consumed by the runtime base.

    ``BlueskyEnv`` provides the default implementations and is the public class
    task authors subclass. This protocol lets the runtime know the methods it
    calls without making ``BlueskyBaseEnvironment`` itself the hook surface.
    """

    def on_episode_loaded(self, episode_spec: EpisodeSpec) -> None: ...

    def on_episode_reset(
        self,
        *,
        seed: int | None,
        options: EnvOptions | None,
    ) -> None: ...

    def on_before_spawn(self) -> None: ...

    def on_after_spawn(self, *, rng: np.random.Generator) -> None: ...

    def on_before_step(self) -> None: ...

    def on_sim_step(self) -> None: ...

    def on_agent_action(self, idx: int, action: Any) -> bool: ...

    def on_aircraft_spawned(
        self,
        callsign: Callsign,
        route: list[str] | None,
    ) -> None: ...

    def define_initial_aircraft_control_state(
        self,
        callsign: Callsign,
        route: list[str] | None,
    ) -> AircraftControlState: ...

    def on_agent_done(
        self,
        acid: Callsign,
        info: BaseAgentInfo,
        *,
        terminated: bool,
        truncated: bool,
    ) -> AircraftControlState: ...

    def on_before_agent_contexts(self) -> None: ...

    def define_agent_context(self, acid: Callsign, acidx: int) -> object: ...

    def reward(
        self,
        obs,
        action,
        terminated,
        truncated,
        context,
        info,
        rng,
    ) -> float: ...

    def terminated(self, obs, action, context, info, rng) -> bool: ...

    def truncated(self, obs, action, context, info, rng) -> bool: ...


class BlueskyBaseEnvironment(ParallelEnv):
    """Internal BlueSky multi-agent runner (PettingZoo ParallelEnv).

    Observation and action fields may carry per-field normalizers. Fields
    without a normalizer use raw physical values.

    Parameters
    ----------
    config:
        Internal EnvConfig instance. Task authors should normally inherit
        BlueskyEnv instead of constructing this runner directly.
    render_mode:
        How to display the simulation:

        * ``"qtgl"`` - open the full BlueSky QtGL radar window.
        * ``"pygame"`` - open a lightweight pygame top-down view in the
          bluesky-gym style.
        * ``"panda3d"`` - open an interactive Panda3D viewer in true-scale
          metres (orbit camera, click-to-select aircraft).
        * ``None`` - no rendering (default).
    """

    metadata = {
        "name": "bluesky-base-v0",
        "render_modes": [*get_args(RenderMode)],
    }

    def __init__(
        self,
        config: EnvConfig,
        scenario: Scenario,
        render_mode: RenderMode = None,
        realtime: bool = False,
        views: ViewSpec | None = None,
    ) -> None:
        """Construct the env.

        Parameters
        ----------
        views:
            Per-renderer layout spec. Shape depends on ``render_mode``:

            ``"pygame"`` - a :class:`PygameView` class (or instance), a
            nested tuple, or a :func:`HSplit` / :func:`VSplit` tag::

                # Vertical stack (legacy default).
                views=(VerticalView, HorizontalView)

                # Plan on the left, TSAS column on the right.
                from bluesky_sandbox.ui.drivers import HSplit
                views=HSplit(HorizontalView, TSASView)

                # Profile on top of (plan + TSAS side-by-side).
                views=(VerticalView, HSplit(HorizontalView, TSASView))

            Drag a panel header to rearrange at runtime; drag a divider
            to resize.

            ``"panda3d"`` - a flat list of :class:`Panda3DView`
            instances. Defaults to ``[WorldView(), TSASView()]``::

                views=[WorldView(), TSASView()]

            Rejected for ``"qtgl"`` and ``None`` (those drivers don't
            compose views).
        """
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}, "
                f"got {render_mode!r}"
            )

        self.render_mode = render_mode
        self.realtime = realtime

        if not isinstance(config, EnvConfig):
            raise TypeError(f"config must be EnvConfig, got {type(config)!r}")
        self.config = copy.deepcopy(config)
        self._hooks = cast(TaskHooks, self)

        self._aircraft_spawn_time: dict[str, float] = {}
        self._agent_context_cache: dict[str, AgentStepContext] = {}
        self._agent_context_cache_enabled = False

        # Aircraft scheduled to spawn later in the episode (spawn_time > 0).
        # Sorted ascending by spawn_time; drained at the top of each step()
        # when ``bs.sim.simt`` has caught up.
        # Consecutive fully-failed resets per region index. A
        # persistently unsatisfiable spawn guard warns once instead of silently
        # holding the region below its target. Cleared on a successful spawn.
        # Total spawns sampled at the most recent reset (live + queued +
        # already-dead). Used by HUD to render ``spawned X/N``.
        # Origin spawn-region index per live aircraft, and per-region target
        # live counts for regions in steady-state ``maintain`` mode. Together
        # they drive the top-up that holds each maintain region at its target.
        self._aircraft_region: dict[str, int] = {}
        self._aircraft_control_state: AircraftControlStates = {}

        self._live_info: AgentInfos = {}
        self.scenario = scenario
        self.episode_spec: EpisodeSpec = scenario.support()
        self.config.bind_env(self)

        self._action_dispatcher = ActionDispatcher()
        self._observation_assembler = ObservationAssembler()
        self._query_state_monitor = QueryStateMonitor()
        self._renderable_builder = RenderableBuilder()
        self._traffic_monitor = TrafficMonitor()
        self._agent_info_builder = AgentInfoBuilder()
        self._spawn_generator = SpawnGenerator()
        self._runtime = BlueSkyRuntime()
        self._action_dispatcher.bind_env(self)
        self._observation_assembler.bind_env(self)
        self._query_state_monitor.bind_env(self)
        self._renderable_builder.bind_env(self)
        self._traffic_monitor.bind_env(self)
        self._agent_info_builder.bind_env(self)
        self._spawn_generator.bind_env(self)
        self._runtime.bind_env(self)

        # Publish the (normalized) action-space bounds so a PrevActionNorm obs
        # field can size itself to the real action space instead of assuming a
        # range. Static per task (from the action fields' normalizers), so
        # resolving once here - before any observation_space query - is enough.
        try:
            action_box = self._observation_assembler.action_space(None)
            set_action_space_bounds(action_box.low, action_box.high)
        except Exception:
            pass  # leave PrevActionNorm on its [-1, 1] fallback

        self._n_substeps = int(round(self.config.dt / self.config.simdt))
        # Fields that keep per-aircraft state of their own (a rate, an
        # accumulator, a history). Resolved once: dt/simdt is commonly 100
        # substeps per env step, so walking every configured field each substep
        # to call a no-op would be the expensive way to do nothing.
        self._stateful_fields = tuple(self._collect_stateful_fields())
        # The airspace's wind, built once from the config's scalar settings.
        # Single source of truth: the runtime applies it, spawn clearance
        # predicts against it, and nothing re-derives a vector from
        # ``wind_dir_deg`` / ``wind_kts`` on its own.
        self._wind: WindField = wind_from_config(self.config)

        self._runtime.configure()

        driver_cls = get_driver_class(render_mode)
        driver_kwargs = {"realtime": realtime}
        if views is not None:
            if render_mode not in _VIEWS_BY_MODE:
                raise ValueError(
                    f"`views` is only supported by render_mode in "
                    f"{sorted(_VIEWS_BY_MODE)}, got {render_mode!r}"
                )
            driver_kwargs["views"] = views
        self._driver = driver_cls(**driver_kwargs)
        self._driver.bind_env(self)

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------

    @property
    def agents(self) -> list[str]:
        return list(self._controlled_live_agents)

    @property
    def episode_done(self) -> bool:
        """True when no active/background aircraft remain and no spawn is queued.

        With deferred-spawn configs (``SpawnRegion.spawn_time > 0``) the
        live-traffic list can transiently empty mid-episode while later
        cohorts are still queued. Steady-state ``maintain`` regions always have
        pending replenishment, so the episode never ends by drain - it runs
        until the task's ``truncated`` hook stops it.
        """
        if self._spawn_generator.has_maintain:
            return False
        return not self._aircraft_control_state and not self._spawn_generator.pending

    @property
    def has_future_agents(self) -> bool:
        """True when later steps can expose controllable agents."""
        return (
            self._spawn_generator.has_maintain
            or bool(self._spawn_generator.pending)
            or any(
                state is AircraftControlState.BACKGROUND
                for state in self._aircraft_control_state.values()
            )
        )

    @property
    def episode_spawn_progress(self) -> SpawnProgress:
        """Spawn progress for this episode.

        Cumulative: dead aircraft still count as ``spawned``. The
        denominator is fixed at reset by sampling the spawn config, so
        the ratio always trends toward N/N as the queue drains.
        """
        scheduled = self._spawn_generator.scheduled_count
        return SpawnProgress(
            spawned=scheduled - self._spawn_generator.pending,
            scheduled=scheduled,
        )

    def reset(
        self,
        seed: int | None = None,
        options: EnvOptions | None = None,
    ) -> tuple[AgentObservations, AgentInfos]:
        self._clear_agent_context_cache()
        self._rng = np.random.default_rng(seed)
        self._wind.reset()
        self.episode_spec = self.scenario.sample(self._rng)
        normalize_spawn_aircraft_types(self.config, self.episode_spec.spawn)
        # Per-aircraft sampled queryables accumulate per-callsign targets within an
        # episode; clear them so a re-used scenario object starts fresh each reset.
        for queryable in self.episode_queryables.values():
            reset_episode = getattr(queryable, "reset_episode", None)
            if callable(reset_episode):
                reset_episode()
        self._hooks.on_episode_reset(seed=seed, options=options)
        self._hooks.on_episode_loaded(self.episode_spec)

        self._live_info = {}

        reset_all_field_state(seed)
        self._aircraft_spawn_time.clear()
        self._aircraft_control_state.clear()
        self._aircraft_region.clear()
        self._spawn_generator.begin_episode(self._rng)
        self._invalidate_agent_cache()
        self._query_state_monitor.clear()

        # ``bs.sim.reset()`` clears traffic-attached state; CDMETHOD and ASAS
        # toggle don't always survive, so re-issue both to keep conflict
        # detection running every episode.
        self._runtime.reset(seed=seed)
        # ``bs.sim.reset()`` clears BlueSky's wind field; re-apply ours.
        self._runtime.apply_wind(self._wind)

        self._hooks.on_before_spawn()
        self._spawn_generator.schedule_episode(self._rng)
        self._spawn_generator.drain(self._rng)
        self._spawn_generator.maintain(self._rng)
        self._purge_missing_aircraft_state()
        self._hooks.on_after_spawn(rng=self._rng)
        self._traffic_monitor.clear()
        self._query_state_monitor.begin_step()

        # Force the sim into OP even when no aircraft spawned at reset
        # (e.g. every SpawnRegion has ``spawn_time > 0``). Otherwise
        # the BlueSky sim clock can stay pinned at 0 and prevent deferred
        # spawns from draining on later steps. Idempotent when already in OP.
        self._runtime.operate()
        self._driver.on_reset()

        controlled_agents = self._controlled_live_agents
        observations = self._assemble_observations(controlled_agents)
        infos = self._build_infos(controlled_agents)

        self._populate_task_info(observations, {}, infos)
        return observations, infos

    def step(
        self,
        actions: AgentActions,
    ) -> tuple[AgentObservations, AgentRewards, DoneFlags, DoneFlags, AgentInfos]:
        self._clear_agent_context_cache()
        self._delete_marked_aircraft()
        self._spawn_generator.drain(self._rng)
        self._traffic_monitor.begin_step()
        self._query_state_monitor.begin_step()
        self._hooks.on_before_step()
        live_index = self._live_agent_index()

        for acid, action in actions.items():
            idx = live_index.get(acid)
            if (
                idx is None
                or self._aircraft_control_state.get(acid)
                is not AircraftControlState.CONTROLLED
            ):
                continue
            # Record the normalized action so PrevActionNorm can expose a_{t-1};
            # done before obs assembly this step so the returned obs carries a_t.
            for obs_field in self._stateful_fields:
                obs_field.on_action_applied(acid, action)
            # Task hooks may consume non-generic action semantics before the
            # configured action fields dispatch aircraft-control commands.
            if not self._hooks.on_agent_action(idx, action):
                self._action_dispatcher.apply(idx, action)

        # A steady field is applied once at reset; only a gusting one needs
        # pushing into BlueSky again each step.
        self._wind.advance(float(self.config.dt), self._rng)
        if self._wind.is_dynamic:
            self._runtime.apply_wind(self._wind)

        for _ in range(self._n_substeps):
            self._driver.step()
            self._traffic_monitor.record_substep()
            self._query_state_monitor.record_substep()
            self._hooks.on_sim_step()

        self._purge_missing_aircraft_state()
        self._update_stateful_fields()
        controlled_agents = self._controlled_live_agents
        observations = self._assemble_observations(controlled_agents)
        infos = self._build_infos(controlled_agents)
        self._populate_task_info(observations, actions, infos)
        terminations, truncations = self._apply_done_conditions(
            controlled_agents,
            observations,
            actions,
            infos,
        )
        rewards = self._compute_rewards(
            observations,
            actions,
            terminations,
            truncations,
            infos,
        )
        for acid in controlled_agents:
            if terminations[acid] or truncations[acid]:
                infos[acid]["final_observation"] = observations[acid]
                infos[acid]["final_observation_agent_ids"] = tuple(controlled_agents)
        self._transition_done_agents(
            controlled_agents,
            terminations,
            truncations,
            infos,
        )
        self._purge_missing_aircraft_state()
        self._delete_marked_aircraft()
        # Steady-state top-up runs after this step's rewards/dones are settled,
        # so replacements first appear in ``next_observations`` and receive an
        # action next step (never a null-action reward this step).
        self._spawn_generator.maintain(self._rng)

        # Return post-cleanup observations for the next policy step. Terminal
        # observations are preserved in info["final_observation"] for replay.
        next_controlled_agents = self._controlled_live_agents
        if tuple(next_controlled_agents) == tuple(controlled_agents):
            next_observations = observations
        else:
            next_observations = self._assemble_observations(next_controlled_agents)

        return next_observations, rewards, terminations, truncations, infos

    def _transition_done_agents(
        self,
        agent_ids: Sequence[Callsign],
        terminations: DoneFlags,
        truncations: DoneFlags,
        infos: AgentInfos,
    ) -> None:
        """Apply done-agent events to the aircraft control-state map."""
        for acid in agent_ids:
            if terminations[acid] or truncations[acid]:
                self.set_aircraft_control_state(
                    acid,
                    self._hooks.on_agent_done(
                        acid,
                        infos[acid],
                        terminated=terminations[acid],
                        truncated=truncations[acid],
                    ),
                )

    def render(self) -> None:
        """Open (or keep alive) the BlueSky QtGL radar window.

        On the first call the ZMQ proxy thread is started, the sim node is
        connected to it, and the QtGL client subprocess is launched.
        Subsequent calls flush the sim node's ZMQ I/O so the GUI receives
        fresh traffic data.
        """
        if not self._driver._started:
            self._driver.start()
            self._driver.wait_until_ready()
            self._driver.on_render()
        else:
            self._driver.update()

    def close(self) -> None:
        """Terminate the QtGL subprocess and release resources."""
        self._driver.close()

    @property
    def episode_spawn(self):
        """Spawn resource active for the current episode."""
        return self.episode_spec.spawn

    @property
    def episode_queryables(self) -> dict[str, Queryable]:
        """Queryable resources active for the current episode."""
        return self.episode_spec.queryables

    @property
    def episode_airspace_bounds(self):
        """Airspace bounds active for the current episode."""
        return self.episode_spec.airspace_bounds

    @property
    def episode_max_aircraft(self) -> int:
        """Maximum aircraft selected from the current episode spawn candidates."""
        return int(self.episode_spec.max_aircraft)

    @property
    def rng(self) -> np.random.Generator:
        """Random generator for runtime task hooks."""
        return self._rng

    @property
    def wind(self) -> WindField:
        """The airspace's wind field."""
        return self._wind

    @property
    def sim_time(self) -> float:
        """Current BlueSky simulation time in seconds."""
        return float(self._runtime.sim_time)

    @property
    def live_info(self) -> AgentInfos:
        """Latest cached info keyed by live aircraft callsign."""
        return self._live_info

    def build_aircraft_infos(
        self,
        agent_ids: Sequence[Callsign] | None = None,
    ) -> AgentInfos:
        """Build aircraft info for selected callsigns, or all live aircraft."""
        return self._agent_info_builder.build(agent_ids)

    def cache_live_info(self, infos: AgentInfos) -> None:
        """Replace the latest cached live-aircraft info."""
        self._live_info = infos

    def aircraft_control_states(
        self,
    ) -> tuple[tuple[Callsign, AircraftControlState], ...]:
        """Return a stable snapshot of aircraft control states."""
        return tuple(self._aircraft_control_state.items())

    def aircraft_spawn_time(
        self,
        acid: Callsign,
        default: float | None = None,
    ) -> float | None:
        """Return when an aircraft entered the environment."""
        return self._aircraft_spawn_time.get(acid, default)

    def replace_aircraft_route(
        self,
        callsign: Callsign,
        _queryables,
        route,
        *,
        commit: bool = True,
    ) -> None:
        """Replace one aircraft route through the bound BlueSky runtime."""
        resolved_route = self._spawn_generator.resolve_route(
            callsign, route, self._rng
        )
        self._runtime.replace_aircraft_route(
            callsign,
            None if resolved_route is None else resolved_route.targets,
            commit=commit,
        )
        self._query_state_monitor.set_aircraft_route(
            callsign,
            None if resolved_route is None else resolved_route.names,
        )

    def observation_space(self, agent: str):
        return self._observation_assembler.observation_space(agent)

    @property
    def max_intruders(self) -> int:
        """Deterministic upper bound on per-agent intruder slot count.

        ``spawn.max_aircraft() - 1`` (every aircraft except ownship). Consumed by
        ``IntruderPaddingWrapper`` to size its padded output.
        """
        return max(0, self.episode_max_aircraft - 1)

    def action_space(self, agent: str):
        return self._observation_assembler.action_space(agent)

    # ------------------------------------------------------------------
    # PettingZoo helpers
    # ------------------------------------------------------------------
    # Helpers that drive the PettingZoo lifecycle:

    def _update_stateful_fields(self) -> None:
        """Drive the per-step hook for observation fields that keep state.

        A field sees only ``bs.traf`` in ``get``/``get_many``, which holds the
        *current* state - so anything needing history (the previous step's
        value) or env bookkeeping (spawn times, the last action) cannot be
        computed there. The environment is the only thing that knows where the
        step boundary falls, so it pumps ``on_step`` once per env step, after
        the substep loop, and each field updates its own store.

        Only fields the config actually lists are in ``_stateful_fields``, so a
        task using none of them pays nothing here.
        """
        dt = float(self.config.dt) or 1.0
        ids = bs.traf.id
        now = float(self._runtime.sim_time)
        if self._stateful_fields:
            spawn = self._aircraft_spawn_time
            ctx = StepContext(
                ids=tuple(ids),
                dt=dt,
                sim_time=now,
                age_s={a: max(0.0, now - spawn.get(a, now)) for a in ids},
            )
            for obs_field in self._stateful_fields:
                obs_field.on_step(ctx)

    def set_aircraft_control_state(
        self,
        acid: Callsign,
        state: AircraftControlState,
    ) -> None:
        """Set the desired lifecycle state for one live aircraft."""
        previous = self._aircraft_control_state.get(acid)
        self._aircraft_control_state[acid] = state
        if (previous is AircraftControlState.CONTROLLED) != (
            state is AircraftControlState.CONTROLLED
        ):
            self._invalidate_agent_cache()

    def mark_aircraft_for_deletion(self, acid: Callsign) -> None:
        """Request deletion for one aircraft on the next transition pass."""
        self.set_aircraft_control_state(acid, AircraftControlState.DELETE)

    def _purge_missing_aircraft_state(self) -> None:
        """Remove environment state for aircraft no longer present in BlueSky."""
        live = self._live_agent_id_set()
        # BlueSky can remove traffic outside this environment's lifecycle
        # path. Mirror that removal in the environment-owned state maps.
        removed_controlled = False
        for acid in list(self._aircraft_control_state):
            if acid in live:
                continue
            removed_controlled = self._purge_aircraft_state(acid) or removed_controlled

        if removed_controlled:
            self._invalidate_agent_cache()

    def _purge_aircraft_state(self, acid: Callsign) -> bool:
        """Remove env-owned state for one aircraft; return whether it was controlled."""
        was_controlled = (
            self._aircraft_control_state.get(acid) is AircraftControlState.CONTROLLED
        )
        self._aircraft_control_state.pop(acid, None)
        self._aircraft_spawn_time.pop(acid, None)
        self._aircraft_region.pop(acid, None)
        self._query_state_monitor.clear_aircraft_route(acid)
        for obs_field in self._stateful_fields:
            obs_field.on_aircraft_removed(acid)
        return was_controlled

    def _delete_marked_aircraft(self) -> None:
        aircraft_to_delete = [
            acid
            for acid, state in self._aircraft_control_state.items()
            if state is AircraftControlState.DELETE
        ]
        if not aircraft_to_delete:
            return
        removed_controlled = False
        for acid in aircraft_to_delete:
            self._runtime.delete_aircraft(acid)
            removed_controlled = self._purge_aircraft_state(acid) or removed_controlled
        if removed_controlled:
            self._invalidate_agent_cache()

    @contextmanager
    def _agent_context_cache_scope(self):
        """Cache per-agent contexts while observation fields are evaluated."""
        self._clear_agent_context_cache()
        self._agent_context_cache_enabled = True
        self._hooks.on_before_agent_contexts()
        try:
            yield
        finally:
            self._agent_context_cache_enabled = False
            self._clear_agent_context_cache()

    def _clear_agent_context_cache(self) -> None:
        """Drop cached per-agent context objects after lifecycle boundaries."""
        self._agent_context_cache.clear()
        self._agent_context_cache_enabled = False

    def agent_context(self, idx: int) -> AgentStepContext:
        """Return the agent-bound context for one aircraft index.

        Subclasses override :meth:`define_agent_context` to provide the
        optional task-specific ``data`` payload. The context is an invocation
        object for fields and task functions, not an environment state store.
        """
        if idx < 0 or idx >= len(self._runtime.agent_ids):
            raise IndexError(f"aircraft index out of range: {idx}")
        acid = self._runtime.agent_ids[idx]
        if not self._agent_context_cache_enabled:
            return self._build_agent_step_context(acid, idx)
        if acid not in self._agent_context_cache:
            self._agent_context_cache[acid] = self._build_agent_step_context(acid, idx)
        return self._agent_context_cache[acid]

    def _collect_stateful_fields(self):
        """Configured observation fields that override the state hooks.

        Every field list is walked, critic-only blocks included: a privileged
        field maintains state the same way an actor-visible one does, and a
        field that is recorded but never dropped on despawn is exactly the leak
        the hooks exist to prevent.
        """
        seen: set[int] = set()
        for name in (
            "obs_fields",
            "intruder_obs_fields",
            "critic_obs_fields",
            "critic_intruder_obs_fields",
        ):
            for obs_field in getattr(self.config, name, None) or ():
                if type(obs_field).is_stateful() and id(obs_field) not in seen:
                    seen.add(id(obs_field))
                    yield obs_field

    def _build_agent_step_context(self, acid: str, acidx: int) -> AgentStepContext:
        return AgentStepContext(
            acid=acid,
            acidx=acidx,
            data=self._hooks.define_agent_context(acid, acidx),
            queryables=self.episode_queryables,
            query_state=self._query_state_monitor,
            airspace=self._build_airspace_context(acidx),
            separation=self._traffic_monitor.build_separation_context(acid, acidx),
        )

    def _build_airspace_context(self, acidx: int) -> RegionResult:
        airspace = self.episode_airspace_bounds
        if airspace is None:
            return RegionResult.for_aircraft(
                None,
                acidx,
                current=RegionCurrent(inside=True),
            )
        region = QueryRegion(airspace)
        return RegionResult.for_aircraft(
            region,
            acidx,
            current=RegionCurrent(inside=region.contains_aircraft(acidx)),
        )

    def _assemble_observations(
        self,
        agent_ids: Sequence[Callsign],
    ) -> AgentObservations:
        with self._agent_context_cache_scope():
            return self._observation_assembler.get_obs(agent_ids)

    def _apply_done_conditions(
        self,
        agent_ids: Sequence[Callsign],
        observations: AgentObservations,
        actions: AgentActions,
        infos: AgentInfos,
    ) -> tuple[DoneFlags, DoneFlags]:
        terminations: DoneFlags = {}
        truncations: DoneFlags = {}
        for acid in agent_ids:
            obs = observations[acid]
            action = actions.get(acid)
            info = infos[acid]
            context = self.agent_context(info["acidx"])
            terminations[acid] = self._hooks.terminated(
                obs,
                action,
                context,
                info,
                self._rng,
            )
            truncations[acid] = self._hooks.truncated(
                obs,
                action,
                context,
                info,
                self._rng,
            )
        return terminations, truncations

    def _compute_rewards(
        self,
        observations: AgentObservations,
        actions: AgentActions,
        terminations: DoneFlags,
        truncations: DoneFlags,
        infos: AgentInfos,
    ) -> AgentRewards:
        return {
            a: self._hooks.reward(
                observations[a],
                actions.get(a),
                terminations[a],
                truncations[a],
                self.agent_context(infos[a]["acidx"]),
                infos[a],
                self._rng,
            )
            for a in observations
        }

    def _populate_task_info(
        self,
        observations: AgentObservations,
        actions: AgentActions,
        infos: AgentInfos,
    ) -> None:
        for provider in self.config.task_info_providers:
            for acid in observations:
                info = infos[acid]
                context = self.agent_context(info["acidx"])
                provider(
                    observations[acid],
                    actions.get(acid),
                    info,
                    context,
                    self._rng,
                )

    def _build_infos(self, controlled_agents: Sequence[Callsign]) -> AgentInfos:
        infos = self.build_aircraft_infos(controlled_agents)
        self.cache_live_info(infos)
        return infos

    # ------------------------------------------------------------------
    # Pure helpers
    # ------------------------------------------------------------------
    # Read-only utilities

    def _live_agent_id_set(self) -> set[Callsign]:
        return set(self._runtime.agent_ids)

    def _live_agent_index(self) -> dict[Callsign, int]:
        return {acid: idx for idx, acid in enumerate(self._runtime.agent_ids)}

    @cached_property
    def _controlled_live_agents(self) -> list[Callsign]:
        controlled = self._controlled_aircraft_ids()
        return [acid for acid in self._runtime.agent_ids if acid in controlled]

    def _controlled_aircraft_ids(self) -> set[Callsign]:
        return {
            acid
            for acid, state in self._aircraft_control_state.items()
            if state is AircraftControlState.CONTROLLED
        }

    def _invalidate_agent_cache(self) -> None:
        self.__dict__.pop("_controlled_live_agents", None)

    def _field_bounds(self, idx: int, fields) -> tuple[np.ndarray, np.ndarray]:
        return self._observation_assembler.field_bounds(idx, fields)

    def _ownship_bounds(self, idx: int, fields) -> tuple[np.ndarray, np.ndarray]:
        return self._observation_assembler.ownship_bounds(idx, fields)
