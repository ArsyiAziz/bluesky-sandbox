from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from functools import cache, cached_property
from typing import (
    TYPE_CHECKING,
    Any,
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
    resolve_spawn_aircraft_types,
)
from bluesky_sandbox.core.batch import (
    RawBatch,
    StepBatch,
    intruder_indices,
    stack_actions,
    stack_observations,
)
from bluesky_sandbox.core.layout import (
    Slot,
    action_applied,
    action_layout,
    flatten_action,
    observation_layout,
    observation_parts,
    state_parts,
)
from bluesky_sandbox.core.step_values import (
    RawObservation,
    StepValues,
    raw_action_values,
    unique_names_of,
)
from bluesky_sandbox.interface.fields._common import reset_field_state
from bluesky_sandbox.interface.fields._state import set_action_space_bounds
from bluesky_sandbox.interface.fields.actions.mask import ActionMask
from bluesky_sandbox.interface.fields.base import StepContext
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
from bluesky_sandbox.ui.drivers import FRAME_DRIVERS, RenderMode, get_driver_class

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
    SpawnRecord,
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
    "SpawnRecord",
    "SpawnQueueItem",
    "ViewSpec",
    "overridable",
]

if TYPE_CHECKING:
    # Only for the ``_hooks`` cast below. ``env`` imports this module, so a real
    # import would be circular - but a type checker resolves cycles fine, and at
    # runtime this block never executes.
    from bluesky_sandbox.env import BlueskyEnv
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
_VIEWS_BY_MODE = {"pygame", "panda3d", "rgb_array"}


F = TypeVar("F", bound=Callable[..., Any])
AgentActions: TypeAlias = Mapping[str, Any]
AgentInfos: TypeAlias = dict[str, BaseAgentInfo]
AgentObservations: TypeAlias = dict[str, BaseObs]
#: A reward or cost: a number, or a 1-D array of components.
Outcome: TypeAlias = float | np.ndarray
AgentRewards: TypeAlias = dict[str, Outcome]
DoneFlags: TypeAlias = dict[str, bool]
EnvOptions: TypeAlias = Mapping[str, Any]


def overridable(func: F) -> F:
    """Mark methods intended for environment subclasses to override.

    Tags the function so tooling (e.g. the designer catalog) can discover the
    available hooks by introspection rather than a hard-coded list.
    """
    func.__overridable__ = True  # type: ignore[attr-defined]
    return func


@cache
def _task_hook_names() -> frozenset[str]:
    """The hooks ``BlueskyEnv`` declares, as the runtime's required surface.

    Read from the authoring class rather than a protocol restating its
    signatures. ``env`` imports this module, so this import has to be deferred
    to call time - by which point ``env`` is importable, whether or not it has
    been loaded yet. Cached: the answer is fixed once both modules exist.
    """
    from bluesky_sandbox.env import BlueskyEnv  # noqa: PLC0415 - env imports us

    return frozenset(
        name
        for name in dir(BlueskyEnv)
        if getattr(getattr(BlueskyEnv, name, None), "__overridable__", False)
    )


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
          meters (orbit camera, click-to-select aircraft).
        * ``"rgb_array"`` - open no window: ``render()`` draws offscreen and
          returns the frame, ``(height, width, 3)`` RGB, drawn by
          ``frame_driver`` - pygame unless told otherwise. Nothing is drawn
          between ``render()`` calls.
        * ``None`` - no rendering (default).
    """

    #: Hooks the task defines batched (``reward`` for ``reward_batch``...);
    #: set per task class by :class:`~bluesky_sandbox.env.BlueskyEnv`.
    _batched_hooks: frozenset[str] = frozenset()
    #: The batchable hooks the task defines either way (``cost`` is optional).
    _defined_hooks: frozenset[str] = frozenset()

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
        frame_driver: str | None = None,
    ) -> None:
        """Construct the env.

        Parameters
        ----------
        frame_driver:
            With ``render_mode="rgb_array"``, which driver draws the frames:
            ``"pygame"`` (the default) or ``"panda3d"``. ``views`` then takes
            that driver's shape.
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

        if frame_driver is not None and render_mode != "rgb_array":
            raise ValueError(
                f'frame_driver picks who draws render_mode="rgb_array" frames; '
                f"render_mode {render_mode!r} draws with its own driver"
            )
        frame_driver = frame_driver or next(iter(FRAME_DRIVERS))
        if frame_driver not in FRAME_DRIVERS:
            raise ValueError(
                f"frame_driver must be one of {list(FRAME_DRIVERS)}, got {frame_driver!r}"
            )

        self.render_mode = render_mode
        self.frame_driver = frame_driver if render_mode == "rgb_array" else None
        self.realtime = realtime

        if not isinstance(config, EnvConfig):
            raise TypeError(f"config must be EnvConfig, got {type(config)!r}")
        self.config = copy.deepcopy(config)

        missing = sorted(n for n in _task_hook_names() if not hasattr(self, n))
        if missing:
            raise TypeError(
                f"{type(self).__name__} does not implement the task hooks "
                f"{missing}. Subclass BlueskyEnv, which supplies defaults for "
                "all of them, rather than BlueskyBaseEnvironment directly."
            )
        self._hooks = cast("BlueskyEnv", self)
        # The shape each of reward and cost first came in, which it keeps.
        self._outcome_shapes: dict[str, tuple[int, ...]] = {}

        self._aircraft_spawn_time: dict[str, float] = {}
        self._agent_context_cache: dict[str, AgentStepContext] = {}
        self._query_batch_cache: dict[tuple, Any] = {}
        # Raw field values this step, shared by the observation and the hooks.
        self._step_values = StepValues()
        # This step for every agent at once, built on first use by a batched hook.
        self._batch: StepBatch | None = None
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

        # Publish the (normalized) action bounds, fields in config order - the
        # order PrevActionNorm stores the action in - so it can size itself to
        # the real action space instead of assuming a range. Static per task
        # (from the action fields' normalizers), so resolving once here -
        # before any observation_space query - is enough.
        try:
            set_action_space_bounds(
                *self._observation_assembler.field_output_bounds(
                    None, self.config.action_fields
                )
            )
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

        driver_cls = get_driver_class(self.frame_driver or render_mode)
        driver_kwargs: dict[str, Any] = {"realtime": realtime}
        if self.frame_driver is not None:
            driver_kwargs["offscreen"] = True
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
    def spawn_log(self) -> tuple[SpawnRecord, ...]:
        """Every aircraft created this episode, in creation order, with its
        state as created: callsign, type, time, position, heading, speeds."""
        return tuple(self._spawn_generator.log)

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
        self._step_values.clear()
        self._batch = None
        self._rng = np.random.default_rng(seed)
        self._wind.reset()
        self.episode_spec = self.scenario.sample(self._rng)
        resolve_spawn_aircraft_types(self.config, self.episode_spec.spawn)
        self._hooks.on_episode_reset(seed=seed, options=options)
        self._hooks.on_episode_loaded(self.episode_spec)

        self._live_info = {}

        reset_field_state(seed)
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
        self._add_action_applied(infos, {})

        self._populate_task_info(observations, {}, infos)
        return observations, infos

    def step(
        self,
        actions: AgentActions,
    ) -> tuple[AgentObservations, AgentRewards, DoneFlags, DoneFlags, AgentInfos]:
        self._clear_agent_context_cache()
        self._step_values.begin_step()
        self._batch = None
        self._delete_marked_aircraft()
        self._spawn_generator.drain(self._rng)
        self._traffic_monitor.begin_step()
        self._query_state_monitor.begin_step()
        self._hooks.on_before_step()
        live_index = self._live_agent_index()
        applied: dict[Callsign, list[bool]] = {}

        for acid, action in actions.items():
            idx = live_index.get(acid)
            if (
                idx is None
                or self._aircraft_control_state.get(acid)
                is not AircraftControlState.CONTROLLED
            ):
                continue
            # One vector, fields in config order, whichever shape the action
            # space gives it: what the fields and the hooks are applied from.
            action = flatten_action(self.config, action)
            # Record the normalized action so PrevActionNorm can expose a_{t-1};
            # done before obs assembly this step so the returned obs carries a_t.
            for obs_field in self._stateful_fields:
                obs_field.on_action_applied(acid, action)
            # Task hooks may consume non-generic action semantics before the
            # configured action fields dispatch aircraft-control commands; an
            # action a hook consumed has no raw values.
            if not self._hooks.on_agent_action(idx, action):
                values = self._action_dispatcher.denormalize(idx, action)
                self._step_values.record_action(acid, raw_action_values(values))
                applied[acid] = self._action_dispatcher.apply_values(idx, values)

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
        self._add_action_applied(infos, applied)
        self._populate_task_info(observations, actions, infos)
        terminations, truncations = self._apply_done_conditions(
            controlled_agents,
            observations,
            actions,
            infos,
        )
        rewards = self._compute_outcomes(
            "reward", observations, actions, terminations, truncations, infos
        )
        if "cost" in self._defined_hooks:
            costs = self._compute_outcomes(
                "cost", observations, actions, terminations, truncations, infos
            )
            for acid, cost in costs.items():
                infos[acid]["cost"] = cost
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

    def render(self) -> np.ndarray | None:
        """Draw the current state: in its window, or - with ``render_mode``
        ``"rgb_array"`` - offscreen, returned as a ``(height, width, 3)`` RGB
        frame.

        The first call opens the window (for QtGL: starts the ZMQ proxy
        thread, connects the sim node and launches the client subprocess);
        later calls draw the latest state.
        """
        if self.frame_driver is not None:
            return self._driver.frame()
        if not self._driver._started:
            self._driver.start()
            self._driver.wait_until_ready()
            self._driver.on_render()
        else:
            self._driver.update()

    def close(self) -> None:
        """Terminate the QtGL subprocess and release resources."""
        try:
            self._driver.close()
        finally:
            self._runtime.close()

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
        resolved_route = self._spawn_generator.resolve_route(callsign, route, self._rng)
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

    def observation_layout(self, agent: str | None = None) -> dict[str, list[Slot]]:
        """Which columns of each observation part belong to which field.

        Fixed by the config, so the same for every agent; ``agent`` mirrors
        :meth:`observation_space`.
        """
        return observation_layout(self.config)

    def action_layout(self, agent: str | None = None) -> dict[str, list[Slot]]:
        """Which columns of each action part belong to which field."""
        return action_layout(self.config)

    def raw_observation(self, agent: str) -> RawObservation:
        """``agent``'s observation in raw values: ``raw["ownship"]["alt_ft"]``,
        ``raw["intruders"]["dist_to_own_nm"][i]`` for intruder row ``i``.

        Read from the values the observation was built from this step, so it
        costs no recomputation; see :class:`~.step_values.RawObservation`.
        """
        return self._raw_observation(self._runtime.agent_ids.index(agent))

    def raw_state(self, agent: str) -> RawObservation:
        """``agent``'s state fields in raw values, as :meth:`raw_observation`
        gives its observation: ``raw["ownship"]["alt_ft"]``. Seen by no agent."""
        return self._raw_state(self._runtime.agent_ids.index(agent))

    def raw_action(self, agent: str) -> Mapping[str, Any]:
        """The action ``agent`` was given this step, in raw values by field
        name: what each action field was set to. Empty if it had none."""
        return self._step_values.action(agent)

    def _raw_observation(self, acidx: int) -> RawObservation:
        return RawObservation(
            self._step_values, self._observation_parts(), acidx, self._agent_indices
        )

    def _observation_parts(self) -> dict[str, list[Any]]:
        return observation_parts(self.config)

    def _raw_state(self, acidx: int) -> RawObservation:
        return RawObservation(
            self._step_values,
            state_parts(self.config),
            acidx,
            self._agent_indices,
            what="state",
        )

    def _step_batch(
        self,
        agent_ids: Sequence[Callsign],
        observations: AgentObservations,
        infos: AgentInfos,
    ) -> StepBatch:
        """This step for ``agent_ids`` at once, built once per step."""
        if self._batch is not None and self._batch.acids == tuple(agent_ids):
            return self._batch
        index = self._live_agent_index()
        acidx = np.array([index[acid] for acid in agent_ids], dtype=np.intp)
        raw_action, has_action = stack_actions(
            unique_names_of(self.config.action_fields),
            [self._step_values.action(acid) for acid in agent_ids],
        )
        self._batch = StepBatch(
            acids=tuple(agent_ids),
            acidx=acidx,
            obs=stack_observations([observations[acid] for acid in agent_ids]),
            raw_obs=RawBatch(self._step_values, self._observation_parts(), acidx),
            state=RawBatch(
                self._step_values, state_parts(self.config), acidx, what="state"
            ),
            raw_action=raw_action,
            has_action=has_action,
            intruder_idx=intruder_indices(acidx),
            infos=[infos[acid] for acid in agent_ids],
            rng=self._rng,
            _query=self.query_batch,
            _context=self.agent_context,
        )
        return self._batch

    def _agent_indices(self) -> np.ndarray:
        """The controlled agents' traffic indices: the observation's ownships."""
        index = self._live_agent_index()
        return np.array(
            [index[acid] for acid in self._controlled_live_agents], dtype=np.intp
        )

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
        self._query_batch_cache.clear()
        self._agent_context_cache_enabled = False

    def query_batch(self, name: str, indices) -> Any:
        """Queryable ``name`` for many aircraft at once, as arrays.

        The batched counterpart of ``agent_context(idx).query(name)`` for
        observation fields, cached for the same scope as the contexts - so
        every field reading one queryable shares one computation.
        """
        indices = tuple(int(i) for i in indices)
        key = (name, indices)
        cached = self._query_batch_cache.get(key)
        if cached is not None:
            return cached
        try:
            queryable = self.episode_queryables[name]
        except KeyError as exc:
            raise KeyError(f"queryable {name!r} is not configured") from exc
        batch = self._query_state_monitor.batch(name, queryable, indices)
        if self._agent_context_cache_enabled:
            self._query_batch_cache[key] = batch
        return batch

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
        """Configured fields that override the state hooks.

        Every field list is walked, critic-only blocks and actions included: a
        privileged field maintains state the same way an actor-visible one does,
        an action (an ``ActionMask``) may remember what it was set to, and a
        field that is recorded but never dropped on despawn is exactly the leak
        the hooks exist to prevent.
        """
        seen: set[int] = set()
        for name in (
            "action_fields",
            "obs_fields",
            "intruder_obs_fields",
            "critic_obs_fields",
            "critic_intruder_obs_fields",
            "state_fields",
            "intruder_state_fields",
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
            raw_obs=self._raw_observation(acidx),
            state=self._raw_state(acidx),
            raw_action=self._step_values.action(acid),
            _step_values=self._step_values,
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
        batched = self._batched_hooks & {"terminated", "truncated"}
        contexts: dict[Callsign, AgentStepContext] = {}

        def decide(hook: str) -> DoneFlags:
            if hook in batched:
                batch = self._step_batch(agent_ids, observations, infos)
                flags = _one_per_agent(
                    getattr(self._hooks, f"{hook}_batch")(batch), batch, f"{hook}_batch"
                ).astype(bool)
                setattr(batch, hook, flags)
                return dict(zip(agent_ids, map(bool, flags)))
            flags = {}
            for acid in agent_ids:
                info = infos[acid]
                if acid not in contexts:
                    contexts[acid] = self.agent_context(info["acidx"])
                flags[acid] = getattr(self._hooks, hook)(
                    observations[acid],
                    actions.get(acid),
                    contexts[acid],
                    info,
                    self._rng,
                )
            return flags

        if batched:
            return decide("terminated"), decide("truncated")
        # Both per agent: one agent's pair, then the next's - the order hooks
        # drawing from rng have always seen.
        terminations: DoneFlags = {}
        truncations: DoneFlags = {}
        for acid in agent_ids:
            info = infos[acid]
            context = self.agent_context(info["acidx"])
            args = (observations[acid], actions.get(acid), context, info, self._rng)
            terminations[acid] = self._hooks.terminated(*args)
            truncations[acid] = self._hooks.truncated(*args)
        return terminations, truncations

    def _compute_outcomes(
        self,
        hook: str,
        observations: AgentObservations,
        actions: AgentActions,
        terminations: DoneFlags,
        truncations: DoneFlags,
        infos: AgentInfos,
    ) -> AgentRewards:
        """Each agent's ``reward`` or ``cost``: a number, or a 1-D array whose
        shape is the same for every agent and every step."""
        if hook in self._batched_hooks:
            agent_ids = list(observations)
            batch = self._step_batch(agent_ids, observations, infos)
            batch.terminated = np.array([terminations[a] for a in agent_ids], bool)
            batch.truncated = np.array([truncations[a] for a in agent_ids], bool)
            name = f"{hook}_batch"
            values = _per_agent_outcomes(getattr(self._hooks, name)(batch), batch, name)
            return {acid: self._outcome(name, v) for acid, v in zip(agent_ids, values)}
        per_agent = getattr(self._hooks, hook)
        return {
            a: self._outcome(
                hook,
                per_agent(
                    observations[a],
                    actions.get(a),
                    terminations[a],
                    truncations[a],
                    self.agent_context(infos[a]["acidx"]),
                    infos[a],
                    self._rng,
                ),
            )
            for a in observations
        }

    def _outcome(self, hook: str, value: Any) -> Outcome:
        """``value`` as a number or 1-D array, checked against the shape
        ``hook`` returned before."""
        array = np.asarray(value, dtype=float)
        if array.ndim > 1:
            raise ValueError(
                f"{hook} returned shape {array.shape} for one agent; return a "
                "number, or a 1-D array of components."
            )
        seen = self._outcome_shapes.setdefault(hook.removesuffix("_batch"), array.shape)
        if array.shape != seen:
            raise ValueError(
                f"{hook} returned shape {array.shape}, but it returned {seen} "
                "before; a reward or cost keeps one shape."
            )
        return float(array) if array.ndim == 0 else array

    def _populate_task_info(
        self,
        observations: AgentObservations,
        actions: AgentActions,
        infos: AgentInfos,
    ) -> None:
        for provider in self.config.task_info_providers:
            if getattr(provider, "batched", False):
                provider(self._step_batch(list(observations), observations, infos))
                continue
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

    def _add_action_applied(
        self, infos: AgentInfos, applied: Mapping[Callsign, list[bool]]
    ) -> None:
        """With an ``ActionMask`` configured, ``info["action_applied"]``: which
        values of each agent's action took effect, shaped as the action - all
        of them for an agent given no action, or one a hook consumed."""
        if not any(isinstance(f, ActionMask) for f in self.config.action_fields):
            return
        for acid, info in infos.items():
            info["action_applied"] = action_applied(self.config, applied.get(acid))

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


def _per_agent_outcomes(values: Any, batch: StepBatch, hook: str) -> np.ndarray:
    """A batched reward or cost: one per agent, each a number or a 1-D array."""
    array = np.asarray(values, dtype=float)
    if array.ndim not in (1, 2) or array.shape[0] != len(batch):
        raise ValueError(
            f"{hook} returned shape {array.shape}; it must return one value per "
            f"agent in the order of batch.acids - shape ({len(batch)},), or "
            f"({len(batch)}, k) for k components."
        )
    return array


def _one_per_agent(values: Any, batch: StepBatch, hook: str) -> np.ndarray:
    """A batched hook's result, checked to hold one value per agent."""
    array = np.asarray(values)
    if array.shape != (len(batch),):
        raise ValueError(
            f"{hook} returned shape {array.shape}; it must return one value per "
            f"agent, shape ({len(batch)},), in the order of batch.acids."
        )
    return array
