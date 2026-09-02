from __future__ import annotations

import copy
import math
import string
import warnings
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
    get_args,
)

import bluesky as bs
import numpy as np
from bluesky.tools.aero import ft, kts, nm, vcas2tas
from bluesky.tools.geo import kwikqdrdist
from pettingzoo import ParallelEnv

from bluesky_sandbox.config import (
    EnvConfig,
    normalize_spawn_aircraft_types,
)
from bluesky_sandbox.interface.fields.base import SubstepContext
from bluesky_sandbox.interface.fields.observations import (
    reset_all_field_state,
    set_action_space_bounds,
)
from bluesky_sandbox.interface.task import (
    AgentStepContext,
    BaseAgentInfo,
    BaseObs,
)
from bluesky_sandbox.sim.geometry.conflict import cd_hpz_m, cd_rpz_m
from bluesky_sandbox.sim.performance.envelope import (
    feasible_alt_cas,
    feasible_cas_at_alt,
)
from bluesky_sandbox.sim.queryables import (
    Queryable,
    QueryRegion,
    RegionCurrent,
    RegionResult,
    Waypoint,
    WaypointTarget,
)
from bluesky_sandbox.sim.scenario import EpisodeSpec, Scenario
from bluesky_sandbox.sim.spawn import route_step_name
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


_ALPHABET = list(string.ascii_uppercase)

Callsign: TypeAlias = str
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


# Steady-state (``SpawnRegion.maintain``) respawn guard: a topped-up aircraft
# must clear all live traffic (avoiding an instant loss of separation), retried
# up to this many times before deferring to a later step. The distance itself is
# The zone itself is the live CD protected zone plus the region's
# ``spawn_sep_*``, so it tracks whatever the scenario actually detects
# conflicts against.
_MAINTAIN_SPAWN_MAX_TRIES = 20
# Consecutive fully-failed spawns for one region before warning once. A single
# failure is normal; a sustained failure means
# the guard is unsatisfiable and the region is silently under-populated.
_MAINTAIN_SPAWN_WARN_AFTER = 5


@dataclass(frozen=True)
class SpawnPosition:
    lat_deg: float
    lon_deg: float
    alt_ft: float
    spd_kts: float
    hdg_deg: float | None = None  # None -> uniform 0-360 at spawn
    # When True, ``spd_kts`` is a provisional placeholder and the real CAS is
    # drawn from the aircraft's flight envelope at ``alt_ft`` post-creation.
    spd_from_envelope: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> SpawnPosition:
        return cls(
            lat_deg=values["lat_deg"],
            lon_deg=values["lon_deg"],
            alt_ft=values["alt_ft"],
            spd_kts=values["spd_kts"],
            hdg_deg=values.get("hdg_deg"),
            spd_from_envelope=bool(values.get("_spd_from_envelope", False)),
        )


@dataclass(frozen=True)
class SpawnQueueItem:
    spawn_time: float
    actype: str
    position: SpawnPosition
    callsign_prefix: str | None
    route: list[str | dict] | None
    region_index: int = -1


class ResolvedRoute(NamedTuple):
    names: list[str]
    targets: list[WaypointTarget]


class AircraftControlState(Enum):
    CONTROLLED = "controlled"
    BACKGROUND = "background"
    DELETE = "delete"


class SpawnProgress(NamedTuple):
    spawned: int
    scheduled: int


AircraftControlStates: TypeAlias = dict[Callsign, AircraftControlState]


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
        self._spawn_queue: list[SpawnQueueItem] = []
        # Consecutive fully-failed resets per region index. A
        # persistently unsatisfiable spawn guard warns once instead of silently
        # holding the region below its target. Cleared on a successful spawn.
        self._maintain_spawn_failures: dict[int, int] = {}
        # Total spawns sampled at the most recent reset (live + queued +
        # already-dead). Used by HUD to render ``spawned X/N``.
        self._episode_scheduled_aircraft_count: int = 0
        # Origin spawn-region index per live aircraft, and per-region target
        # live counts for regions in steady-state ``maintain`` mode. Together
        # they drive the top-up that holds each maintain region at its target.
        self._aircraft_region: dict[str, int] = {}
        self._maintain_target: dict[int, int] = {}
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
        self._runtime = BlueSkyRuntime()
        self._action_dispatcher.bind_env(self)
        self._observation_assembler.bind_env(self)
        self._query_state_monitor.bind_env(self)
        self._renderable_builder.bind_env(self)
        self._traffic_monitor.bind_env(self)
        self._agent_info_builder.bind_env(self)
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
        # Time-correlated wind gust state (north, east) in m/s, evolved by an
        # Ornstein-Uhlenbeck process when ``config.turbulence_kts > 0``.
        self._wind_gust_ne: tuple[float, float] = (0.0, 0.0)

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
        if self._maintain_target:
            return False
        return not self._aircraft_control_state and not self._spawn_queue

    @property
    def has_future_agents(self) -> bool:
        """True when later steps can expose controllable agents."""
        return (
            bool(self._maintain_target)
            or bool(self._spawn_queue)
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
        return SpawnProgress(
            spawned=self._episode_scheduled_aircraft_count - len(self._spawn_queue),
            scheduled=self._episode_scheduled_aircraft_count,
        )

    def reset(
        self,
        seed: int | None = None,
        options: EnvOptions | None = None,
    ) -> tuple[AgentObservations, AgentInfos]:
        self._clear_agent_context_cache()
        self._rng = np.random.default_rng(seed)
        self._wind_gust_ne = (0.0, 0.0)
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
        self._spawn_queue.clear()
        self._aircraft_control_state.clear()
        self._aircraft_region.clear()
        self._sample_maintain_targets(self._rng)
        self._invalidate_agent_cache()
        self._query_state_monitor.clear()

        # ``bs.sim.reset()`` clears traffic-attached state; CDMETHOD and ASAS
        # toggle don't always survive, so re-issue both to keep conflict
        # detection running every episode.
        self._runtime.reset(seed=seed)

        self._hooks.on_before_spawn()
        self._enqueue_spawns(self._rng)
        # Snapshot the total scheduled before any drain so the HUD counter
        # ``spawned X/N`` has a stable denominator across the episode.
        self._episode_scheduled_aircraft_count = len(self._spawn_queue)
        self._drain_spawn_queue(self._rng)
        self._maintain_spawns(self._rng)
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
        self._drain_spawn_queue(self._rng)
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

        self._advance_wind_gust()

        for _ in range(self._n_substeps):
            self._driver.step()
            self._traffic_monitor.record_substep()
            self._query_state_monitor.record_substep()
            self._hooks.on_sim_step()

        self._purge_missing_aircraft_state()
        self._publish_turn_rates()
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
        self._maintain_spawns(self._rng)

        # Return post-cleanup observations for the next policy step. Terminal
        # observations are preserved in info["final_observation"] for replay.
        next_controlled_agents = self._controlled_live_agents
        if tuple(next_controlled_agents) == tuple(controlled_agents):
            next_observations = observations
        else:
            next_observations = self._assemble_observations(next_controlled_agents)

        return next_observations, rewards, terminations, truncations, infos

    def _advance_wind_gust(self) -> None:
        """Evolve the wind gust one step and re-apply the wind field.

        Turbulence is an Ornstein-Uhlenbeck process on the (north, east) gust
        velocity: mean-reverting to zero with a correlation time
        ``config.gust_tau_s`` and a stationary RMS of ``config.turbulence_kts``.
        No-op (steady mean wind, applied once at reset) when turbulence is off.
        """
        turb = float(getattr(self.config, "turbulence_kts", 0.0))
        if turb <= 0.0:
            return
        dt = float(self.config.dt)
        tau = max(float(getattr(self.config, "gust_tau_s", 30.0)), 1e-3)
        decay = math.exp(-dt / tau)
        step_sigma = turb * kts * math.sqrt(max(1.0 - decay * decay, 0.0))
        gvn, gve = self._wind_gust_ne
        self._wind_gust_ne = (
            decay * gvn + step_sigma * float(self._rng.standard_normal()),
            decay * gve + step_sigma * float(self._rng.standard_normal()),
        )
        self._runtime.apply_wind(self._wind_gust_ne)

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
        resolved_route = self._resolve_route_for_aircraft(callsign, route, self._rng)
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

    def _enqueue_spawns(self, rng: np.random.Generator) -> None:
        """Sample one episode's worth of spawns into ``self._spawn_queue``.

        Sampled items are sorted ascending by ``spawn_time`` so the drain
        path can stop as soon as it hits an entry that isn't due yet.
        """
        self._maintain_spawn_failures.clear()
        self._spawn_queue = sorted(
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
                    self.episode_spawn.iter_spawns(
                        rng,
                        limit=self.episode_max_aircraft,
                    )
                )
            ),
            key=lambda item: item.spawn_time,
        )

    def _sample_maintain_targets(self, rng: np.random.Generator) -> None:
        """Fix each ``maintain`` region's target live count for this episode."""
        self._maintain_target = {
            region_index: region.sample_n(rng)
            for region_index, region in enumerate(self.episode_spawn.regions)
            if region.maintain
        }

    def _maintain_spawns(self, rng: np.random.Generator) -> None:
        """Top up each steady-state region to its target live count.

        Called after spawns/deletions each step (and at reset): counts live
        aircraft per maintain region and materialises separation-guarded
        replacements until the target is met. A region that can't place a
        clear spawn this step simply retries next step.
        """
        if not self._maintain_target:
            return
        live = self._live_agent_id_set()
        counts: dict[int, int] = {}
        for acid, region_index in self._aircraft_region.items():
            if acid in live:
                counts[region_index] = counts.get(region_index, 0) + 1
        used = set(self._runtime.agent_ids)
        changed = False
        for region_index, target in self._maintain_target.items():
            for _ in range(max(0, target - counts.get(region_index, 0))):
                item = self._sample_clear_spawn(region_index, rng)
                if item is None:
                    break  # too crowded to place clear of traffic; retry later
                changed = self._materialize_spawn(item, used, rng) or changed
        if changed:
            self._invalidate_agent_cache()

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
            return not self._runtime.predicted_conflict(
                pos.lat_deg,
                pos.lon_deg,
                pos.alt_ft,
                hdg_deg,
                pos.spd_kts,
                sep_nm=sep_nm,
                sep_ft=sep_ft,
                lookahead_s=lookahead_s,
            )
        return not self._runtime.inside_separation_zone(
            pos.lat_deg,
            pos.lon_deg,
            pos.alt_ft,
            min_sep_nm=cd_rpz_m() / nm if sep_nm is None else float(sep_nm),
            min_sep_ft=cd_hpz_m() / ft if sep_ft is None else float(sep_ft),
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
        used = set(self._runtime.agent_ids)
        callsign = self._generate_callsign(used, rng)
        self._runtime.create_aircraft(
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
                self._runtime.index(callsign), pos.alt_ft, rng
            )
        finally:
            self._runtime.delete_aircraft(callsign)
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
        sep_nm, sep_ft, look_s = self.episode_spawn.region_spawn_separation(
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
                self.episode_spawn.sample_region_spawn(item.region_index, rng)
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
        conflict_free = self.episode_spawn.region_conflict_free(region_index)
        sep_nm, sep_ft, look_s = self.episode_spawn.region_spawn_separation(
            region_index
        )
        for _ in range(_MAINTAIN_SPAWN_MAX_TRIES):
            _spawn_time, actype, position, prefix, route = (
                self.episode_spawn.sample_region_spawn(region_index, rng)
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
                self._maintain_spawn_failures.pop(region_index, None)
                return SpawnQueueItem(
                    spawn_time=self._runtime.sim_time,
                    actype=actype,
                    position=replace(pos, hdg_deg=hdg),
                    callsign_prefix=prefix,
                    route=route,
                    region_index=region_index,
                )
        n = self._maintain_spawn_failures.get(region_index, 0) + 1
        self._maintain_spawn_failures[region_index] = n
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

    def _publish_turn_rates(self) -> None:
        """Store each live aircraft's realized kinematics over this step for the
        obs fields (and physical-smoothness rewards) that an ObsField can't
        compute alone because they need the step's *starting* velocity.

        Two things are published, both keyed off the change in the ground-velocity
        vector since the previous step (0 on an aircraft's first step):

        * turn rate (deg/s) = wrapped change in track over ``dt``, read by
          ``TurnRateDegPerSec``. A rate, so orientation-invariant.
        * realized acceleration read by ``RealizedAccel{AlongTrack,CrossTrack,
          Vertical}Ms2``: tangential = change in ground speed over ``dt``; normal
          (centripetal) = mean ground speed times the track-change rate (rad/s),
          signed by turn direction; vertical = change in vertical speed over
          ``dt``. Measured across the whole multi-substep step, so it does not
          alias to ~0 when a maneuver is *completing* at the step boundary (the
          level-off / roll-out case that a single end-of-step snapshot misses).

        Also publishes each aircraft's seconds-since-spawn for ``TimeInEnvS``.
        It needs no velocity history, but it shares this method's contract -
        per-aircraft per-step state that only the environment can supply - and
        this pass is already walking every live aircraft.
        """
        dt = float(self.config.dt) or 1.0
        ids = bs.traf.id
        now = float(self._runtime.sim_time)
        # Stateful fields maintain their own per-aircraft state. Only fields the
        # config actually uses are in ``_stateful_fields``, so a task reading no
        # rates or ages pays nothing here - and this runs once per substep, of
        # which there are dt/simdt (commonly 100) per env step.
        if self._stateful_fields:
            spawn = self._aircraft_spawn_time
            ctx = SubstepContext(
                ids=tuple(ids),
                dt=dt,
                sim_time=now,
                age_s={a: max(0.0, now - spawn.get(a, now)) for a in ids},
            )
            for obs_field in self._stateful_fields:
                obs_field.on_substep(ctx)

    def _reachable_alt_window(
        self,
        acidx: int,
        wp_lat: float,
        wp_lon: float,
        vs_fraction: float = 1.0,
        from_lat: float | None = None,
        from_lon: float | None = None,
        from_alt_ft: float | None = None,
    ) -> tuple[float | None, float | None]:
        """Altitude band (ft) the aircraft can reach by the time it flies to the
        waypoint, from the performance model's max vertical rate.

        The aircraft covers the *leg-start*->waypoint distance ``d`` (nm) before
        the fix. ``from_lat``/``from_lon``/``from_alt_ft`` give that leg start;
        they default to the aircraft's live (spawn) state for the first leg, but
        a chained leg must pass the **previous waypoint's** position/altitude, so
        the window is computed from where the leg actually begins (wp1) rather
        than the spawn. Ground speed is taken at the aircraft's *max true
        airspeed* (the shortest possible transit ``t = 60*d/GS`` minutes), so the
        bound stays valid no matter how fast the policy chooses to fly. At up to
        ``vsmax`` (ft/min) it can then change altitude by ``reach = f * vsmax *
        t``, where ``f`` (``vs_fraction``, <=1) reserves margin for the turn-to-fix
        and simultaneous speed matching. Returns ``(alt0 - reach, alt0 + reach)``
        for the caller to intersect with the envelope, or ``(None, None)`` when no
        meaningful speed / rate is available (leaving the window unbounded).
        """
        vsmax = getattr(bs.traf.perf, "vsmax", None)
        if vsmax is None:
            return None, None
        vs_max_fpm = abs(float(vsmax[acidx])) / ft * 60.0  # m/s -> ft/min
        # Ground speed = the aircraft's *maximum true airspeed* (shortest possible
        # transit, so the bound holds however fast the policy flies). ``perf.vmax``
        # is a *CAS* limit whose TAS grows with altitude, so evaluate it at the
        # ceiling for the true max GS (no wind assumed). Using perf.vmax as knots
        # directly understates GS and would over-promise the reachable band.
        vmax_cas_ms = float(bs.traf.perf.vmax[acidx])  # m/s CAS limit
        hmax_m = float(bs.traf.perf.hmax[acidx])  # ceiling, m
        gs_kt = float(vcas2tas(vmax_cas_ms, hmax_m)) / kts  # max TAS -> max GS
        f = float(vs_fraction)
        if not math.isfinite(gs_kt) or gs_kt <= 1.0 or vs_max_fpm <= 0.0 or f <= 0.0:
            return None, None

        lat0 = float(bs.traf.lat[acidx]) if from_lat is None else float(from_lat)
        lon0 = float(bs.traf.lon[acidx]) if from_lon is None else float(from_lon)
        _qdr, d_nm = kwikqdrdist(lat0, lon0, float(wp_lat), float(wp_lon))
        alt0_ft = (
            float(bs.traf.alt[acidx]) / ft
            if from_alt_ft is None
            else float(from_alt_ft)
        )

        reach_ft = f * vs_max_fpm * (60.0 * float(d_nm) / gs_kt)  # f * vsmax * t_min
        return alt0_ft - reach_ft, alt0_ft + reach_ft

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
            queryable = self.episode_queryables[step]
            if not isinstance(queryable, Waypoint):
                raise TypeError(f"route step {step!r} does not reference a Waypoint")
            return queryable.target
        if not isinstance(step, dict) or "waypoint" not in step:
            raise TypeError(f"route step must be a waypoint name or dict, got {step!r}")

        queryable = self.episode_queryables[step["waypoint"]]
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
            acidx = self._runtime.index(callsign)
            alt_min_ft, alt_max_ft = region_alt_lo, region_alt_hi
            if reachable_from_spawn:
                vs_fraction = float(step.get("reachable_vs_fraction", 1.0))
                from_lat, from_lon, from_alt_ft = from_state or (None, None, None)
                reach_lo, reach_hi = self._reachable_alt_window(
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

    def _drain_spawn_queue(self, rng: np.random.Generator) -> None:
        """Materialise every queued aircraft whose ``spawn_time`` has arrived.

        A conflict-free item with no clear candidate state is *deferred* (its
        ``spawn_time`` pushed one env step) rather than spawned in conflict -
        the conflict-free guarantee is absolute at spawn time; density catches
        up as soon as the airspace admits a clear state.
        """
        if not self._spawn_queue:
            return
        now = self._runtime.sim_time
        used = set(self._runtime.agent_ids)
        # Queue is sorted by spawn_time; pop from the front while due.
        i = 0
        agent_cache_changed = False
        deferred: list[SpawnQueueItem] = []
        for item in self._spawn_queue:
            if item.spawn_time > now:
                break
            i += 1
            if self.episode_spawn.region_conflict_free(item.region_index):
                cleared = self._conflict_free_item(item, rng)
                if cleared is None:
                    retry_at = now + (float(self.config.dt) or 1.0)
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
            del self._spawn_queue[:i]
        if deferred:
            self._spawn_queue.extend(deferred)
            self._spawn_queue.sort(key=lambda queued: queued.spawn_time)
        if agent_cache_changed:
            self._invalidate_agent_cache()

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
        self._runtime.create_aircraft(
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
            acidx = self._runtime.index(callsign)
            cas_kts = feasible_cas_at_alt(acidx, item.position.alt_ft, rng)
            self._runtime.delete_aircraft(callsign)
            self._runtime.create_aircraft(
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
            self._runtime.append_aircraft_route(
                callsign,
                route.targets,
            )
            self._query_state_monitor.set_aircraft_route(
                callsign,
                route.names,
            )
        # Hooks see a names-only view (tasks may treat the route as a list of
        # waypoint names); the full step list with any per-step crossing
        # restrictions only reaches ADDWPT above.
        route_names = route.names if route else None
        self._hooks.on_aircraft_spawned(callsign, route_names)
        state = self._hooks.define_initial_aircraft_control_state(
            callsign,
            route_names,
        )
        # A spawn region marked ``controlled=False`` forces its aircraft to
        # background (uncooperative) traffic regardless of the task hook: they
        # fly their route but the policy never commands them.
        regions = self.episode_spawn.regions
        if (
            0 <= item.region_index < len(regions)
            and not regions[item.region_index].controlled
        ):
            state = AircraftControlState.BACKGROUND
        self.set_aircraft_control_state(callsign, state)
        # Use the actual materialisation time (``bs.sim.simt``) rather than the
        # scheduled ``spawn_time`` so ``time_in_env`` measures real simulator
        # presence - not the gap between schedule and the draining step.
        self._aircraft_spawn_time[callsign] = self._runtime.sim_time
        self._aircraft_region[callsign] = item.region_index
        return state is AircraftControlState.CONTROLLED

    # Naming convention in this section:
    # - "agent" means a controlled PettingZoo participant.
    # - "aircraft" means a live BlueSky traffic object in any control state.
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

    def _field_bounds(self, idx: int, fields) -> tuple[np.ndarray, np.ndarray]:
        return self._observation_assembler.field_bounds(idx, fields)

    def _ownship_bounds(self, idx: int, fields) -> tuple[np.ndarray, np.ndarray]:
        return self._observation_assembler.ownship_bounds(idx, fields)
