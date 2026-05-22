from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.core.base_environment import (
    AircraftControlState,
    BlueskyBaseEnvironment,
    RenderMode,
    ViewSpec,
    overridable,
)
from bluesky_sandbox.interface.fields.base import (
    ActionField,
    EnvObsField,
    ObsField,
    PairObsField,
)
from bluesky_sandbox.interface.task import (
    AircraftReadoutItem,
    BaseAgentInfo,
    TaskInfoProvider,
    WaypointReadoutItem,
)
from bluesky_sandbox.sim.scenario import EpisodeSpec, Scenario


class BlueskyEnv(BlueskyBaseEnvironment):
    """Pythonic task-authoring base class for BlueSky environments.

    Subclasses define Python methods for task behavior and pass an explicit
    static ``EnvConfig`` plus a ``Scenario`` to this base class.
    """

    def __init__(
        self,
        *,
        scenario: Scenario,
        config: EnvConfig,
        render_mode: RenderMode = None,
        realtime: bool = False,
        views: ViewSpec | None = None,
    ) -> None:
        super().__init__(
            config=config,
            scenario=scenario,
            render_mode=render_mode,
            realtime=realtime,
            views=views,
        )
        task_info_providers = list(self.define_task_info_providers())
        if any(not callable(provider) for provider in task_info_providers):
            raise ValueError("define_task_info_providers() must return callables.")
        self.config.task_info_providers.extend(task_info_providers)

    @overridable
    def define_obs_fields(self) -> Sequence[ObsField]:
        raise NotImplementedError

    @overridable
    def define_intruder_obs_fields(self) -> Sequence[ObsField | PairObsField] | None:
        return None

    @overridable
    def define_action_fields(self) -> Sequence[ActionField]:
        return ()

    @overridable
    def define_task_info_providers(self) -> Sequence[TaskInfoProvider]:
        return ()

    @overridable
    def on_episode_loaded(self, _episode_spec: EpisodeSpec) -> None:
        """Hook for tasks to read sampled episode-specific data."""
        return

    @overridable
    def on_episode_reset(
        self,
        *,
        seed: int | None,
        options: Mapping[str, Any] | None,
    ) -> None:
        """Hook for task-local reset bookkeeping before spawning."""
        return

    @overridable
    def on_before_spawn(self) -> None:
        """Hook after reset state is cleared and before spawns are sampled."""
        return

    @overridable
    def on_after_spawn(self, *, rng: np.random.Generator) -> None:
        """Hook after reset spawns have been drained and state transitioned."""
        del rng

    @overridable
    def on_before_step(self) -> None:
        """Hook before actions are applied for one environment step."""
        return

    @overridable
    def on_sim_step(self) -> None:
        """Hook for tasks that need per-physics-step lifecycle sampling."""
        return

    @overridable
    def on_agent_action(self, _idx: int, _action: Any) -> bool:
        """Handle a live controlled-agent action before field dispatch.

        Return ``True`` when the task consumed the action and the configured
        action fields should not be dispatched. Return ``False`` to let the
        base environment apply the configured action fields.
        """
        return False

    @overridable
    def define_aircraft_readouts(self, _acid: str) -> tuple[AircraftReadoutItem, ...]:
        """Return task-specific aircraft readout rows."""
        return ()

    @overridable
    def define_waypoint_readouts(self, _acid: str) -> tuple[WaypointReadoutItem, ...]:
        """Return atomic route-readout annotations for one aircraft."""
        return ()

    @overridable
    def on_aircraft_spawned(
        self,
        _callsign: str,
        _route: list[str] | None,
    ) -> None:
        """Hook for tasks that need to tune aircraft after route creation."""
        return

    @overridable
    def define_initial_aircraft_control_state(
        self,
        _callsign: str,
        _route: list[str] | None,
    ) -> AircraftControlState:
        """Define whether a spawned aircraft starts controlled or background."""
        return AircraftControlState.CONTROLLED

    @overridable
    def on_agent_done(
        self,
        _acid: str,
        _info: BaseAgentInfo,
        *,
        terminated: bool,
        truncated: bool,
    ) -> AircraftControlState:
        """Return the next aircraft control state for a done PettingZoo agent."""
        del terminated, truncated
        return AircraftControlState.DELETE

    @overridable
    def on_before_agent_contexts(self) -> None:
        """Prepare task-local data used by ``define_agent_context``."""
        return

    @overridable
    def define_agent_context(self, _acid: str, _acidx: int) -> object:
        """Build task-specific data for one agent context."""
        return None

    @overridable
    def reward(
        self,
        _obs,
        _action,
        _terminated,
        _truncated,
        _context,
        _info,
        _rng,
    ) -> float:
        """Per-agent reward for one aircraft this step."""
        return 0.0

    @overridable
    def terminated(self, _obs, _action, _context, _info, _rng) -> bool:
        """Per-agent termination (goal reached / out of bounds / ...)."""
        return False

    @overridable
    def truncated(self, _obs, _action, _context, _info, _rng) -> bool:
        """Per-agent truncation (time limit / give-up condition)."""
        return False

    def read_aircraft_obs_field(
        self,
        idx: int,
        field: type[ObsField] | ObsField,
    ) -> Any:
        """Read one raw per-aircraft observation value.

        Passing an ``ObsField`` class is shorthand for using its default
        constructor, e.g. ``read_aircraft_obs_field(idx, obs.LatDeg)``. Pass
        an instance when the read should use configured field state such as
        custom bounds, names, or normalizers.
        """
        return self._resolve_obs_field(field).get(idx)

    def _resolve_obs_field(self, field: type[ObsField] | ObsField) -> ObsField:
        field_obj = field() if isinstance(field, type) else field
        if not isinstance(field_obj, ObsField):
            raise TypeError(
                "read_aircraft_obs_field expects an ObsField class or instance, "
                f"got {field!r}."
            )
        if isinstance(field_obj, EnvObsField):
            field_obj = field_obj.bind_env(self)
        return field_obj

    def read_aircraft_obs_fields(
        self,
        idx: int,
        *fields: type[ObsField] | ObsField,
    ) -> tuple[Any, ...]:
        """Read several raw per-aircraft observation values in order."""
        return tuple(self.read_aircraft_obs_field(idx, field) for field in fields)
