"""Task-facing types: step context, readouts, and the hook protocols.

Pure declarations - dataclasses, ``StrEnum``s and ``Protocol``s naming the
callables a task supplies (:class:`RewardFn`, :class:`TerminationFn`,
:class:`TruncationFn`). The concrete providers that *implement* task info live
in :mod:`.providers`; both are re-exported from the package, so
``from bluesky_sandbox.interface.task import X`` keeps working for either.
"""


from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, NoReturn, NotRequired, Protocol, TypedDict

import bluesky as bs
import numpy as np

from bluesky_sandbox.sim.geometry.conflict import ConflictView

#: Flat array (no intruder obs) or dict with "ownship" / "intruders" keys.
BaseObs = np.ndarray | dict[str, np.ndarray | tuple[np.ndarray, ...]]


class SeparationEventInfo(TypedDict):
    """Conflict or loss-of-separation status for one agent."""

    current: bool
    during_step: bool
    substeps: int
    partners: list[str]
    step_partners: list[str]
    time: NotRequired[dict[str, float]]


class SeparationInfo(TypedDict):
    """Grouped separation information for one agent."""

    conflict: SeparationEventInfo
    los: SeparationEventInfo


Goal = dict[str, Any]


class GoalTaskInfo(TypedDict):
    """Goal-conditioned payload a task's goal provider writes."""

    achieved: Goal
    desired: Goal


class ConstraintTaskInfo(TypedDict):
    """Constrained-RL payload a task's constraint provider writes."""

    cost: np.ndarray
    names: tuple[str, ...]
    limits: np.ndarray
    violated: np.ndarray
    extrinsic_cost: NotRequired[np.ndarray]
    intrinsic_cost: NotRequired[np.ndarray]


TaskInfo = dict[str, object]
"""Task-owned public diagnostics returned under ``info["task"]``."""


AircraftRenderState = Literal["normal", "violation", "conflict", "los"]
"""Renderer-owned aircraft marker state consumed by human drivers."""


@dataclass(frozen=True)
class StepTime:
    """Time spent in a boolean step event."""

    total_s: float = 0.0
    during_step_s: float = 0.0


class QueryableTemporalStateUnavailable(RuntimeError):
    """Raised when unavailable query temporal state is accessed."""


_QUERYABLE_TEMPORAL_STATE_UNAVAILABLE = (
    "Queryable temporal state is unavailable because "
    "track_temporal_state is False for this queryable. Enable "
    "track_temporal_state on the queryable or use current-state query values only."
)


def _raise_queryable_temporal_state_unavailable(
    reason: str = _QUERYABLE_TEMPORAL_STATE_UNAVAILABLE,
) -> NoReturn:
    raise QueryableTemporalStateUnavailable(reason)


@dataclass(frozen=True)
class UnavailableStepTime:
    """Step-time placeholder for query results without temporal tracking."""

    reason: str = _QUERYABLE_TEMPORAL_STATE_UNAVAILABLE

    @property
    def total_s(self) -> float:
        _raise_queryable_temporal_state_unavailable(self.reason)

    @property
    def during_step_s(self) -> float:
        _raise_queryable_temporal_state_unavailable(self.reason)


@dataclass(frozen=True)
class StepEvent:
    """Common boolean event shape used by context query results."""

    current: bool = False
    during_step: bool = False
    substeps: int = 0
    time: StepTime = field(default_factory=StepTime)

    def __bool__(self) -> bool:
        return self.current


@dataclass(frozen=True)
class SeparationEvent:
    """Conflict/loss-of-separation event in agent context."""

    current: bool = False
    during_step: bool = False
    substeps: int = 0
    partners: tuple[str, ...] = ()
    step_partners: tuple[str, ...] = ()
    time: StepTime = field(default_factory=StepTime)

    def __bool__(self) -> bool:
        return self.current

    def as_info(self) -> dict[str, object]:
        return {
            "current": self.current,
            "during_step": self.during_step,
            "substeps": self.substeps,
            "partners": list(self.partners),
            "step_partners": list(self.step_partners),
            "time": {
                "total_s": self.time.total_s,
                "during_step_s": self.time.during_step_s,
            },
        }


@dataclass(frozen=True)
class SeparationContext:
    """Grouped separation events in agent context."""

    conflict: SeparationEvent = field(default_factory=SeparationEvent)
    los: SeparationEvent = field(default_factory=SeparationEvent)

    def as_info(self) -> dict[str, object]:
        return {
            "conflict": self.conflict.as_info(),
            "los": self.los.as_info(),
        }


@dataclass(frozen=True)
class AircraftReadoutItem:
    """Atomic aircraft-level readout row for GUI drivers."""

    label: str
    value: object
    color: str | tuple[int, int, int] | None = None
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("AircraftReadoutItem.label must be non-empty")


@dataclass(frozen=True)
class WaypointReadoutTarget:
    """Selector for one route waypoint in aircraft readouts.

    ``future=True`` with no name or index selects the next future waypoint.
    """

    name: str | None = None
    index: int | None = None
    active: bool | None = None
    future: bool | None = None

    def __post_init__(self) -> None:
        if (
            self.name is None
            and self.index is None
            and self.active is None
            and self.future is None
        ):
            raise ValueError("WaypointReadoutTarget must define at least one selector")


class WaypointReadoutNamespace(StrEnum):
    """Destination bucket for a waypoint readout item."""

    METADATA = "metadata"
    CONSTRAINTS = "constraints"


class WaypointReadoutKey(StrEnum):
    """Standard waypoint readout fields consumed by GUI drivers."""

    NAME = "name"
    TARGET_ALT_FT = "target_alt_ft"
    TARGET_SPEED_KTS = "target_speed_kts"
    RADIUS_NM = "radius_nm"
    ALT_TOLERANCE_FT = "alt_tolerance_ft"
    SPEED_KTS = "speed_kts"
    SPEED_MIN_KTS = "speed_min_kts"
    SPEED_MAX_KTS = "speed_max_kts"
    SPEED_TOLERANCE_KTS = "speed_tolerance_kts"
    GOAL_TYPE = "goal_type"


@dataclass(frozen=True)
class WaypointReadoutItem:
    """Atomic readout annotation for one selected route waypoint."""

    target: WaypointReadoutTarget
    namespace: WaypointReadoutNamespace
    key: str | WaypointReadoutKey
    value: object

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("WaypointReadoutItem.key must be non-empty")


@dataclass(frozen=True)
class AgentStepContext:
    """Agent-bound task/development context for the current step."""

    acid: str
    acidx: int
    data: Any = None
    queryables: Mapping[str, Any] = field(default_factory=dict, repr=False)
    query_state: Any = field(default=None, repr=False)
    airspace: Any = field(default=None, repr=False)
    separation: SeparationContext = field(default_factory=SeparationContext)
    _query_result_cache: dict[str, Any] = field(default_factory=dict, repr=False)
    _obs_value_cache: dict[int, Any] = field(default_factory=dict, repr=False)

    def queryable(self, name: str) -> Any:
        try:
            return self.queryables[name]
        except KeyError as exc:
            raise KeyError(f"queryable {name!r} is not configured") from exc

    @property
    def conflicts(self) -> Any:
        """The ownship's shared per-step conflict geometry view.

        A :class:`~bluesky_sandbox.sim.geometry.conflict.ConflictView` over the other
        live aircraft, backed by one cached pairwise-CPA computation shared with
        the cost and keep mask. Exposes only raw kinematic primitives (derived
        breach/severity is the consumer's job): per-intruder ``.tcpa_s`` /
        ``.dcpa_nm`` / ``.vsep_at_cpa_ft`` / ``.horiz_dist_now_nm`` /
        ``.dalt_now_ft``, reductions ``.min_tcpa_s`` / ``.min_dcpa_nm`` /
        ``.min_horiz_dist_now_nm``, the ``.within(horizon_s=...)`` filter, and
        iteration over per-intruder records.
        """

        return ConflictView(self.acidx)

    def query(self, name: str) -> Any:
        if name in self._query_result_cache:
            return self._query_result_cache[name]
        queryable = self.queryable(name)
        if self.query_state is not None:
            result = self.query_state.query(self.acid, self.acidx, name, queryable)
        else:
            result = queryable.result_type.for_aircraft(queryable, self.acidx)
        self._query_result_cache[name] = result
        return result

    def own_value(self, field: Any) -> Any:
        """Raw (un-normalized) value of an ownship obs ``field`` for this agent.

        Reads the field's own getter, bypassing the policy's normalizers/ordering
        - use this in cost/reward code instead of slicing the observation vector.
        Computed lazily and cached for the step.
        """
        key = id(field)
        cached = self._obs_value_cache.get(key)
        if cached is not None:
            return cached
        value = field.get(self.acidx)
        self._obs_value_cache[key] = value
        return value

    def intruder_values(self, field: Any) -> np.ndarray:
        """Raw (un-normalized) values of a pair obs ``field`` over **all** other
        live aircraft (global - independent of the policy's intruder selection).

        Computed lazily and cached for the step, so repeated reads of the same
        field are free. Returns an empty array when the agent is alone.
        """
        key = id(field)
        cached = self._obs_value_cache.get(key)
        if cached is not None:
            return cached
        others = tuple(i for i in range(bs.traf.ntraf) if i != self.acidx)
        values = (
            np.empty(0, dtype=np.float64)
            if not others
            else np.asarray(field.get_pairs(self.acidx, others), dtype=np.float64)
        )
        self._obs_value_cache[key] = values
        return values


class BaseAgentInfo(TypedDict):
    """Public info dict returned for one live agent."""
    acid: str                       # agent callsign / unique identifier
    acidx: int                      # BlueSky traffic index while callbacks run
    type: str                       # ICAO aircraft type, e.g. "B744"
    performance_model: str          # resolved BlueSky performance model name

    phase: str                      # flight phase string from BlueSky
    time_in_env: float              # seconds since the agent was spawned
    in_airspace: bool               # True if agent is within the airspace bounds

    substeps: int                   # Number of internal sim steps sampled
    separation: SeparationInfo      # grouped conflict / LoS status
    task: TaskInfo                  # explicit task-owned public diagnostics


class AchievedGoalFn(Protocol):
    """Return the achieved goal for one agent after a step."""

    def __call__(
        self,
        obs: BaseObs,
        info: BaseAgentInfo,
        context: AgentStepContext,
    ) -> Goal: ...


class DesiredGoalFn(Protocol):
    """Return the active desired goal for one agent after a step."""

    def __call__(
        self,
        obs: BaseObs,
        info: BaseAgentInfo,
        context: AgentStepContext,
    ) -> Goal: ...


class TaskInfoProvider(Protocol):
    """Ordered hook that populates public ``info["task"]`` payloads."""

    def __call__(
        self,
        obs: BaseObs,
        action: np.ndarray | None,
        info: BaseAgentInfo,
        context: AgentStepContext,
        rng: np.random.Generator,
    ) -> None: ...


class ConstraintFn(Protocol):
    """Compute a vector of constrained-RL costs for one agent after a step."""

    def __call__(
        self,
        obs: BaseObs,
        action: np.ndarray | None,
        info: BaseAgentInfo,
        context: AgentStepContext,
        rng: np.random.Generator,
    ) -> np.ndarray: ...


class RewardFn(Protocol):
    """Compute the scalar reward for one agent after a sim step.

    Parameters
    ----------
    obs:
        The agent's observation for this step.
    action:
        The action applied to this agent for this step, in the env's
        ``action_space`` units (i.e. the values handed to
        ``BlueskyEnv.step``). Fields with action normalizers are denormalized
        before BlueSky dispatch. ``None`` if the caller did not supply an
        action for this agent (e.g. an aircraft that spawned mid-step).
    terminated:
        Whether the agent was terminated this step (natural end, e.g. conflict).
    truncated:
        Whether the agent was truncated this step (artificial end, e.g. time limit).
    context:
        Agent-bound task/development context for this step.
    info:
        The public info dict returned for this step.
    rng:
        The environment's seeded random generator.

    Returns
    -------
    float

    Example
    -------
    ::

        def my_reward(obs, action, terminated, truncated, context, info, rng):
            return -1.0 if terminated else 0.0
    """
    def __call__(
        self,
        obs: BaseObs,
        action: np.ndarray | None,
        terminated: bool,
        truncated: bool,
        context: AgentStepContext,
        info: BaseAgentInfo,
        rng: np.random.Generator,
    ) -> float: ...


class TerminationFn(Protocol):
    """Decide whether an agent has reached a natural episode end.

    A terminated agent leaves the controllable agent set. Tasks may keep
    the aircraft simulated as background traffic or let the base environment
    delete it immediately.

    Parameters
    ----------
    obs:
        The agent's observation for this step.
    action:
        The action applied to this agent for this step, in the env's
        ``action_space`` units. ``None`` if no action was supplied for
        this agent.
    context:
        Agent-bound task/development context for this step.
    info:
        The public info dict returned for this step.
    rng:
        The environment's seeded random generator.

    Returns
    -------
    bool
        ``True`` to terminate the controllable agent episode.

    Example
    -------
    ::

        def my_termination(obs, action, context, info, rng):
            return not info['in_airspace']
    """
    def __call__(
        self,
        obs: BaseObs,
        action: np.ndarray | None,
        context: AgentStepContext,
        info: BaseAgentInfo,
        rng: np.random.Generator,
    ) -> bool: ...


class TruncationFn(Protocol):
    """Decide whether an agent has hit an artificial episode cutoff.

    A truncated agent leaves the controllable agent set. Tasks may keep
    the aircraft simulated as background traffic or let the base environment
    delete it immediately.

    Parameters
    ----------
    obs:
        The agent's observation for this step.
    action:
        The action applied to this agent for this step, in the env's
        ``action_space`` units. ``None`` if no action was supplied for
        this agent.
    context:
        Agent-bound task/development context for this step.
    info:
        The public info dict returned for this step.
    rng:
        The environment's seeded random generator.

    Returns
    -------
    bool
        ``True`` to truncate the controllable agent episode.

    Example
    -------
    ::

        def my_truncation(obs, action, context, info, rng):
            return info['time_in_env'] >= 3600.0
    """
    def __call__(
        self,
        obs: BaseObs,
        action: np.ndarray | None,
        context: AgentStepContext,
        info: BaseAgentInfo,
        rng: np.random.Generator,
    ) -> bool: ...
