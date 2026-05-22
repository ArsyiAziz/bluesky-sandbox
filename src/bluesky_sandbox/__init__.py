__version__ = "0.1.0"

import sys as _sys
from typing import TYPE_CHECKING

from bluesky_sandbox.core.base_environment import AircraftControlState
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.integrations.asymmetric import (
    actor_obs,
    actor_observation_space,
    critic_obs,
    critic_observation_space,
    has_privileged_obs,
)
from bluesky_sandbox.interface.fields import actions
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.fields.base import (
    ActionField,
    ActionMeta,
    ActionMode,
    ControlAxis,
    EnvObsField,
    EnvPairObsField,
    ObsField,
    ObsMeta,
    ObsQuantity,
    PairObsField,
    QueryableFieldCardinality,
    QueryableFieldRequirement,
    QueryableFieldSpec,
    QueryableKind,
    SwitchActionMixin,
    TaskContextObsField,
    TaskContextPairObsField,
    Unit,
)
from bluesky_sandbox.interface.task import (
    AchievedGoalFn,
    AgentStepContext,
    AircraftReadoutItem,
    BaseAgentInfo,
    BaseObs,
    ConstraintFn,
    ConstraintTaskInfo,
    DesiredGoalFn,
    Goal,
    GoalTaskInfo,
    RewardFn,
    SeparationContext,
    SeparationEvent,
    SeparationEventInfo,
    SeparationInfo,
    StepEvent,
    StepTime,
    TaskInfo,
    TaskInfoProvider,
    TerminationFn,
    TruncationFn,
    WaypointReadoutItem,
    WaypointReadoutKey,
    WaypointReadoutNamespace,
    WaypointReadoutTarget,
)
from bluesky_sandbox.interface.wrappers import (
    CircularNormalizer,
    MinMaxNormalizer,
    Normalizer,
    PerFieldNormalizer,
    RawNormalizer,
    SymmetricNormalizer,
)
from bluesky_sandbox.sim.bounds import (
    AltitudeBand,
    AnnularSectorFootprint,
    BooleanFootprint,
    Bounds,
    BoxFootprint,
    ConstantAltitudeBand,
    CorridorFootprint,
    DiskFootprint,
    Footprint,
    LatLon,
    LinearAltitudeBand,
    LocalFrame,
    PolygonFootprint,
    RadialAltitudeBand,
    RegionBounds,
    SectorFootprint,
    VertexAltitudeBand,
)
from bluesky_sandbox.sim.performance.envelope import EnvelopeSample
from bluesky_sandbox.sim.queryables import (
    Queryable,
    QueryRegion,
    RegionCurrent,
    RegionResult,
    RegionStep,
    Waypoint,
    WaypointCurrent,
    WaypointResult,
    WaypointRoute,
    WaypointStep,
    WaypointTarget,
)
from bluesky_sandbox.sim.sampling.distributions import (
    Categorical,
    CountDistribution,
    TypeDistribution,
)
from bluesky_sandbox.sim.scenario import EpisodeSpec, Scenario
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion
from bluesky_sandbox.ui import drivers as _drivers
from bluesky_sandbox.ui.display.readouts import waypoint_readouts

_sys.modules.setdefault(__name__ + ".driver", _drivers)


if TYPE_CHECKING:
    from bluesky_sandbox.interface.fields import queryables as qobs


def __getattr__(name: str):
    if name == "qobs":
        from importlib import import_module

        module = import_module(f"{__name__}.fields.queryables")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AchievedGoalFn",
    "ActionField",
    "ActionMeta",
    "ActionMode",
    "AgentStepContext",
    "AircraftControlState",
    "AircraftReadoutItem",
    "AltitudeBand",
    "AnnularSectorFootprint",
    "BaseAgentInfo",
    "BaseObs",
    "BlueskyEnv",
    "BooleanFootprint",
    "Bounds",
    "BoxFootprint",
    "Categorical",
    "CircularNormalizer",
    "ConstantAltitudeBand",
    "ConstraintFn",
    "ConstraintTaskInfo",
    "ControlAxis",
    "CorridorFootprint",
    "CountDistribution",
    "DesiredGoalFn",
    "DiskFootprint",
    "EnvObsField",
    "EnvPairObsField",
    "EnvelopeSample",
    "EpisodeSpec",
    "Footprint",
    "Goal",
    "GoalTaskInfo",
    "LatLon",
    "LinearAltitudeBand",
    "LocalFrame",
    "MinMaxNormalizer",
    "Normalizer",
    "ObsField",
    "ObsMeta",
    "ObsQuantity",
    "PairObsField",
    "PerFieldNormalizer",
    "PolygonFootprint",
    "QueryRegion",
    "Queryable",
    "QueryableFieldCardinality",
    "QueryableFieldRequirement",
    "QueryableFieldSpec",
    "QueryableKind",
    "RadialAltitudeBand",
    "RawNormalizer",
    "RegionBounds",
    "RegionCurrent",
    "RegionResult",
    "RegionStep",
    "RewardFn",
    "Scenario",
    "SectorFootprint",
    "SeparationContext",
    "SeparationEvent",
    "SeparationEventInfo",
    "SeparationInfo",
    "SpawnConfig",
    "SpawnRegion",
    "StepEvent",
    "StepTime",
    "SwitchActionMixin",
    "SymmetricNormalizer",
    "TaskContextObsField",
    "TaskContextPairObsField",
    "TaskInfo",
    "TaskInfoProvider",
    "TerminationFn",
    "TruncationFn",
    "TypeDistribution",
    "Unit",
    "VertexAltitudeBand",
    "Waypoint",
    "WaypointCurrent",
    "WaypointReadoutItem",
    "WaypointReadoutKey",
    "WaypointReadoutNamespace",
    "WaypointReadoutTarget",
    "WaypointResult",
    "WaypointRoute",
    "WaypointStep",
    "WaypointTarget",
    "actions",
    "actor_obs",
    "actor_observation_space",
    "critic_obs",
    "critic_observation_space",
    "has_privileged_obs",
    "obs",
    "qobs",
    "waypoint_readouts",
]
