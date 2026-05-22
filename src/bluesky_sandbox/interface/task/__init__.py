"""Task-facing types and the concrete providers that implement them.

Declarations only: the dataclasses, protocols and hook signatures a task is
written against. Concrete task-info providers are not here on purpose - what
counts as a cost or a goal is the task's business, so that code lives in the
task package (the designer scaffolds it for you).
"""

from __future__ import annotations

from .types import (
    AchievedGoalFn,
    AgentStepContext,
    AircraftReadoutItem,
    AircraftRenderState,
    BaseAgentInfo,
    BaseObs,
    ConstraintFn,
    ConstraintTaskInfo,
    DesiredGoalFn,
    Goal,
    GoalTaskInfo,
    QueryableTemporalStateUnavailable,
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
    UnavailableStepTime,
    WaypointReadoutItem,
    WaypointReadoutKey,
    WaypointReadoutNamespace,
    WaypointReadoutTarget,
)

__all__ = [
    "AchievedGoalFn",
    "AgentStepContext",
    "AircraftReadoutItem",
    "AircraftRenderState",
    "BaseAgentInfo",
    "BaseObs",
    "ConstraintFn",
    "ConstraintTaskInfo",
    "DesiredGoalFn",
    "Goal",
    "GoalTaskInfo",
    "QueryableTemporalStateUnavailable",
    "RewardFn",
    "SeparationContext",
    "SeparationEvent",
    "SeparationEventInfo",
    "SeparationInfo",
    "StepEvent",
    "StepTime",
    "TaskInfo",
    "TaskInfoProvider",
    "TerminationFn",
    "TruncationFn",
    "UnavailableStepTime",
    "WaypointReadoutItem",
    "WaypointReadoutKey",
    "WaypointReadoutNamespace",
    "WaypointReadoutTarget",
]
