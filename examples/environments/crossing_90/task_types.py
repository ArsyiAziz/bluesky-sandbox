"""Types for this task's code, generated from its design.

Editors and type checkers read these to complete and check the design's
keys - observation parts and fields, action fields, queryables. Nothing
here runs: hooks receive the library's objects, and these subclasses only
describe what those objects hold for this task. Regenerate the package
to update them.
"""

# Narrowing the library's types, in methods without bodies, is the point:
# say so to type checkers.
# pyright: reportIncompatibleVariableOverride=false, reportIncompatibleMethodOverride=false
# mypy: disable-error-code="assignment, override, empty-body"

from __future__ import annotations

from bluesky_sandbox import AgentStepContext, RegionBounds, StepBatch, Waypoint, WaypointResult
from bluesky_sandbox.core.services import QueryBatch
from numpy import ndarray
from typing import Any, Literal, TypedDict, overload


class Queryables(TypedDict):
    exit: Waypoint
    """Waypoint - A place aircraft are evaluated against: a BlueSky navdb fix or an
ad-hoc point."""


class ObsOwnship(TypedDict):
    active_route_waypoint_distance_nm: float
    """float · nm - ActiveRouteWaypointDistanceNm: Distance from ownship to its active route waypoint, nm (0 if none)."""
    active_route_waypoint_track_error_deg: float
    """float · deg - ActiveRouteWaypointTrackErrorDeg: Signed track error from ownship track toward its active route waypoint."""
    alt_ft: float
    """float · ft - AltFt: Aircraft altitude in feet."""
    cas_kts: float
    """float · kts - CasKts: Calibrated airspeed in knots."""
    vs_ftmin: float
    """float · ft/min - VsFtMin: Vertical speed in ft/min."""
    trk_deg: float
    """float · deg - TrkDeg: Aircraft track angle in degrees."""
    gs_kts: float
    """float · kts - GsKts: Ground speed in knots."""
    ap_cas_kts: float
    """float · kts - ApCasKts: Autopilot selected calibrated airspeed in knots."""
    perf_vmin_kts: float
    """float · kts - PerfVminKts: Minimum operating CAS from the performance model, in knots."""
    perf_vmax_kts: float
    """float · kts - PerfVmaxKts: Maximum operating CAS from the performance model, in knots."""


class ObsIntruders(TypedDict):
    acid: ndarray
    """ndarray (n_intruders,) str - Each intruder row's callsign."""
    rel_pos_along_track_nm: ndarray
    """ndarray (n_intruders,) · nm - RelPosAlongTrackNm: Intruder position relative to ownship ALONG the own track, nm (ahead +)."""
    rel_pos_cross_track_nm: ndarray
    """ndarray (n_intruders,) · nm - RelPosCrossTrackNm: Intruder position relative to ownship ACROSS the own track, nm (right +)."""
    relative_alt_ft: ndarray
    """ndarray (n_intruders,) · ft - Difference: Difference between an intruder field and an ownship field."""
    rel_vel_along_track_kts: ndarray
    """ndarray (n_intruders,) · kts - RelVelAlongTrackKts: Intruder velocity relative to ownship ALONG the own track, kts."""
    rel_vel_cross_track_kts: ndarray
    """ndarray (n_intruders,) · kts - RelVelCrossTrackKts: Intruder velocity relative to ownship ACROSS the own track, kts (right +)."""
    rel_vs_ft_min: ndarray
    """ndarray (n_intruders,) · ft/min - RelVsFtMin: Intruder vertical speed minus ownship vertical speed, in ft/min."""
    dist_to_own_nm: ndarray
    """ndarray (n_intruders,) · nm - DistToOwnNm: Ownship-relative intruder distance in nautical miles."""
    conflict_tcpa_s: ndarray
    """ndarray (n_intruders,) · s - ConflictTcpaS: Time to horizontal CPA (s), for every intruder; negative past it."""
    conflict_tlos_s: ndarray
    """ndarray (n_intruders,) · s - ConflictTlosS: Time until the pair enters the protected zone (BlueSky tLOS), s."""
    in_conf: ndarray
    """ndarray (n_intruders,) - InConf: 1.0 when BlueSky's conflict detection flags the pair in conflict now."""
    in_los_now: ndarray
    """ndarray (n_intruders,) - InLosNow: 1.0 when the intruder is inside the ownship's protected zone now - a loss
of separation, as BlueSky's conflict detection scores it on the current
state (:func:~bluesky_sandbox.sim.geometry.conflict.detection)."""


class Obs(TypedDict):
    ownship: ObsOwnship
    """10 fields"""
    intruders: ObsIntruders
    """11 fields"""


class State(TypedDict):
    pass


Action = TypedDict('Action', {'hdg_deg': float, 'action_mask': float, 'spd_kts': float, 'action_mask#2': float, 'autopilot_lnav': float, 'action_mask#3': float})


class Shapes(TypedDict):
    airspace: RegionBounds
    """Disk - Bounds built from a horizontal footprint and an altitude band."""
    exit: RegionBounds
    """Point - Bounds built from a horizontal footprint and an altitude band."""


class ObsBatchOwnship(TypedDict):
    active_route_waypoint_distance_nm: ndarray
    """ndarray (n_agents,) · nm - ActiveRouteWaypointDistanceNm: Distance from ownship to its active route waypoint, nm (0 if none)."""
    active_route_waypoint_track_error_deg: ndarray
    """ndarray (n_agents,) · deg - ActiveRouteWaypointTrackErrorDeg: Signed track error from ownship track toward its active route waypoint."""
    alt_ft: ndarray
    """ndarray (n_agents,) · ft - AltFt: Aircraft altitude in feet."""
    cas_kts: ndarray
    """ndarray (n_agents,) · kts - CasKts: Calibrated airspeed in knots."""
    vs_ftmin: ndarray
    """ndarray (n_agents,) · ft/min - VsFtMin: Vertical speed in ft/min."""
    trk_deg: ndarray
    """ndarray (n_agents,) · deg - TrkDeg: Aircraft track angle in degrees."""
    gs_kts: ndarray
    """ndarray (n_agents,) · kts - GsKts: Ground speed in knots."""
    ap_cas_kts: ndarray
    """ndarray (n_agents,) · kts - ApCasKts: Autopilot selected calibrated airspeed in knots."""
    perf_vmin_kts: ndarray
    """ndarray (n_agents,) · kts - PerfVminKts: Minimum operating CAS from the performance model, in knots."""
    perf_vmax_kts: ndarray
    """ndarray (n_agents,) · kts - PerfVmaxKts: Maximum operating CAS from the performance model, in knots."""


class ObsBatchIntruders(TypedDict):
    acid: ndarray
    """ndarray (n_agents, n_intruders) str - Each intruder row's callsign."""
    rel_pos_along_track_nm: ndarray
    """ndarray (n_agents, n_intruders) · nm - RelPosAlongTrackNm: Intruder position relative to ownship ALONG the own track, nm (ahead +)."""
    rel_pos_cross_track_nm: ndarray
    """ndarray (n_agents, n_intruders) · nm - RelPosCrossTrackNm: Intruder position relative to ownship ACROSS the own track, nm (right +)."""
    relative_alt_ft: ndarray
    """ndarray (n_agents, n_intruders) · ft - Difference: Difference between an intruder field and an ownship field."""
    rel_vel_along_track_kts: ndarray
    """ndarray (n_agents, n_intruders) · kts - RelVelAlongTrackKts: Intruder velocity relative to ownship ALONG the own track, kts."""
    rel_vel_cross_track_kts: ndarray
    """ndarray (n_agents, n_intruders) · kts - RelVelCrossTrackKts: Intruder velocity relative to ownship ACROSS the own track, kts (right +)."""
    rel_vs_ft_min: ndarray
    """ndarray (n_agents, n_intruders) · ft/min - RelVsFtMin: Intruder vertical speed minus ownship vertical speed, in ft/min."""
    dist_to_own_nm: ndarray
    """ndarray (n_agents, n_intruders) · nm - DistToOwnNm: Ownship-relative intruder distance in nautical miles."""
    conflict_tcpa_s: ndarray
    """ndarray (n_agents, n_intruders) · s - ConflictTcpaS: Time to horizontal CPA (s), for every intruder; negative past it."""
    conflict_tlos_s: ndarray
    """ndarray (n_agents, n_intruders) · s - ConflictTlosS: Time until the pair enters the protected zone (BlueSky tLOS), s."""
    in_conf: ndarray
    """ndarray (n_agents, n_intruders) - InConf: 1.0 when BlueSky's conflict detection flags the pair in conflict now."""
    in_los_now: ndarray
    """ndarray (n_agents, n_intruders) - InLosNow: 1.0 when the intruder is inside the ownship's protected zone now - a loss
of separation, as BlueSky's conflict detection scores it on the current
state (:func:~bluesky_sandbox.sim.geometry.conflict.detection)."""


class ObsBatch(TypedDict):
    ownship: ObsBatchOwnship
    """10 fields"""
    intruders: ObsBatchIntruders
    """11 fields"""


class StateBatch(TypedDict):
    pass


ActionBatch = TypedDict('ActionBatch', {'hdg_deg': ndarray, 'action_mask': ndarray, 'spd_kts': ndarray, 'action_mask#2': ndarray, 'autopilot_lnav': ndarray, 'action_mask#3': ndarray})


class TaskAgentStepContext(AgentStepContext):
    """``AgentStepContext``, typed for this task: the members that hold
    the design's keys, as this design has them."""

    queryables: Queryables
    obs: Obs
    state: State
    action: Action
    shapes: Shapes
    def bounds(self, name: Literal['airspace', 'exit']) -> RegionBounds: ...
    @overload
    def query(self, name: Literal['exit']) -> WaypointResult: ...
    def query(self, name: str) -> Any: ...
    @overload
    def queryable(self, name: Literal['exit']) -> Waypoint: ...
    def queryable(self, name: str) -> Any: ...
    def shape(self, name: Literal['airspace', 'exit']) -> RegionBounds: ...


class TaskStepBatch(StepBatch):
    """``StepBatch``, typed for this task: the members that hold
    the design's keys, as this design has them."""

    obs: ObsBatch
    state: StateBatch
    action: ActionBatch
    def bounds(self, name: Literal['airspace', 'exit']) -> RegionBounds: ...
    def query(self, name: Literal['exit']) -> QueryBatch: ...
    def shape(self, name: Literal['airspace', 'exit']) -> RegionBounds: ...
    def context(self, k: int) -> TaskAgentStepContext: ...
