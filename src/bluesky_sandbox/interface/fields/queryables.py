"""Observation fields projected from configured queryable results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bluesky_sandbox.sim.queryables import WaypointResult

from .base import (
    EnvObsField,
    ObsMeta,
    ObsQuantity,
    QueryableFieldCardinality,
    QueryableFieldRequirement,
    QueryableFieldSpec,
    QueryableKind,
    Unit,
)


@dataclass(frozen=True)
class QueryableObsField(EnvObsField):
    query_name: str = ""
    low: float = 0.0
    high: float = 1.0
    normalizer: Any | None = None

    @property
    def bound_queryable(self):
        queryables = self.bound_env.episode_queryables
        try:
            return queryables[self.query_name]
        except KeyError as exc:
            raise KeyError(f"queryable {self.query_name!r} is not configured") from exc

    def bounds(self, idx: int) -> tuple[float, float]:
        del idx
        return float(self.low), float(self.high)

    def query_result(self, idx: int):
        return self.query_result_for(self.query_name, idx)

    def query_result_for(self, query_name: str, idx: int):
        return self.bound_env.agent_context(idx).query(query_name)


@dataclass(frozen=True)
class WaypointResultObsField(QueryableObsField):
    def waypoint_result(self, idx: int) -> WaypointResult:
        result = self.query_result(idx)
        if not isinstance(result, WaypointResult):
            raise TypeError(
                f"{self.query_name!r} must return WaypointResult, "
                f"got {type(result)!r}"
            )
        return result


@dataclass(frozen=True)
class ActiveWaypointObsField(QueryableObsField):
    """Observation field projected from the active waypoint query result."""

    query_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "query_names",
            tuple(str(name) for name in self.query_names),
        )

    def active_waypoint(self, idx: int) -> tuple[str, WaypointResult] | None:
        queryables = self.bound_env.episode_queryables
        names = self.query_names or tuple(queryables)
        candidates: list[tuple[str, WaypointResult]] = []
        for name in names:
            if name not in queryables:
                continue
            result = self.query_result_for(name, idx)
            if not isinstance(result, WaypointResult):
                continue
            candidates.append((name, result))
            if result.route.active:
                return candidates[-1]
        for candidate in candidates:
            if candidate[1].route.future:
                return candidate
        return None


@dataclass(frozen=True)
class QueryRegionInside(QueryableObsField):
    """Whether the queryable result is true for ownship."""

    queryable_spec = QueryableFieldSpec(
        kind=QueryableKind.REGION,
        path="current.inside",
        label="Inside",
        description="Whether ownship is currently inside the region.",
    )

    @property
    def meta(self) -> ObsMeta:
        return ObsMeta(
            f"{self.query_name.lower()}_inside",
            Unit.SWITCH,
            ObsQuantity.DISTANCE,
        )

    def get(self, idx: int) -> float:
        return 1.0 if bool(self.query_result(idx)) else 0.0


@dataclass(frozen=True)
class QueryRegionInsideDuringStep(QueryableObsField):
    """Whether ownship entered/was inside the query region during this step."""

    queryable_spec = QueryableFieldSpec(
        kind=QueryableKind.REGION,
        path="step.inside",
        label="Inside during step",
        description="Whether ownship was inside the region during this env step.",
        requirements=(QueryableFieldRequirement.STEP,),
    )

    @property
    def meta(self) -> ObsMeta:
        return ObsMeta(
            f"{self.query_name.lower()}_inside_during_step",
            Unit.SWITCH,
            ObsQuantity.PHASE,
        )

    def get(self, idx: int) -> float:
        result = self.query_result(idx)
        return 1.0 if bool(result.step.inside) else 0.0


@dataclass(frozen=True)
class QueryRegionInsideTimeTotalS(QueryableObsField):
    """Total seconds ownship has spent inside the query region."""

    queryable_spec = QueryableFieldSpec(
        kind=QueryableKind.REGION,
        path="time.total_s",
        label="Time inside",
        description="Total simulated seconds ownship has spent inside the region.",
        requirements=(QueryableFieldRequirement.TIME,),
    )

    high: float = 3600.0

    @property
    def meta(self) -> ObsMeta:
        return ObsMeta(
            f"{self.query_name.lower()}_inside_time_total_s",
            Unit.S,
            ObsQuantity.TIME,
        )

    def get(self, idx: int) -> float:
        result = self.query_result(idx)
        time = getattr(result, "time", None)
        return float(getattr(time, "total_s", 0.0))


@dataclass(frozen=True)
class WaypointDistanceNm(WaypointResultObsField):
    """Distance from ownship to a waypoint query result."""

    queryable_spec = QueryableFieldSpec(
        kind=QueryableKind.WAYPOINT,
        path="current.distance_nm",
        label="Distance",
        description="Distance from ownship to the waypoint in nautical miles.",
    )

    @property
    def meta(self) -> ObsMeta:
        return ObsMeta(
            f"{self.query_name.lower()}_distance_nm",
            Unit.NM,
            ObsQuantity.DISTANCE,
        )

    def get(self, idx: int) -> float:
        return self.waypoint_result(idx).current.distance_nm


@dataclass(frozen=True)
class WaypointBearingDeg(WaypointResultObsField):
    """Bearing from ownship to a waypoint query result."""

    queryable_spec = QueryableFieldSpec(
        kind=QueryableKind.WAYPOINT,
        path="current.bearing_deg",
        label="Bearing",
        description="True bearing from ownship to the waypoint.",
    )

    low: float = 0.0
    high: float = 360.0

    @property
    def meta(self) -> ObsMeta:
        return ObsMeta(
            f"{self.query_name.lower()}_bearing_deg",
            Unit.DEG,
            ObsQuantity.BEARING,
            circular=True,
        )

    def get(self, idx: int) -> float:
        return self.waypoint_result(idx).current.bearing_deg


@dataclass(frozen=True)
class WaypointTrackErrorDeg(WaypointResultObsField):
    """Signed track error from ownship track toward a waypoint query result."""

    queryable_spec = QueryableFieldSpec(
        kind=QueryableKind.WAYPOINT,
        path="current.track_error_deg",
        label="Track error",
        description="Signed ownship track error toward the waypoint.",
    )

    low: float = -180.0
    high: float = 180.0

    @property
    def meta(self) -> ObsMeta:
        return ObsMeta(
            f"{self.query_name.lower()}_track_error_deg",
            Unit.DEG,
            ObsQuantity.TRACK,
            circular=True,
        )

    def get(self, idx: int) -> float:
        return self.waypoint_result(idx).current.track_error_deg


@dataclass(frozen=True)
class WaypointAltDiffFt(WaypointResultObsField):
    """Ownship altitude minus waypoint query altitude."""

    queryable_spec = QueryableFieldSpec(
        kind=QueryableKind.WAYPOINT,
        path="current.alt_diff_ft",
        label="Altitude error",
        description="Ownship altitude minus waypoint altitude.",
        requirements=(QueryableFieldRequirement.ALTITUDE,),
    )

    @property
    def meta(self) -> ObsMeta:
        return ObsMeta(
            f"{self.query_name.lower()}_alt_diff_ft",
            Unit.FT,
            ObsQuantity.ALTITUDE,
        )

    def get(self, idx: int) -> float:
        return self.waypoint_result(idx).current.alt_diff_ft


@dataclass(frozen=True)
class WaypointRouteIndex(WaypointResultObsField):
    """Index of this waypoint in the aircraft's BlueSky route, or -1."""

    queryable_spec = QueryableFieldSpec(
        kind=QueryableKind.WAYPOINT,
        path="route.index",
        label="Route index",
        description="Index of this waypoint in the aircraft route, or -1.",
        requirements=(QueryableFieldRequirement.ROUTE,),
    )

    low: float = -1.0
    high: float = 128.0

    @property
    def meta(self) -> ObsMeta:
        return ObsMeta(
            f"{self.query_name.lower()}_route_index",
            Unit.UNITLESS,
            ObsQuantity.PHASE,
        )

    def get(self, idx: int) -> float:
        route_index = self.waypoint_result(idx).route.index
        return -1.0 if route_index is None else float(route_index)


@dataclass(frozen=True)
class _WaypointRouteFlag(WaypointResultObsField):
    flag_name: str = "active"

    @property
    def meta(self) -> ObsMeta:
        return ObsMeta(
            f"{self.query_name.lower()}_route_{self.flag_name}",
            Unit.SWITCH,
            ObsQuantity.PHASE,
        )

    def get(self, idx: int) -> float:
        return 1.0 if bool(getattr(self.waypoint_result(idx).route, self.flag_name)) else 0.0


@dataclass(frozen=True)
class WaypointRouteActive(_WaypointRouteFlag):
    """Whether this waypoint is the aircraft's active route waypoint."""

    queryable_spec = QueryableFieldSpec(
        kind=QueryableKind.WAYPOINT,
        path="route.active",
        label="Route active",
        description="Whether this waypoint is the aircraft's active route waypoint.",
        requirements=(QueryableFieldRequirement.ROUTE,),
    )

    flag_name: str = "active"


@dataclass(frozen=True)
class WaypointRouteReached(_WaypointRouteFlag):
    """Whether this waypoint has been reached in the aircraft's route."""

    queryable_spec = QueryableFieldSpec(
        kind=QueryableKind.WAYPOINT,
        path="route.reached",
        label="Route reached",
        description="Whether this waypoint has been reached in the aircraft route.",
        requirements=(QueryableFieldRequirement.ROUTE,),
    )

    flag_name: str = "reached"


@dataclass(frozen=True)
class WaypointRouteFuture(_WaypointRouteFlag):
    """Whether this waypoint is still ahead in the aircraft's route."""

    queryable_spec = QueryableFieldSpec(
        kind=QueryableKind.WAYPOINT,
        path="route.future",
        label="Route future",
        description="Whether this waypoint is still ahead in the aircraft route.",
        requirements=(QueryableFieldRequirement.ROUTE,),
    )

    flag_name: str = "future"


@dataclass(frozen=True)
class WaypointSatisfied(WaypointResultObsField):
    """Whether ownship currently satisfies waypoint tolerances."""

    queryable_spec = QueryableFieldSpec(
        kind=QueryableKind.WAYPOINT,
        path="current.satisfied",
        label="Satisfied",
        description="Whether ownship currently satisfies waypoint tolerances.",
        requirements=(QueryableFieldRequirement.TOLERANCE,),
    )

    @property
    def meta(self) -> ObsMeta:
        return ObsMeta(
            f"{self.query_name.lower()}_satisfied",
            Unit.SWITCH,
            ObsQuantity.PHASE,
        )

    def get(self, idx: int) -> float:
        return 1.0 if self.waypoint_result(idx).current.satisfied else 0.0


@dataclass(frozen=True)
class WaypointSatisfiedDuringStep(WaypointResultObsField):
    """Whether ownship satisfied waypoint tolerances during this step."""

    queryable_spec = QueryableFieldSpec(
        kind=QueryableKind.WAYPOINT,
        path="step.satisfied",
        label="Satisfied during step",
        description="Whether ownship satisfied waypoint tolerances during this env step.",
        requirements=(
            QueryableFieldRequirement.TOLERANCE,
            QueryableFieldRequirement.STEP,
        ),
    )

    @property
    def meta(self) -> ObsMeta:
        return ObsMeta(
            f"{self.query_name.lower()}_satisfied_during_step",
            Unit.SWITCH,
            ObsQuantity.PHASE,
        )

    def get(self, idx: int) -> float:
        return 1.0 if bool(self.waypoint_result(idx).step.satisfied) else 0.0


@dataclass(frozen=True)
class WaypointSatisfiedTimeTotalS(WaypointResultObsField):
    """Total seconds ownship has spent satisfying waypoint tolerances."""

    queryable_spec = QueryableFieldSpec(
        kind=QueryableKind.WAYPOINT,
        path="time.total_s",
        label="Time satisfied",
        description="Total simulated seconds ownship has spent satisfying waypoint tolerances.",
        requirements=(
            QueryableFieldRequirement.TOLERANCE,
            QueryableFieldRequirement.TIME,
        ),
    )

    high: float = 3600.0

    @property
    def meta(self) -> ObsMeta:
        return ObsMeta(
            f"{self.query_name.lower()}_satisfied_time_total_s",
            Unit.S,
            ObsQuantity.TIME,
        )

    def get(self, idx: int) -> float:
        result = self.waypoint_result(idx)
        time = getattr(result, "time", None)
        return float(getattr(time, "total_s", 0.0))


@dataclass(frozen=True)
class WaypointMinDistanceNm(WaypointResultObsField):
    """Minimum waypoint distance reached during this step."""

    queryable_spec = QueryableFieldSpec(
        kind=QueryableKind.WAYPOINT,
        path="step.min_distance_nm",
        label="Minimum distance",
        description="Minimum waypoint distance reached during this env step.",
        requirements=(QueryableFieldRequirement.STEP,),
    )

    @property
    def meta(self) -> ObsMeta:
        return ObsMeta(
            f"{self.query_name.lower()}_min_distance_nm",
            Unit.NM,
            ObsQuantity.DISTANCE,
        )

    def get(self, idx: int) -> float:
        result = self.waypoint_result(idx)
        step = getattr(result, "step", None)
        return float(getattr(step, "min_distance_nm", result.current.distance_nm))


@dataclass(frozen=True)
class ActiveWaypointAvailable(ActiveWaypointObsField):
    """Whether an active or future waypoint query result is available."""

    queryable_spec = QueryableFieldSpec(
        kind=QueryableKind.WAYPOINT,
        path="route.active|route.future",
        label="Active waypoint available",
        description="Whether any selected waypoint is active or still ahead in the route.",
        requirements=(QueryableFieldRequirement.ROUTE,),
        cardinality=QueryableFieldCardinality.ACTIVE,
    )

    @property
    def meta(self) -> ObsMeta:
        return ObsMeta("active_waypoint_available", Unit.SWITCH, ObsQuantity.PHASE)

    def get(self, idx: int) -> float:
        return 1.0 if self.active_waypoint(idx) is not None else 0.0


@dataclass(frozen=True)
class ActiveWaypointRouteIndex(ActiveWaypointObsField):
    """Route index of the active waypoint query result, or -1."""

    queryable_spec = QueryableFieldSpec(
        kind=QueryableKind.WAYPOINT,
        path="route.index",
        label="Active route index",
        description="Route index of the active waypoint query result, or -1.",
        requirements=(QueryableFieldRequirement.ROUTE,),
        cardinality=QueryableFieldCardinality.ACTIVE,
    )

    low: float = -1.0
    high: float = 128.0

    @property
    def meta(self) -> ObsMeta:
        return ObsMeta("active_waypoint_route_index", Unit.UNITLESS, ObsQuantity.PHASE)

    def get(self, idx: int) -> float:
        active = self.active_waypoint(idx)
        if active is None or active[1].route.index is None:
            return -1.0
        return float(active[1].route.index)


@dataclass(frozen=True)
class ActiveWaypointOneHot(ActiveWaypointObsField):
    """One-hot identity of the active waypoint query result."""

    queryable_spec = QueryableFieldSpec(
        kind=QueryableKind.WAYPOINT,
        path="route.active",
        label="Active waypoint one-hot",
        description="One-hot identity of the active waypoint among selected queryables.",
        requirements=(QueryableFieldRequirement.ROUTE,),
        cardinality=QueryableFieldCardinality.ACTIVE,
        allow_empty_selection=False,
    )

    @property
    def meta(self) -> ObsMeta:
        return ObsMeta("active_waypoint_one_hot", Unit.UNITLESS, ObsQuantity.PHASE)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.query_names:
            raise ValueError("ActiveWaypointOneHot requires explicit query_names")

    def output_size(self) -> int:
        return len(self.query_names)

    def bounds(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        del idx
        size = self.output_size()
        return (
            np.zeros(size, dtype=np.float32),
            np.ones(size, dtype=np.float32),
        )

    def get(self, idx: int) -> np.ndarray:
        values = np.zeros(self.output_size(), dtype=np.float32)
        active = self.active_waypoint(idx)
        if active is None:
            return values
        try:
            values[self.query_names.index(active[0])] = 1.0
        except ValueError:
            return values
        return values


@dataclass(frozen=True)
class ActiveWaypointDistanceNm(ActiveWaypointObsField):
    """Distance from ownship to the active waypoint query result."""

    queryable_spec = QueryableFieldSpec(
        kind=QueryableKind.WAYPOINT,
        path="current.distance_nm",
        label="Active distance",
        description="Distance from ownship to the active waypoint in nautical miles.",
        cardinality=QueryableFieldCardinality.ACTIVE,
    )

    @property
    def meta(self) -> ObsMeta:
        return ObsMeta("active_waypoint_distance_nm", Unit.NM, ObsQuantity.DISTANCE)

    def get(self, idx: int) -> float:
        active = self.active_waypoint(idx)
        return 0.0 if active is None else active[1].current.distance_nm


@dataclass(frozen=True)
class ActiveWaypointBearingDeg(ActiveWaypointObsField):
    """Bearing from ownship to the active waypoint query result."""

    queryable_spec = QueryableFieldSpec(
        kind=QueryableKind.WAYPOINT,
        path="current.bearing_deg",
        label="Active bearing",
        description="True bearing from ownship to the active waypoint.",
        cardinality=QueryableFieldCardinality.ACTIVE,
    )

    low: float = 0.0
    high: float = 360.0

    @property
    def meta(self) -> ObsMeta:
        return ObsMeta(
            "active_waypoint_bearing_deg",
            Unit.DEG,
            ObsQuantity.BEARING,
            circular=True,
        )

    def get(self, idx: int) -> float:
        active = self.active_waypoint(idx)
        return 0.0 if active is None else active[1].current.bearing_deg


@dataclass(frozen=True)
class ActiveWaypointAltDiffFt(ActiveWaypointObsField):
    """Ownship altitude minus active waypoint query altitude."""

    queryable_spec = QueryableFieldSpec(
        kind=QueryableKind.WAYPOINT,
        path="current.alt_diff_ft",
        label="Active altitude error",
        description="Ownship altitude minus active waypoint altitude.",
        requirements=(QueryableFieldRequirement.ALTITUDE,),
        cardinality=QueryableFieldCardinality.ACTIVE,
    )

    @property
    def meta(self) -> ObsMeta:
        return ObsMeta("active_waypoint_alt_diff_ft", Unit.FT, ObsQuantity.ALTITUDE)

    def get(self, idx: int) -> float:
        active = self.active_waypoint(idx)
        return 0.0 if active is None else active[1].current.alt_diff_ft


@dataclass(frozen=True)
class ActiveWaypointTrackErrorDeg(ActiveWaypointObsField):
    """Signed track error toward the active waypoint query result."""

    queryable_spec = QueryableFieldSpec(
        kind=QueryableKind.WAYPOINT,
        path="current.track_error_deg",
        label="Active track error",
        description="Signed ownship track error toward the active waypoint.",
        cardinality=QueryableFieldCardinality.ACTIVE,
    )

    low: float = -180.0
    high: float = 180.0

    @property
    def meta(self) -> ObsMeta:
        return ObsMeta(
            "active_waypoint_track_error_deg",
            Unit.DEG,
            ObsQuantity.TRACK,
            circular=True,
        )

    def get(self, idx: int) -> float:
        active = self.active_waypoint(idx)
        if active is None:
            return 0.0
        return active[1].current.track_error_deg
