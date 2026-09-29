"""Small data types shared by the environment and its services.

They live here rather than in :mod:`.base_environment` because
:mod:`.spawning` constructs most of them, and the environment imports
``SpawnGenerator`` - importing them back from the environment would make that
a cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, TypeAlias

from bluesky_sandbox.sim.queryables import WaypointTarget

Callsign: TypeAlias = str


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
    #: Each step's arrival-time slack, seconds or a distribution of them;
    #: ``None`` for a step with no target arrival time.
    arrival_slacks: tuple = ()


class AircraftControlState(Enum):
    CONTROLLED = "controlled"
    BACKGROUND = "background"
    DELETE = "delete"


class SpawnProgress(NamedTuple):
    spawned: int
    scheduled: int


class SpawnRecord(NamedTuple):
    """One aircraft as it was created: who, when, where, and how it flew off."""

    callsign: Callsign
    actype: str
    time_s: float
    lat_deg: float
    lon_deg: float
    alt_ft: float
    hdg_deg: float
    cas_kts: float
    gs_kts: float
    mach: float
    controlled: bool
    region_index: int
    route: tuple[str, ...] | None
    #: The route's waypoints as resolved for this aircraft, constraints and all.
    targets: tuple[WaypointTarget, ...] = ()


AircraftControlStates: TypeAlias = dict[Callsign, AircraftControlState]
