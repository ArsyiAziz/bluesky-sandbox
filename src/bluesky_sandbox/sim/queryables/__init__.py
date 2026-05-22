"""Queryable resources evaluated against BlueSky traffic indices.

Split into :mod:`.base` (the protocol), :mod:`.regions` and
:mod:`.waypoints`; everything is re-exported here, so the import path
is unchanged.
"""

from __future__ import annotations

from .base import (
    Queryable,
)
from .base import (
    # Private, but imported by designer.nav - re-exported so the flat
    # module's import surface is preserved exactly.
    _ensure_navdb_loaded as _ensure_navdb_loaded,
)
from .regions import (
    QueryRegion,
    RegionCurrent,
    RegionResult,
    RegionStep,
    UnavailableRegionStep,
)
from .waypoints import (
    UnavailableWaypointStep,
    Waypoint,
    WaypointCurrent,
    WaypointResult,
    WaypointRoute,
    WaypointStep,
    WaypointTarget,
)

__all__ = [
    "QueryRegion",
    "Queryable",
    "RegionCurrent",
    "RegionResult",
    "RegionStep",
    "UnavailableRegionStep",
    "UnavailableWaypointStep",
    "Waypoint",
    "WaypointCurrent",
    "WaypointResult",
    "WaypointRoute",
    "WaypointStep",
    "WaypointTarget",
]
