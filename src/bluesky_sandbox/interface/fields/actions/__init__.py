"""The built-in action fields, one module per kind of command.

Every action is importable from here, whichever module defines it.
"""

from __future__ import annotations

from ..base import (
    ActionField,
    ActionKind,
    ActionMeta,
    ActionMode,
    ControlAxis,
    SwitchActionMixin,
    Unit,
)
from ..grid import Grid
from .crossover import Crossover, MachRegime
from .autopilot import (
    ApAltDeltaFt,
    ApAltDeltaM,
    ApHdgDeltaDeg,
    ApSpdDeltaCrossover,
    ApSpdDeltaKts,
    AutopilotLnav,
    AutopilotLnavVnav,
    AutopilotVnav,
)
from .comm import (
    CommBroadcast,
)
from .kinematics import (
    AltDeltaFt,
    AltDeltaM,
    AltFt,
    AltM,
    HdgDeg,
    HdgDeltaDeg,
    SpdDeltaKts,
    SpdDeltaMs,
    SpdKts,
    SpdMs,
)
from .clearance import (
    Clearance,
)
from .mask import (
    ActionMask,
    ClearanceDuration,
)
from .route import (
    ActiveRouteWaypointAltDeltaFt,
    ActiveRouteWaypointHdgDeltaDeg,
    ActiveRouteWaypointSpdDeltaCrossover,
    ActiveRouteWaypointSpdDeltaKts,
)

__all__ = [
    "ActionField",
    "ActionKind",
    "ActionMeta",
    "ActionMode",
    "ControlAxis",
    "SwitchActionMixin",
    "Unit",
    "Grid",
    "Crossover",
    "MachRegime",
    "HdgDeg",
    "HdgDeltaDeg",
    "ApHdgDeltaDeg",
    "SpdKts",
    "SpdMs",
    "SpdDeltaKts",
    "ApSpdDeltaKts",
    "SpdDeltaMs",
    "ApSpdDeltaCrossover",
    "AltFt",
    "AltM",
    "AltDeltaFt",
    "ApAltDeltaFt",
    "AltDeltaM",
    "ApAltDeltaM",
    "ActiveRouteWaypointHdgDeltaDeg",
    "ActiveRouteWaypointAltDeltaFt",
    "ActiveRouteWaypointSpdDeltaKts",
    "ActiveRouteWaypointSpdDeltaCrossover",
    "AutopilotLnav",
    "AutopilotVnav",
    "AutopilotLnavVnav",
    "CommBroadcast",
    "ActionMask",
    "ClearanceDuration",
    "Clearance",
]
