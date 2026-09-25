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
from .altitude import (
    AltDeltaFt,
    AltDeltaM,
    AltFt,
    AltM,
    ApAltDeltaFt,
    ApAltDeltaM,
)
from .autopilot import (
    AutopilotLnav,
    AutopilotLnavVnav,
    AutopilotVnav,
)
from .comm import (
    CommBroadcast,
)
from .heading import (
    ApHdgDeltaDeg,
    HdgDeg,
    HdgDeltaDeg,
)
from .route import (
    ActiveRouteWaypointAltDeltaFt,
    ActiveRouteWaypointHdgDeltaDeg,
    ActiveRouteWaypointSpdDeltaCrossover,
    ActiveRouteWaypointSpdDeltaKts,
)
from .speed import (
    ApSpdDeltaCrossover,
    ApSpdDeltaKts,
    SpdDeltaKts,
    SpdDeltaMs,
    SpdKts,
    SpdMs,
)

__all__ = [
    "ActionField",
    "ActionKind",
    "ActionMeta",
    "ActionMode",
    "ControlAxis",
    "SwitchActionMixin",
    "Unit",
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
]
