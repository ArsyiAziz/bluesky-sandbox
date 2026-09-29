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
from .autopilot import (
    ApAltDeltaFt,
    ApAltDeltaM,
    ApHdgDeltaDeg,
    ApSpdDeltaCrossover,
    ApSpdDeltaKts,
    AutopilotLnav,
    AutopilotLnavVnav,
    AutopilotVnav,
    ResumeOwnNav,
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
from .mask import (
    ActionMask,
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
    "ResumeOwnNav",
    "CommBroadcast",
    "ActionMask",
]
