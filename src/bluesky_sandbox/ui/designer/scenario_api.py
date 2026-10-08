"""What a design's scenario code works with: the hooks' typed signatures, so
the code editor completes and checks them.

A scenario hook's body has :data:`.setup_code.SCENARIO_API` in scope - the
library names the generated ``scenario.py`` imports - in the designer and in
the generated package alike.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, TypedDict

import numpy as np

from bluesky_sandbox.interface.task import DesignKeys
from bluesky_sandbox.sim.bounds import Bounds, RegionBounds
from bluesky_sandbox.sim.queryables import Queryable
from bluesky_sandbox.sim.spawn import SpawnConfig


class EpisodeGeometry(TypedDict, total=False):
    """One episode's geometry, as the structured design built it - what
    ``episode_geometry`` receives and returns (changed or not)."""

    #: The airspace's shape, or None.
    airspace_bounds: RegionBounds | None
    #: The spawn config: its regions, routes, separation settings.
    spawn: SpawnConfig
    #: The queryables, by name.
    queryables: Annotated[Mapping[str, Queryable], DesignKeys("queryable")]
    #: Waypoints whose position is drawn from a region each episode, by name.
    sampled_waypoints: Mapping[str, Bounds]
    #: The design's shapes - areas and points - by name (``"bounds"``, their
    #: older name, still reads them).
    shapes: Annotated[Mapping[str, RegionBounds], DesignKeys("shapes")]


def episode_geometry(geometry: EpisodeGeometry, rng: np.random.Generator) -> EpisodeGeometry:
    """Post-process this episode's geometry; return it (changed or not)."""
    return geometry


#: Each scenario hook's typed signature, by name (see ``spec.SCENARIO_HOOKS``).
SIGNATURES = {"episode_geometry": episode_geometry}

