"""Spawn configuration: where aircraft appear and what route they fly.

Split into :mod:`~bluesky_sandbox.sim.spawn.routes` (the route-spec grammar)
and :mod:`~bluesky_sandbox.sim.spawn.regions` (spawn volumes and the episode
spawn queue); everything is re-exported
here, so the import path is unchanged.
"""

from __future__ import annotations

from .regions import (
    SpawnConfig,
    SpawnRegion,
)
from .routes import (
    RouteSpec,
    RouteStep,
    expand_route_paths,
    resolve_route,
    route_step_name,
    route_step_names,
    sample_route_path,
)

__all__ = [
    "RouteSpec",
    "RouteStep",
    "SpawnConfig",
    "SpawnRegion",
    "expand_route_paths",
    "resolve_route",
    "route_step_name",
    "route_step_names",
    "sample_route_path",
]
