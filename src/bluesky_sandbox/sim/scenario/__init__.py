"""Scenario sampling: the episode contract and the randomized sampler.

Split into :mod:`.base` (``EpisodeSpec`` and the ``Scenario`` protocol - what
used to be ``bluesky_sandbox.sim.sampling``) and :mod:`.randomized` (the concrete
sampler). Both are re-exported here, so ``bluesky_sandbox.sim.scenario`` is
unchanged and the two names no longer live in confusingly similar modules.
"""

from __future__ import annotations

from .base import (
    EpisodeSpec,
    Scenario,
)
from .randomized import (
    RandomizedScenario,
    GeometryDict,
    RegionParamSampler,
)

__all__ = [
    "EpisodeSpec",
    "RandomizedScenario",
    "GeometryDict",
    "RegionParamSampler",
    "Scenario",
]
