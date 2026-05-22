"""The episode contract: what one sampled episode contains, and who can
produce one.

Declarations only. :class:`Scenario` is the protocol a sampler satisfies;
:class:`RandomizedScenario` in :mod:`.randomized` is the implementation the
designer generates against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class EpisodeSpec:
    """Concrete resources used by one sampled episode."""

    airspace_bounds: Any
    spawn: Any
    queryables: dict[str, Any]
    max_aircraft: int
    data: Any = None


class Scenario(Protocol):
    """Episode sampler that exposes a stable schema-support episode."""

    def sample(self, rng: np.random.Generator) -> EpisodeSpec:
        """Return concrete resources for the current episode."""
        ...

    def support(self) -> EpisodeSpec:
        """Return resources whose bounds cover all sampled episodes."""
        ...
