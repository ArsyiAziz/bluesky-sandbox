"""The episode contract: what one sampled episode contains, and who can
produce one.

Declarations only. :class:`Scenario` is the protocol a sampler satisfies;
:class:`RandomizedScenario` in :mod:`.randomized` is the implementation the
designer generates against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from bluesky_sandbox._renames import renamed


@renamed(bounds="shapes")
@dataclass(frozen=True)
class EpisodeSpec:
    """Concrete resources used by one sampled episode.

    ``shapes`` is the design's shapes - areas and points - by name, in this
    episode's frame - rotated and resampled with everything that references
    them - for code to read their geometry (``env.episode_shapes``,
    ``ctx.shape(name)``). ``bounds``, its older name, still works."""

    airspace_bounds: Any
    spawn: Any
    queryables: dict[str, Any]
    max_aircraft: int
    data: Any = None
    shapes: dict[str, Any] = field(default_factory=dict)


class Scenario(Protocol):
    """Episode sampler that exposes a stable schema-support episode."""

    def sample(self, rng: np.random.Generator) -> EpisodeSpec:
        """Return concrete resources for the current episode."""
        ...

    def support(self) -> EpisodeSpec:
        """Return resources whose bounds cover all sampled episodes."""
        ...
