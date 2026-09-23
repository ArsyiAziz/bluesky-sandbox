"""Wind: one definition of which way the air is moving.

Everything that needs wind - BlueSky's own wind field, spawn conflict
prediction, any future observation field - asks a :class:`WindField` rather
than re-deriving a vector from ``wind_dir_deg`` / ``wind_kts``. That derivation
used to be written out in two places (the runtime's stack command and the spawn
clearance check), which is exactly how a sign convention drifts: ``dir_deg`` is
the direction the wind blows *from*, so every consumer has to remember to
negate.

Vectors are ``(north, east)`` in m/s and point the way the air **moves** - the
opposite of the aviation ``dir_deg`` convention - so they add directly to a
ground-referenced velocity.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from bluesky.tools.aero import kts

__all__ = ["UniformWind", "WindField", "wind_from_config"]


class WindField(ABC):
    """A wind field over the airspace."""

    @abstractmethod
    def mean_ne_ms(self) -> tuple[float, float]:
        """Steady ``(north, east)`` component in m/s, excluding any gust.

        This is the component to use for anything predictive: gusts are
        zero-mean and decorrelate within a step, so extrapolating one forward
        asserts a persistence it does not have.
        """

    def current_ne_ms(self) -> tuple[float, float]:
        """Total wind right now, m/s - mean plus whatever gust is live."""
        return self.mean_ne_ms()

    def advance(self, dt_s: float, rng: np.random.Generator) -> None:
        """Evolve time-varying components one env step. No-op when steady."""

    def reset(self) -> None:
        """Drop per-episode state (a new episode starts from the mean)."""

    @property
    def is_still(self) -> bool:
        """True when this field never moves any air, so callers can skip work."""
        return False

    @property
    def is_dynamic(self) -> bool:
        """True when the vector changes between steps.

        A steady field only needs pushing into the simulator once per episode;
        re-applying it every step would be wasted work.
        """
        return False


@dataclass
class UniformWind(WindField):
    """Spatially uniform wind with an optional Ornstein-Uhlenbeck gust.

    ``dir_deg`` is aviation-standard: the direction the wind blows FROM,
    degrees true clockwise from north (270 = westerly, pushing aircraft east).
    ``turbulence_kts`` is the stationary RMS of a zero-mean gust that
    decorrelates over ``gust_tau_s`` seconds; 0 leaves the field steady.
    """

    dir_deg: float = 270.0
    speed_kts: float = 0.0
    turbulence_kts: float = 0.0
    gust_tau_s: float = 30.0

    def __post_init__(self) -> None:
        self._gust_ne: tuple[float, float] = (0.0, 0.0)

    @property
    def is_still(self) -> bool:
        return float(self.speed_kts) <= 0.0 and float(self.turbulence_kts) <= 0.0

    @property
    def is_dynamic(self) -> bool:
        return float(self.turbulence_kts) > 0.0

    def mean_ne_ms(self) -> tuple[float, float]:
        speed_ms = float(self.speed_kts) * kts
        rad = math.radians(float(self.dir_deg))
        # Negated: ``dir_deg`` says where the wind comes from, not where it goes.
        return -speed_ms * math.cos(rad), -speed_ms * math.sin(rad)

    def current_ne_ms(self) -> tuple[float, float]:
        mean_n, mean_e = self.mean_ne_ms()
        gust_n, gust_e = self._gust_ne
        return mean_n + gust_n, mean_e + gust_e

    def advance(self, dt_s: float, rng: np.random.Generator) -> None:
        turb = float(self.turbulence_kts)
        if turb <= 0.0 or dt_s <= 0.0:
            return
        # Ornstein-Uhlenbeck: mean-reverting to zero with correlation time
        # ``gust_tau_s`` and stationary RMS ``turbulence_kts``. ``step_sigma``
        # is chosen so the stationary variance is independent of ``dt_s``.
        tau = max(float(self.gust_tau_s), 1e-3)
        decay = math.exp(-dt_s / tau)
        step_sigma = turb * kts * math.sqrt(max(1.0 - decay * decay, 0.0))
        gust_n, gust_e = self._gust_ne
        self._gust_ne = (
            decay * gust_n + step_sigma * float(rng.standard_normal()),
            decay * gust_e + step_sigma * float(rng.standard_normal()),
        )

    def reset(self) -> None:
        self._gust_ne = (0.0, 0.0)


def wind_from_config(config) -> UniformWind:
    """Build the wind field an ``EnvConfig``'s scalar wind settings describe.

    The config keeps the four scalars rather than a ``WindField`` so the
    designer's spec round-trip and every existing scenario keep working; this
    is the one place that turns them into a field.
    """
    return UniformWind(
        dir_deg=float(getattr(config, "wind_dir_deg", 270.0)),
        speed_kts=float(getattr(config, "wind_kts", 0.0)),
        turbulence_kts=float(getattr(config, "turbulence_kts", 0.0)),
        gust_tau_s=float(getattr(config, "gust_tau_s", 30.0)),
    )
