from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class ParamDistribution(Protocol):
    """scipy-compatible interface for continuous scalar parameter distributions.

    Any frozen ``scipy.stats`` continuous distribution satisfies this protocol.
    The environment calls ``dist.rvs(random_state=rng)`` with its own seeded
    RNG, so ``env.reset(seed=N)`` controls all parameter sampling automatically.
    ``dist.support()`` is used to derive observation/action space bounds.

    ::

        from scipy.stats import norm, uniform, truncnorm

        params = {
            "alt_ft": uniform(loc=5_000, scale=10_000),  # uniform over [5000, 15000]
            "spd_kts": truncnorm(a=-2, b=2, loc=250, scale=30),  # clipped normal
        }
    """
    def rvs(self, random_state: np.random.Generator) -> float: ...
    def support(self) -> tuple[float, float]: ...


class CountDistribution(Protocol):
    """scipy-compatible interface for integer count distributions.

    Any frozen ``scipy.stats`` integer distribution satisfies this protocol.
    The environment calls ``dist.rvs(random_state=rng)`` with its own seeded
    RNG, so ``env.reset(seed=N)`` controls all count sampling automatically.
    ``dist.support()`` is used to derive schema bounds when a count can vary
    between resets.

    ::

        from scipy.stats import randint, poisson, nbinom

        n_aircraft = randint(2, 7)       # uniform over {2, 3, 4, 5, 6}
        n_aircraft = poisson(mu=4)       # Poisson with mean 4
        n_aircraft = nbinom(n=3, p=0.5)  # negative binomial
    """
    def rvs(self, random_state: np.random.Generator) -> int: ...
    def support(self) -> tuple[int, int]: ...


@dataclass(frozen=True)
class Bounded:
    """Restrict a frozen ``scipy.stats`` distribution to a finite ``[lo, hi]``.

    Wraps any frozen scipy distribution so it exposes a finite ``support()`` and
    only ever samples within ``[lo, hi]`` - letting otherwise-unbounded shapes
    (``poisson``, ``nbinom``, ...) be used where a hard finite cap is required.
    ``SpawnRegion.n_aircraft``, for instance, needs a bounded support because its
    max sizes the padded observation space, so a bare ``poisson`` is rejected;
    ``Bounded(poisson(50), 1, 100)`` is accepted.

    Satisfies :class:`CountDistribution` / :class:`ParamDistribution` (exposes
    ``rvs`` and ``support``), so it drops into any ``scipy.stats``-typed field.

    ::

        from scipy.stats import poisson, nbinom
        n_aircraft = Bounded(poisson(mu=40), 1, 100)              # truncated
        n_aircraft = Bounded(nbinom(5, 0.1), 1, 100, mode="clip")  # clamped

    Parameters
    ----------
    dist:
        A frozen ``scipy.stats`` distribution (the value returned by e.g.
        ``poisson(mu=40)``).
    lo, hi:
        Inclusive finite bounds on the sampled value.
    mode:
        ``"truncate"`` (default) draws from the distribution *renormalised* onto
        ``[lo, hi]`` via inverse-CDF, so no probability mass piles at the edges.
        ``"clip"`` draws from the base distribution and clamps into range
        (simpler; mass accumulates at the bounds - fine when the cap sits in the
        far tail).
    """
    dist: Any
    lo: float
    hi: float
    mode: str = "truncate"

    def __post_init__(self) -> None:
        lo, hi = float(self.lo), float(self.hi)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(f"Bounded requires finite [lo, hi], got {(lo, hi)!r}.")
        if lo > hi:
            raise ValueError(f"Bounded requires lo <= hi, got {(lo, hi)!r}.")
        if self.mode not in ("truncate", "clip"):
            raise ValueError(
                f"Bounded.mode must be 'truncate' or 'clip', got {self.mode!r}."
            )
        if not (hasattr(self.dist, "rvs") and hasattr(self.dist, "cdf")):
            raise ValueError("Bounded.dist must be a frozen scipy.stats distribution.")

    def support(self) -> tuple[float, float]:
        return (float(self.lo), float(self.hi))

    def mean(self) -> float:
        # Approximate representative value for the schema-support frame; clamping
        # the base mean is enough to keep that frame stable and in-range.
        return float(np.clip(self.dist.mean(), self.lo, self.hi))

    def median(self) -> float:
        return float(np.clip(self.dist.median(), self.lo, self.hi))

    def rvs(self, random_state: np.random.Generator | None = None):
        if self.mode == "clip":
            return float(
                np.clip(self.dist.rvs(random_state=random_state), self.lo, self.hi)
            )
        # Inverse-CDF truncation. For a discrete base, ``cdf(lo - 1)`` keeps the
        # lower integer endpoint ``lo`` itself reachable.
        discrete = hasattr(getattr(self.dist, "dist", None), "pmf")
        c_lo = self.dist.cdf(self.lo - 1) if discrete else self.dist.cdf(self.lo)
        c_hi = self.dist.cdf(self.hi)
        if not (c_hi > c_lo):
            # Bounds enclose ~no probability mass; fall back to clamping so we
            # still return an in-range value instead of a NaN from ppf.
            return float(
                np.clip(self.dist.rvs(random_state=random_state), self.lo, self.hi)
            )
        src = random_state if random_state is not None else np.random.default_rng()
        return float(self.dist.ppf(src.uniform(c_lo, c_hi)))


class TypeDistribution(Protocol):
    """scipy-compatible interface for aircraft type distributions.

    The environment calls ``dist.rvs(random_state=rng)`` with its own seeded
    RNG.  :class:`Categorical` satisfies this protocol over string labels.

    ::

        aircraft_type = Categorical({"A320": 3.0, "B738": 1.0})
    """
    def rvs(self, random_state: np.random.Generator) -> str: ...


@dataclass
class Categorical:
    """Sample an aircraft type according to relative weights.

    Satisfies :class:`TypeDistribution` using scipy-style ``rvs`` interface.

    Parameters
    ----------
    weights:
        Mapping of ICAO type -> relative weight, e.g.
        ``{"A320": 3.0, "B738": 1.0}``.  Weights do not need to sum to 1
        but must all be strictly positive.
    """
    weights: dict[str, float]

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("Categorical weights must not be empty.")
        negative = [k for k, v in self.weights.items() if v <= 0]
        if negative:
            raise ValueError(
                f"Categorical weights must be strictly positive; got non-positive values for: {negative}"
            )

    def rvs(self, random_state: np.random.Generator) -> str:
        keys  = list(self.weights)
        probs = np.array(list(self.weights.values()), dtype=float)
        probs /= probs.sum()
        return keys[int(random_state.choice(len(keys), p=probs))]


def sample_scalar(value: Any, rng: np.random.Generator) -> float:
    """Draw a scalar from a fixed number, a ``(low, high)`` range, or a dist."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return float(rng.uniform(value[0], value[1]))
    if hasattr(value, "rvs"):
        return float(value.rvs(random_state=rng))
    raise ValueError(f"cannot sample a scalar from {value!r}")
