"""Sampling primitives: the distribution protocols and scalar draws that
domain randomisation is built from.

Layer-0 by design - spawn regions, scenarios and the designer all draw from
here, so nothing in this package may import them back.
"""

from __future__ import annotations

from .distributions import (
    Bounded,
    Categorical,
    CountDistribution,
    ParamDistribution,
    TypeDistribution,
    sample_scalar,
)

__all__ = [
    "Bounded",
    "Categorical",
    "CountDistribution",
    "ParamDistribution",
    "TypeDistribution",
    "sample_scalar",
]
