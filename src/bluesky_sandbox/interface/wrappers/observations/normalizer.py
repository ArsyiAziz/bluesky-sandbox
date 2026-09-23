"""Normalizer strategies for field-level scaling.

There is **one** :class:`Normalizer` hierarchy. The same strategies scale
observation fields and denormalize action fields. Relative intruder
features are expressed as ``PairObsField`` objects, for example
``obs.AltFt().relative_to_own(normalizer=SymmetricNormalizer())``; normalizers
only scale the value produced by the field.

Each subclass declares:

  * ``normalize(field, value, idx)`` - scale a value for ``field``.
  * ``denormalize(field, value, idx)`` - map external action value(s) back to
    physical units.

Typical field-level usage::

    obs.TrkDeg(normalizer=CircularNormalizer())
    obs.AltFt().relative_to_own(normalizer=SymmetricNormalizer())
    actions.HdgDeltaDeg(-30.0, 30.0, normalizer=SymmetricNormalizer())
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TypeAlias

import numpy as np

from bluesky_sandbox.interface.fields.base import ActionField, ObsField, PairObsField

FieldLike: TypeAlias = ObsField | PairObsField | ActionField


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Normalizer(ABC):
    """Scales one field value and optionally denormalizes actions."""

    #: True when this normalizer encodes an ANGLE as a ``(cos, sin)`` pair whose
    #: decoded value is the direction of that pair, so the pair's magnitude is
    #: not observable downstream. Trainers read this to give the affected action
    #: slots a distribution on the circle instead of an unconstrained 2-D one -
    #: see ``rl.raw_env._circular_action_dims``. Declared on the base so any
    #: custom normalizer answers the question without an isinstance check.
    is_circular: bool = False

    #: Whether this strategy holds its output to :attr:`output_interval`.
    #: Declared on the base so :meth:`_clip` and :meth:`_action_scalar` are
    #: safe for strategies with no notion of clipping.
    clipped: bool = False

    #: The interval this strategy emits, or ``None`` to pass the field's own
    #: bounds through (:class:`RawNormalizer`). One constant drives three
    #: things that have to agree: the Box space built from
    #: :meth:`output_bounds`, the clamp :meth:`normalize` applies when
    #: ``clipped``, and the clamp :meth:`denormalize` applies to an incoming
    #: action. Each strategy used to write all three out separately, and two of
    #: the denormalize clamps were simply missing - so an out-of-range action
    #: commanded past the field's own bound.
    output_interval: tuple[float, float] | None = None

    def _bounds(self, field: FieldLike, idx: int) -> tuple[float, float]:
        """Resolve ``(low, high)`` for a configured field object.

        A field with per-component bounds has no single span to scale against.
        Letting one through produces a value as wide as the bounds array while
        ``output_size`` still reports 1, and that only surfaces later, as a
        ragged row when the observation is concatenated - far from the field
        that caused it. Fail here, naming it.
        """
        low, high = field.bounds(idx)
        if np.ndim(low) or np.ndim(high):
            raise TypeError(
                f"{field.meta.name!r} has per-component bounds, so "
                f"{self.__class__.__name__} has no single span to scale "
                "against. Attach the normalizer to a scalar field, or none."
            )
        return low, high

    def _clip(self, value):
        """Hold a normalized value to :attr:`output_interval`.

        Takes scalars and arrays alike, so :meth:`normalize` and
        :meth:`normalize_many` cannot end up clipping differently.
        """
        if not self.clipped or self.output_interval is None:
            return value
        low, high = self.output_interval
        if isinstance(value, np.ndarray):
            return np.clip(value, low, high)
        return min(max(value, low), high)

    def _action_scalar(self, field: FieldLike, value) -> float:
        """Unwrap a one-element action and hold it to :attr:`output_interval`.

        Clamping here is the Box space contract: the space advertises a range,
        so a value outside it commands the bound and never past it. Policy
        heads without a squashing layer emit out-of-range actions routinely.
        """
        if isinstance(value, Sequence):
            if len(value) != 1:
                raise ValueError(
                    f"{self.__class__.__name__} expected one action value for "
                    f"{field.meta.name!r}, got {len(value)}."
                )
            value = value[0]
        return float(self._clip(float(value)))

    def _span(self, field: FieldLike, idx: int) -> float:
        """Resolve a strictly positive normalisation span for ``field``."""
        lo, hi = self._bounds(field, idx)
        span = hi - lo
        if span <= 0.0:
            raise ValueError(
                f"{field.meta.name!r} bounds must have high > low for "
                f"{self.__class__.__name__}, got ({lo}, {hi})."
            )
        return span

    @abstractmethod
    def normalize(self, field: FieldLike, value: float, idx: int) -> list[float]:
        """Scale ``value`` for ``field``. ``idx`` is the relevant bs_traf index."""

    def normalize_many(self, field: FieldLike, values, idx: int) -> np.ndarray:
        """Vectorized :meth:`normalize` over a 1-D array of raw values that all
        share ``idx`` (e.g. one ownship's intruder batch). Returns an
        ``(n, output_size(field))`` ``float32`` array.

        The default falls back to per-element ``normalize`` (correct for any
        custom normalizer); subclasses override for speed. Bounds are constant
        across the batch, so overrides resolve them once.
        """
        vals = np.asarray(values, dtype=np.float64)
        out = np.empty((vals.shape[0], self.output_size(field)), dtype=np.float32)
        for i in range(vals.shape[0]):
            out[i] = self.normalize(field, float(vals[i]), idx)
        return out

    def denormalize(
        self,
        field: FieldLike,
        value: float | Sequence[float],
        idx: int,
    ) -> float:
        """Map external action value(s) back to the field's physical units."""
        return self._action_scalar(field, value)

    def output_size(self, field: FieldLike) -> int:
        """Number of output floats produced for this field (default 1)."""
        return 1

    def output_bounds(
        self, field: FieldLike,
    ) -> tuple[list[float], list[float]]:
        """Output bounds for ownship-style values."""
        if self.output_interval is not None:
            low, high = self.output_interval
            width = self.output_size(field)
            return [low] * width, [high] * width
        static = _static_or_custom_bounds(field)
        if static is None:
            lo, hi = float("-inf"), float("inf")
        else:
            lo, hi = static
        return [lo], [hi]


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

class RawNormalizer(Normalizer):
    """Pass raw physical values through unchanged."""

    def normalize(self, field, value, idx):
        return [value]

    def normalize_many(self, field, values, idx):
        return np.asarray(values, dtype=np.float64).reshape(-1, 1).astype(np.float32)


class MinMaxNormalizer(Normalizer):
    """Scale to ``[0, 1]`` using the field's bounds.

    Difference pair fields should expose delta bounds directly; with bounds
    ``[-span, +span]`` this maps zero difference to ``0.5``.
    """

    output_interval = (0.0, 1.0)

    def __init__(self, *, clipped: bool = False) -> None:
        self.clipped = clipped

    def normalize(self, field, value, idx):
        lo, _ = self._bounds(field, idx)
        return [self._clip((value - lo) / self._span(field, idx))]

    def normalize_many(self, field, values, idx):
        lo, _ = self._bounds(field, idx)
        out = (np.asarray(values, dtype=np.float64) - lo) / self._span(field, idx)
        return self._clip(out).reshape(-1, 1).astype(np.float32)

    def denormalize(self, field, value, idx):
        lo, _ = self._bounds(field, idx)
        return lo + self._action_scalar(field, value) * self._span(field, idx)


class SymmetricNormalizer(Normalizer):
    """Scale to ``[-1, 1]`` using the field's bounds.

    Difference pair fields should expose delta bounds directly; with bounds
    ``[-span, +span]`` this maps to ``[-1, 1]``.
    """

    output_interval = (-1.0, 1.0)

    def __init__(self, *, clipped: bool = False) -> None:
        self.clipped = clipped

    def normalize(self, field, value, idx):
        lo, _ = self._bounds(field, idx)
        return [self._clip(2.0 * (value - lo) / self._span(field, idx) - 1.0)]

    def normalize_many(self, field, values, idx):
        lo, _ = self._bounds(field, idx)
        span = self._span(field, idx)
        out = 2.0 * (np.asarray(values, dtype=np.float64) - lo) / span - 1.0
        return self._clip(out).reshape(-1, 1).astype(np.float32)

    def denormalize(self, field, value, idx):
        v = self._action_scalar(field, value)
        lo, _ = self._bounds(field, idx)
        return lo + ((v + 1.0) * 0.5) * self._span(field, idx)


class SignedPowerNormalizer(Normalizer):
    """Expo-style nonlinear scaling to ``[-1, 1]``: fine near the centre, full
    authority at the extremes.

    Maps the field's bounds to ``[-1, 1]`` like :class:`SymmetricNormalizer`, but
    passes the value through a signed power curve so most of the range near the
    centre resolves to *small* physical values while ``+/-1`` still reaches the
    full bound. With symmetric delta bounds ``[-b, b]`` the centre is the
    goal-seeking ``0`` action and ``denormalize(a) = sign(a) * |a|**power * b`` -
    the policy gets fine control near ``0`` without capping the maximum maneuver.

    ``power > 1`` sharpens the curve (finer near the centre); ``power == 1``
    recovers the linear :class:`SymmetricNormalizer`.
    """

    def __init__(self, *, power: float = 3.0, clipped: bool = False) -> None:
        if power <= 0.0:
            raise ValueError(
                f"SignedPowerNormalizer power must be > 0, got {power}."
            )
        self.power = float(power)
        self.clipped = clipped

    output_interval = (-1.0, 1.0)

    @staticmethod
    def _signed_pow(x: float, p: float) -> float:
        return math.copysign(abs(x) ** p, x)

    def normalize(self, field, value, idx):
        lo, _ = self._bounds(field, idx)
        u = 2.0 * (value - lo) / self._span(field, idx) - 1.0  # linear -> [-1, 1]
        return [self._clip(self._signed_pow(u, 1.0 / self.power))]  # inverse curve

    def normalize_many(self, field, values, idx):
        lo, _ = self._bounds(field, idx)
        u = 2.0 * (np.asarray(values, dtype=np.float64) - lo) / self._span(field, idx) - 1.0
        a = np.copysign(np.abs(u) ** (1.0 / self.power), u)
        return self._clip(a).reshape(-1, 1).astype(np.float32)

    def denormalize(self, field, value, idx):
        v = self._action_scalar(field, value)
        u = self._signed_pow(v, self.power)                    # compress toward centre
        lo, _ = self._bounds(field, idx)
        return lo + (u + 1.0) * 0.5 * self._span(field, idx)


class PowerNormalizer(Normalizer):
    """One-sided expo-style scaling to ``[0, 1]``: fine near ``low``, full
    range at ``high``.

    The one-sided counterpart of :class:`SignedPowerNormalizer`: maps the
    field's bounds to ``[0, 1]`` like :class:`MinMaxNormalizer`, but through a
    power curve anchored at ``low`` -
    ``normalize(v) = ((v - low) / span) ** (1/power)`` - so resolution
    concentrates near the *lower bound* instead of the interval midpoint.
    Suits magnitude-like quantities (ranges, times-to-go) whose
    decision-relevant band hugs ``low``: with ``power == 2`` a range field
    resolves sqrt-fine near zero while ``1`` still reaches the full bound.

    ``power > 1`` sharpens the curve (finer near ``low``); ``power == 1``
    recovers the linear :class:`MinMaxNormalizer`. Below-``low`` values stay
    monotonic via a signed power (negative output) unless ``clipped``.
    """

    def __init__(self, *, power: float = 2.0, clipped: bool = False) -> None:
        if power <= 0.0:
            raise ValueError(
                f"PowerNormalizer power must be > 0, got {power}."
            )
        self.power = float(power)
        self.clipped = clipped

    output_interval = (0.0, 1.0)

    def normalize(self, field, value, idx):
        lo, _ = self._bounds(field, idx)
        u = (value - lo) / self._span(field, idx)           # linear -> [0, 1]
        a = math.copysign(abs(u) ** (1.0 / self.power), u)  # inverse curve
        return [self._clip(a)]

    def normalize_many(self, field, values, idx):
        lo, _ = self._bounds(field, idx)
        u = (np.asarray(values, dtype=np.float64) - lo) / self._span(field, idx)
        a = np.copysign(np.abs(u) ** (1.0 / self.power), u)
        return self._clip(a).reshape(-1, 1).astype(np.float32)

    def denormalize(self, field, value, idx):
        v = self._action_scalar(field, value)
        u = math.copysign(abs(v) ** self.power, v)          # compress toward low
        lo, _ = self._bounds(field, idx)
        return lo + u * self._span(field, idx)


class CircularNormalizer(Normalizer):
    """Encode an angle (degrees) as ``(cos, sin) in [-1, 1]^2``.

    Range-independent. For pair angle differences, use
    ``obs.AngleDifference(...)`` or ``obs.TrkDeg().relative_to_own(...)``.

    As an ACTION encoder the pair is decoded by ``atan2``, so only its direction
    reaches the sim and its magnitude is a redundant degree of freedom. See
    ``is_circular`` on :class:`Normalizer`: with a Beta policy over the square
    that redundancy is a flat direction in the gradient, which parks the radius
    near 0 where angular sensitivity explodes. Pair it with an actor
    ``distribution="beta_vonmises"`` so the slot gets a von Mises on the circle.
    """

    is_circular = True
    output_interval = (-1.0, 1.0)

    def normalize(self, field, value, idx):
        rad = math.radians(value)
        return [math.cos(rad), math.sin(rad)]

    def normalize_many(self, field, values, idx):
        rad = np.radians(np.asarray(values, dtype=np.float64))
        return np.stack([np.cos(rad), np.sin(rad)], axis=-1).astype(np.float32)

    def denormalize(self, field, value, idx):
        if not isinstance(value, Sequence) or len(value) != 2:
            raise ValueError(
                f"CircularNormalizer expected two action values for "
                f"{field.meta.name!r}, got {value!r}."
            )
        angle = math.degrees(math.atan2(float(value[1]), float(value[0])))
        lo, hi = self._bounds(field, idx)
        if hi - lo < 360.0:
            raise ValueError(
                f"CircularNormalizer action field {field.meta.name!r} must span "
                f"at least 360 degrees, got bounds ({lo}, {hi})."
            )
        while angle < lo:
            angle += 360.0
        while angle > hi:
            angle -= 360.0
        return angle

    def output_size(self, field):
        return 2


class PerFieldNormalizer(Normalizer):
    """Apply a different normalizer per field, with an optional default."""

    def __init__(
        self,
        field_map: Mapping[str | FieldLike, Normalizer],
        default: Normalizer | None = None,
    ) -> None:
        self._map = {_field_key(k): v for k, v in field_map.items()}
        self._default = default or RawNormalizer()

    def _strategy(self, field: str | FieldLike) -> Normalizer:
        return self._map.get(_field_key(field), self._default)

    def normalize(self, field, value, idx):
        return self._strategy(field).normalize(field, value, idx)

    def denormalize(self, field, value, idx):
        return self._strategy(field).denormalize(field, value, idx)

    def output_size(self, field):
        return self._strategy(field).output_size(field)

    def output_bounds(self, field):
        return self._strategy(field).output_bounds(field)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _field_key(field: str | FieldLike) -> str:
    return field.meta.name if hasattr(field, "meta") else str(field)


def _static_or_custom_bounds(field: FieldLike) -> tuple[float, float] | None:
    if field.bounds_overridden:
        return float(field.low), float(field.high)
    if field.meta.dynamic_bounds:
        return None
    return field.bounds(0)
