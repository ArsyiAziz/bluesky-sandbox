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
from collections.abc import Sequence
from typing import Annotated, Any, TypeAlias

import numpy as np

from bluesky_sandbox.interface.fields.base import (
    ActionField,
    ObsField,
    PairObsField,
    Unit,
)
from bluesky_sandbox.interface.fields.grid import Grid

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

    #: True when this normalizer makes an action a CHOICE among a few values:
    #: the action is an index, ``0 .. n_choices - 1``, in the ``discrete``
    #: (``MultiDiscrete``) part of the action space, and :meth:`denormalize`
    #: takes that index. Declared on the base so a layout asks without an
    #: isinstance check.
    discrete: bool = False

    #: Whether this strategy holds the normalized value to
    #: :attr:`normalized_interval`. Declared on the base so :meth:`_clip` and
    #: :meth:`_action_scalar` are safe for strategies with no notion of
    #: clipping.
    clipped: bool = False

    #: The range of the NORMALIZED value - the one a policy reads and writes.
    #: The field's own ``bounds()`` are the other side of the mapping, in
    #: physical units. Named for the value rather than for a direction,
    #: because the direction flips: ``normalize`` produces this range for an
    #: observation, and ``denormalize`` consumes it for an action.
    #:
    #: ``None`` means the strategy does not impose a range of its own, and the
    #: field's bounds are published unchanged. To leave a field unscaled
    #: entirely, attach no normalizer at all - ``normalizer=None`` on the field
    #: - which also preserves a multi-component field's full width, something
    #: the scalar ``Normalizer`` contract cannot.
    #:
    #: One constant drives three things that have to agree: the Box space
    #: built from :meth:`output_bounds`, the clamp :meth:`normalize` applies
    #: when ``clipped``, and the clamp :meth:`denormalize` applies to an
    #: incoming action. Each strategy used to write all three out separately,
    #: and two of the denormalize clamps were simply missing - so an
    #: out-of-range action commanded past the field's own bound.
    normalized_interval: tuple[float, float] | None = None

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
        """Hold a normalized value to :attr:`normalized_interval`.

        Takes scalars and arrays alike, so :meth:`normalize` and
        :meth:`normalize_many` cannot end up clipping differently.
        """
        if not self.clipped or self.normalized_interval is None:
            return value
        low, high = self.normalized_interval
        if isinstance(value, np.ndarray):
            return np.clip(value, low, high)
        return min(max(value, low), high)

    def _action_scalar(self, field: FieldLike, value) -> float:
        """Unwrap a one-element action and hold it to :attr:`normalized_interval`.

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
        """Resolve a strictly positive normalization span for ``field``."""
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
        if self.normalized_interval is not None:
            low, high = self.normalized_interval
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

class _ScaledNormalizer(Normalizer):
    """Map the field's bounds onto :attr:`normalized_interval`, through a curve.

    Every bounded strategy is this one shape: reduce the raw value to a unit
    position ``u = (value - low) / span``, bend ``u`` with a curve, then place
    the result in the output interval. Only the curve differs between
    strategies, so that is all a subclass defines - :meth:`_curve` and its
    inverse :meth:`_uncurve`, both on ``[0, 1]``.

    ``normalized_low`` / ``normalized_high`` override the class's default
    range, each independently - pass one to move a single end. The range a
    strategy may emit is a property of the *policy head* reading it, not of
    the curve underneath: a tanh head lives on ``[-1, 1]``, a sigmoid on
    ``[0, 1]``, a von Mises on ``[-pi, pi]``. Any of those can sit over any of
    these curves, so the two are separate choices.
    """

    def __init__(
        self,
        *,
        clipped: bool = False,
        normalized_low: float | None = None,
        normalized_high: float | None = None,
    ) -> None:
        self.clipped = clipped
        if normalized_low is not None or normalized_high is not None:
            default_low, default_high = type(self).normalized_interval
            self.normalized_interval = _validated_interval(
                default_low if normalized_low is None else normalized_low,
                default_high if normalized_high is None else normalized_high,
                type(self).__name__,
            )

    @property
    def normalized_low(self) -> float:
        return self.normalized_interval[0]

    @property
    def normalized_high(self) -> float:
        return self.normalized_interval[1]

    @staticmethod
    def _signed_pow(x, p):
        """Signed power, for scalars and arrays alike."""
        return np.copysign(np.abs(x) ** p, x)

    def _curve(self, u):
        """Bend a unit position. Identity unless a subclass says otherwise."""
        return u

    def _uncurve(self, c):
        """Inverse of :meth:`_curve`."""
        return c

    def _place(self, c):
        low, high = self.normalized_interval
        return low + c * (high - low)

    def _unplace(self, value):
        low, high = self.normalized_interval
        return (value - low) / (high - low)

    def normalize(self, field, value, idx):
        lo, _ = self._bounds(field, idx)
        u = (value - lo) / self._span(field, idx)
        return [float(self._clip(self._place(self._curve(u))))]

    def normalize_many(self, field, values, idx):
        lo, _ = self._bounds(field, idx)
        u = (np.asarray(values, dtype=np.float64) - lo) / self._span(field, idx)
        out = np.asarray(self._place(self._curve(u)), dtype=np.float64)
        return self._clip(out).reshape(-1, 1).astype(np.float32)

    def denormalize(self, field, value, idx):
        u = self._uncurve(self._unplace(self._action_scalar(field, value)))
        lo, _ = self._bounds(field, idx)
        return lo + float(u) * self._span(field, idx)


class MinMaxNormalizer(_ScaledNormalizer):
    """Linear scaling, to ``[0, 1]`` by default.

    Difference pair fields should expose delta bounds directly; with bounds
    ``[-span, +span]`` this maps zero difference to ``0.5``.

    Differs from :class:`SymmetricNormalizer` only in its default interval;
    both are the identity curve. Pass ``normalized_interval`` to either for any
    other range.
    """

    normalized_interval = (0.0, 1.0)


class SymmetricNormalizer(_ScaledNormalizer):
    """Linear scaling, to ``[-1, 1]`` by default.

    Difference pair fields should expose delta bounds directly; with bounds
    ``[-span, +span]`` this maps to ``[-1, 1]``.

    For a full-turn angle field, ``normalized_low=-math.pi,
    normalized_high=math.pi`` gives
    a scalar angle in radians on exactly the support of a von Mises - and the
    identification at the interval's ends (``-pi`` and ``+pi`` are one sample)
    lands on the field's own ends (``0`` and ``360`` are one heading), so the
    wrap is consistent without any special handling. That is the scalar
    alternative to :class:`CircularNormalizer`'s ``(cos, sin)`` pair, which
    costs a second action slot to carry a magnitude ``atan2`` throws away.
    """

    normalized_interval = (-1.0, 1.0)


class SignedPowerNormalizer(_ScaledNormalizer):
    """Expo-style nonlinear scaling, to ``[-1, 1]`` by default: fine near the
    center, full authority at the extremes.

    Maps the field's bounds to the output interval like
    :class:`SymmetricNormalizer`, but passes the value through a signed power
    curve so most of the range near the center resolves to *small* physical
    values while the interval's ends still reach the full bound. With symmetric
    delta bounds ``[-b, b]`` the center is the goal-seeking ``0`` action and
    ``denormalize(a) = sign(a) * |a|**power * b`` - the policy gets fine control
    near ``0`` without capping the maximum maneuver.

    ``power > 1`` sharpens the curve (finer near the center); ``power == 1``
    recovers the linear :class:`SymmetricNormalizer`.
    """

    normalized_interval = (-1.0, 1.0)

    def __init__(
        self,
        *,
        power: float = 3.0,
        clipped: bool = False,
        normalized_low: float | None = None,
        normalized_high: float | None = None,
    ) -> None:
        if power <= 0.0:
            raise ValueError(
                f"SignedPowerNormalizer power must be > 0, got {power}."
            )
        self.power = float(power)
        super().__init__(
            clipped=clipped,
            normalized_low=normalized_low,
            normalized_high=normalized_high,
        )

    # Curve about the MIDPOINT of the unit interval, then back into [0, 1].
    def _curve(self, u):
        return 0.5 * (1.0 + self._signed_pow(2.0 * u - 1.0, 1.0 / self.power))

    def _uncurve(self, c):
        return 0.5 * (1.0 + self._signed_pow(2.0 * c - 1.0, self.power))


class PowerNormalizer(_ScaledNormalizer):
    """One-sided expo-style scaling, to ``[0, 1]`` by default: fine near
    ``low``, full range at ``high``.

    The one-sided counterpart of :class:`SignedPowerNormalizer`: maps the
    field's bounds to the output interval like :class:`MinMaxNormalizer`, but
    through a power curve anchored at ``low`` -
    ``normalize(v) = ((v - low) / span) ** (1/power)`` - so resolution
    concentrates near the *lower bound* instead of the interval midpoint.
    Suits magnitude-like quantities (ranges, times-to-go) whose
    decision-relevant band hugs ``low``: with ``power == 2`` a range field
    resolves sqrt-fine near zero while the interval's top still reaches the
    full bound.

    ``power > 1`` sharpens the curve (finer near ``low``); ``power == 1``
    recovers the linear :class:`MinMaxNormalizer`. Below-``low`` values stay
    monotonic via a signed power (negative output) unless ``clipped``.
    """

    normalized_interval = (0.0, 1.0)

    def __init__(
        self,
        *,
        power: float = 2.0,
        clipped: bool = False,
        normalized_low: float | None = None,
        normalized_high: float | None = None,
    ) -> None:
        if power <= 0.0:
            raise ValueError(
                f"PowerNormalizer power must be > 0, got {power}."
            )
        self.power = float(power)
        super().__init__(
            clipped=clipped,
            normalized_low=normalized_low,
            normalized_high=normalized_high,
        )

    # Curve anchored at the BOTTOM of the unit interval.
    def _curve(self, u):
        return self._signed_pow(u, 1.0 / self.power)

    def _uncurve(self, c):
        return self._signed_pow(c, self.power)


class CircularNormalizer(Normalizer):
    """Encode an angle (degrees) as ``(cos, sin) in [-1, 1]^2``.

    Only accepts a field that is genuinely circular - unit ``DEG``, spanning
    one full turn - checked when the space is built. Anything else has no
    circular encoding, and the failure is otherwise silent rather than loud.

    Range-independent *within* that constraint: the encoding reads the value
    directly, so the bounds set the contract but never scale the result. For
    pair angle differences, use ``obs.AngleDifference(...)`` or
    ``obs.TrkDeg().relative_to_own(...)``.

    As an ACTION encoder the pair is decoded by ``atan2``, so only its direction
    reaches the sim and its magnitude is a redundant degree of freedom. See
    ``is_circular`` on :class:`Normalizer`: with a Beta policy over the square
    that redundancy is a flat direction in the gradient, which parks the radius
    near 0 where angular sensitivity explodes. Pair it with an actor
    ``distribution="beta_vonmises"`` so the slot gets a von Mises on the circle.
    """

    is_circular = True
    # Not configurable, unlike the scaled strategies. ``atan2`` recovers the
    # angle from the pair's DIRECTION, which survives a positive scaling but
    # not a translation - so any interval not centered on zero would decode to
    # the wrong angle. ``(-1, 1)`` is the unit circle and the only sensible
    # choice, which is why there is no ``normalized_low``/``normalized_high``
    # here.
    normalized_interval = (-1.0, 1.0)

    def _require_a_full_turn(self, field) -> None:
        """Reject a field that is not an angle in degrees spanning one turn.

        The encoding is ``cos``/``sin`` of the value read as DEGREES, which
        only means anything for a quantity that wraps. Applied to anything
        else it silently produces a plausible-looking pair rather than
        failing: a field holding radians reads ``pi`` as 3.14 *degrees* and
        collapses the whole circle into a 6.3 degree arc, and an altitude in
        feet aliases every 360 ft onto the same point.

        Checked where the space is built rather than per value. ``normalize``
        runs per field, per aircraft, per step, and reading the field's bounds
        there costs more than the ``cos``/``sin`` it would be guarding.
        """
        unit = getattr(field.meta, "unit", None)
        if unit is not Unit.DEG:
            raise ValueError(
                f"CircularNormalizer needs an angle in degrees, but "
                f"{field.meta.name!r} is in {unit}. Use a scaled strategy "
                f"for a non-angular quantity."
            )
        static = _static_or_custom_bounds(field)
        if static is None:
            # Dynamic bounds resolve against live traffic, so there is nothing
            # to check yet; ``denormalize`` still checks the action path.
            return
        span = static[1] - static[0]
        if not math.isclose(span, 360.0):
            raise ValueError(
                f"CircularNormalizer needs a field spanning one full turn, "
                f"but {field.meta.name!r} spans {span} degrees. A quantity "
                f"that does not wrap has no circular encoding - "
                f"SymmetricNormalizer scales it linearly instead."
            )

    def output_bounds(self, field):
        # The one hook every configured field passes through when its space is
        # built, so a bad pairing fails at construction, not mid-episode.
        self._require_a_full_turn(field)
        return super().output_bounds(field)

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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class StepNormalizer(Normalizer):
    """Whole steps, as a choice: the action is a DISCRETE choice among
    ``k * step`` (in the field's own unit), for ``k`` from ``-steps_each_way``
    to ``+steps_each_way`` - that many steps down, and that many up.

    The action is the index of a step in :meth:`steps` (``0 .. n_choices - 1``)
    and sits in the ``discrete`` part of the action space, a ``MultiDiscrete``:
    a policy picks one, as from any discrete space.

    ``StepNormalizer(1000, 10, include_zero=False)`` on a level change: the
    choices are -10..-1 and +1..+10 levels of 1,000 ft - twenty in all, index 0
    being 10 levels down.

    ``include_zero``: whether ``k = 0`` - 0 steps, "change nothing" - is one of
    the choices. Leave it out for a clearance: a call that changes nothing
    says nothing.

    ON A GRID - an action with a :class:`~bluesky_sandbox.interface.fields.grid.Grid`
    - the steps are the grid's, and ``step`` COUNTS them: a whole number of
    grid steps per choice, 1 when left unset. On ``Grid(1000)``
    (``on="value"``), ``k`` is ``k * step`` whole thousands added to the
    nominal: from 23,344 ft, +1 is 24,344 ft (``step=2``: 25,344 ft). On
    ``Grid(1000, on="target")`` the choices are the grid's values, counted
    from the present one: ``k = +1`` the next grid value above it, ``-1`` the
    next below, ``0`` the nearest - from 23,344 ft or 23,600 ft, +1 is FL240;
    from FL240 itself, FL250 (``step=2``: two levels each, FL250 and FL260).
    Without a grid, ``k`` is ``k * step`` in the action's unit, and ``step``
    is required.

    Discretizes any scalar field the way air traffic control does - turns in
    10 deg, levels in 1,000 ft, speeds in 10 kt: a delta counts its steps from
    its nominal (0 flies on), an absolute field from zero. What the field can
    command still holds, dynamically: a step is clipped to the action's
    :meth:`~bluesky_sandbox.interface.fields.base.ActionField.reach` - for a
    delta its full asymmetric range - so a step the aircraft cannot take now
    becomes the largest one it can - on a grid, the reachable grid value
    nearest the one asked for.
    """

    discrete = True

    def __init__(
        self,
        step: Annotated[
            float | None,
            "one step, in the action's unit - on a grid, a whole number of grid "
            "steps; None = one grid step",
        ] = None,
        steps_each_way: int = 10,
        *,
        include_zero: bool = True,
    ) -> None:
        if step is not None and not float(step) > 0.0:
            raise ValueError(f"StepNormalizer step must be > 0, got {step!r}")
        if int(steps_each_way) != steps_each_way or steps_each_way < 1:
            raise ValueError(
                "StepNormalizer steps_each_way must be a whole number >= 1, got "
                f"{steps_each_way!r}"
            )
        self.step = None if step is None else float(step)
        self.steps_each_way = int(steps_each_way)
        self.include_zero = bool(include_zero)

    def grid_for(self, field: Any = None) -> tuple[Grid, int]:
        """The grid the steps are on, and how many of its steps one choice
        is: the action's grid and ``step`` of them (1 unset) where it has one,
        else whole steps of ``step``."""
        grid = getattr(field, "grid", None)
        if grid is not None:
            every = 1.0 if self.step is None else self.step
            if every != int(every):
                raise ValueError(
                    f"StepNormalizer step {self.step:g} on {field.meta.name!r}, whose "
                    f"grid is {grid.step:g}: on a grid the step counts grid steps - "
                    "a whole number"
                )
            return grid, int(every)
        if self.step is None:
            raise ValueError(
                "StepNormalizer has no step: give it one, or give the action a "
                "grid to step along"
            )
        return Grid(self.step), 1

    def step_for(self, field: Any = None) -> float:
        """One step, in the action's unit: ``step`` grid steps on the
        action's grid where it has one, else ``step``."""
        grid, every = self.grid_for(field)
        return grid.step * every

    def steps(self) -> list[int]:
        """Each choice's number of steps ``k``, in index order."""
        n = self.steps_each_way
        return [k for k in range(-n, n + 1) if self.include_zero or k != 0]

    @property
    def n_choices(self) -> int:
        """How many choices: the size of this action's ``MultiDiscrete`` entry."""
        return len(self.steps())

    def output_bounds(self, field):
        return [0.0], [float(self.n_choices - 1)]

    def denormalize(self, field, value, idx):
        """The command for choice ``value`` (an index into :meth:`steps`)."""
        if isinstance(value, Sequence):
            if len(value) != 1:
                raise ValueError(
                    f"StepNormalizer expected one choice for {field.meta.name!r}, "
                    f"got {len(value)}."
                )
            value = value[0]
        steps = self.steps()
        k = steps[min(max(int(round(float(value))), 0), len(steps) - 1)]
        grid, every = self.grid_for(field)
        return grid.nth(field, k, idx, every)

    def normalize(self, field, value, idx):
        """The choice nearest ``value`` (in the field's unit), as its index."""
        steps = np.asarray(self.steps(), dtype=np.float64)
        return [float(np.abs(steps * self.step_for(field) - float(value)).argmin())]

    def normalize_many(self, field, values, idx):
        steps = np.asarray(self.steps(), dtype=np.float64) * self.step_for(field)
        values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        return np.abs(values - steps[None]).argmin(-1).reshape(-1, 1).astype(np.float32)


def _validated_interval(low, high, owner: str) -> tuple[float, float]:
    """Check a caller-supplied normalized range before it reaches arithmetic.

    A flat range divides by zero and an inverted one silently flips the sign
    of every action; both would otherwise surface far from the constructor
    that caused them.
    """
    try:
        low, high = float(low), float(high)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{owner} normalized_low/normalized_high must be numbers, got "
            f"({low!r}, {high!r})."
        ) from exc
    if not high > low:
        raise ValueError(
            f"{owner} needs normalized_high > normalized_low, got "
            f"({low}, {high})."
        )
    return low, high


def _static_or_custom_bounds(field: FieldLike) -> tuple[float, float] | None:
    if field.bounds_overridden:
        return float(field.low), float(field.high)
    if field.meta.dynamic_bounds:
        return None
    return field.bounds(0)
