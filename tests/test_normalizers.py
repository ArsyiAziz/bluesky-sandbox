"""Normalizers: every field, a few normalizers.

The two axes are deliberately asymmetric. **Fields** are swept exhaustively -
every concrete class in :mod:`~.fields.observations` and
:mod:`~.fields.actions`, discovered by reflection rather than listed here, so a
field added tomorrow is covered the day it lands. **Normalizers** are sampled:
one configuration per concrete strategy, because that hierarchy is small,
stable, and its behaviour does not vary with the field beyond the bounds it
reads.

Every field is given explicit ``low``/``high``. That is what makes the sweep
possible at all - roughly 40% of fields resolve their bounds from live
``bs.traf`` (``AltFt``, ``CasKts``, every ``Conflict*``) and would otherwise
need a booted simulator. It is also the right scope: these tests pin the
*normalizer's* arithmetic, and where a field's bounds come from is a separate
question with separate tests.

Four invariants:

* ``normalize_many`` agrees with ``normalize`` elementwise - ``services``
  promises the batched intruder path is "byte-identical to normalizing each
  element", and nothing checked it.
* ``denormalize(normalize(v)) == v`` for action fields.
* The emitted width matches the declared ``output_size`` / ``output_bounds``.
* ``clipped=True`` means an out-of-range action commands the bound, never past
  it.

Known defects are marked ``xfail(strict=True)``, so whichever fix lands first
reports XPASS and has to remove the marker. Pairings the sweep simply cannot
measure - a multi-output field under a scalar normalizer - are skipped
instead, and the defect behind the skip is pinned by a strict test of its own.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass

import numpy as np
import pytest

import bluesky_sandbox.interface.fields.actions as actions
import bluesky_sandbox.interface.fields.observations as observations
from bluesky_sandbox.core import services
from bluesky_sandbox.interface.fields import base
from bluesky_sandbox.interface.fields.base import ActionField, ObsField, PairObsField
from bluesky_sandbox.interface.wrappers.observations import normalizer as nz

# ---- field discovery ----------------------------------------------------- #

# Neither module defines ``__all__``, so reflect over the module namespace and
# filter to concrete leaves of the field hierarchy.
_BASES = (ObsField, PairObsField, ActionField)


def _concrete(module, base) -> list[type]:
    found = []
    for name in sorted(dir(module)):
        if name.startswith("_"):
            continue
        obj = getattr(module, name, None)
        if (
            inspect.isclass(obj)
            and issubclass(obj, base)
            and obj not in _BASES
            and not inspect.isabstract(obj)
        ):
            found.append(obj)
    return found


OBS_FIELDS = _concrete(observations, (ObsField, PairObsField))
ACTION_FIELDS = _concrete(actions, ActionField)

# Composite fields wrap another field, so ``cls(low=..., high=...)`` alone
# leaves them half-built. ``LaggedObs``/``LaggedPair`` deliberately resolve
# their bounds from ``inner`` - a stacked channel has to land on the same scale
# as the live field beside it - so the bounds go on the inner field, not the
# wrapper, or the sweep would be normalizing against a scale it did not set.
_EXTRA_KWARGS = {
    "Difference": lambda lo, hi: {
        "left": observations.LatDeg(low=lo, high=hi),
        "right": observations.LonDeg(low=lo, high=hi),
    },
    "AngleDifference": lambda lo, hi: {
        "left": observations.TrkDeg(low=lo, high=hi),
        "right": observations.HdgDeg(low=lo, high=hi),
    },
    "LaggedObs": lambda lo, hi: {
        "inner": observations.LatDeg(low=lo, high=hi),
        "steps": 1,
    },
    "LaggedPair": lambda lo, hi: {
        "inner": observations.DistToOwnNm(low=lo, high=hi),
        "steps": 1,
    },
}

# Multi-output fields: ``bounds()`` returns one pair per component, so the
# scalar ``Normalizer`` contract does not apply and the sweep cannot measure
# them. That the pairing is *refused* is pinned separately, by
# ``test_per_component_bounds_are_refused_by_name`` and
# ``test_prev_action_norm_refuses_a_normalizer``.
_MULTI_OUTPUT = frozenset({"FlightPhaseOneHot", "PrevActionNorm"})


def _build(cls: type, *, low: float, high: float, normalizer=None):
    extra = _EXTRA_KWARGS.get(cls.__name__)
    kwargs = extra(low, high) if extra is not None else {}
    return cls(low=low, high=high, normalizer=normalizer, **kwargs)


def _ids(classes: list[type]) -> list[str]:
    return [c.__name__ for c in classes]


# ---- normalizer samples --------------------------------------------------- #


@dataclass(frozen=True)
class Sample:
    """One normalizer configuration, with bounds and probe values that suit it.

    ``PowerNormalizer`` anchors its curve at ``low`` rather than the midpoint,
    so it gets a one-sided range; ``CircularNormalizer`` needs a full 360 to
    denormalize at all. Carrying the bounds here keeps the sweep a flat
    cross-product instead of a nest of per-normalizer special cases.
    """

    name: str
    normalizer: nz.Normalizer
    bounds: tuple[float, float]
    values: tuple[float, ...]


SAMPLES = (
    Sample("Raw", nz.RawNormalizer(), (-10.0, 10.0), (-10.0, -5.0, 0.0, 5.0, 10.0)),
    Sample(
        "MinMax",
        nz.MinMaxNormalizer(clipped=True),
        (-10.0, 10.0),
        (-10.0, -5.0, 0.0, 5.0, 10.0),
    ),
    Sample(
        "Symmetric",
        nz.SymmetricNormalizer(clipped=True),
        (-10.0, 10.0),
        (-10.0, -5.0, 0.0, 5.0, 10.0),
    ),
    Sample(
        "SignedPower",
        nz.SignedPowerNormalizer(power=3.0),
        (-10.0, 10.0),
        (-10.0, -1.25, 0.0, 1.25, 10.0),
    ),
    Sample(
        "Power",
        nz.PowerNormalizer(power=2.0),
        (0.0, 100.0),
        (0.0, 6.25, 25.0, 56.25, 100.0),
    ),
    # 360 exactly would round-trip to 0 - same angle, different number - so the
    # probe stops short of it.
    Sample("Circular", nz.CircularNormalizer(), (0.0, 360.0), (0.0, 45.0, 90.0, 180.0, 270.0)),
    # A non-default interval, swept over every field like any other config.
    Sample(
        "SymmetricPi",
        nz.SymmetricNormalizer(normalized_low=-math.pi, normalized_high=math.pi),
        (-10.0, 10.0),
        (-10.0, -5.0, 0.0, 5.0, 10.0),
    ),
)
SAMPLE_IDS = [s.name for s in SAMPLES]

ALL_FIELDS = OBS_FIELDS + ACTION_FIELDS


def _sweep_field(cls: type, sample: Sample):
    """Build ``cls`` for ``sample``, skipping pairings that are out of scope.

    This is a *skip*, not an xfail: the sweep cannot scale a field whose bounds
    are per-component, but that is a missing measurement rather than a
    known-wrong answer. The wrong answer itself is owned by
    :func:`test_scalar_normalizer_on_a_multi_output_field_is_rejected`.
    """
    if cls.__name__ in _MULTI_OUTPUT:
        pytest.skip(f"{cls.__name__} is multi-output; scalar normalizers do not apply")
    field = _build(
        cls, low=sample.bounds[0], high=sample.bounds[1], normalizer=sample.normalizer
    )
    # CircularNormalizer refuses anything that is not an angle in degrees, so
    # the sweep cannot pair it with a speed or a distance. That refusal is
    # behaviour under test in its own right - see
    # ``test_circular_refuses_a_field_that_does_not_wrap``.
    if isinstance(sample.normalizer, nz.CircularNormalizer):
        unit = getattr(field.meta, "unit", None)
        if unit is not base.Unit.DEG:
            pytest.skip(f"{cls.__name__} is in {unit}; CircularNormalizer needs degrees")
    return field


# ---- the registry itself -------------------------------------------------- #


def test_the_sweep_actually_covers_the_field_modules():
    """A reflection-built registry can silently empty out; pin the magnitude."""
    assert len(OBS_FIELDS) > 70, f"only found {len(OBS_FIELDS)} observation fields"
    assert len(ACTION_FIELDS) > 15, f"only found {len(ACTION_FIELDS)} action fields"
    assert observations.AltFt in OBS_FIELDS
    assert actions.HdgDeltaDeg in ACTION_FIELDS


@pytest.mark.parametrize("cls", ALL_FIELDS, ids=_ids(ALL_FIELDS))
def test_every_field_builds_with_explicit_bounds(cls):
    """The premise of the sweep: bounds can be forced without a live sim.

    Every dynamic field resolves through ``_dynamic_or_configured_bounds``,
    which consults ``bounds_overridden`` before touching ``bs.traf`` - but only
    if the caller defers the traffic read into the callable rather than
    computing it first. Nothing static catches that, so this does.
    """
    field = _build(cls, low=-10.0, high=10.0)
    low, high = field.bounds(0)
    assert np.all(np.asarray(low) <= np.asarray(high))


# ---- invariant 1: batched == unbatched ------------------------------------ #


@pytest.mark.parametrize("sample", SAMPLES, ids=SAMPLE_IDS)
@pytest.mark.parametrize("cls", ALL_FIELDS, ids=_ids(ALL_FIELDS))
def test_normalize_many_matches_normalize(cls, sample):
    """``services`` calls the batched path for intruders and the scalar path for
    ownship; a divergence would make an agent's own row disagree with how it
    sees everyone else."""
    field = _sweep_field(cls, sample)
    one_by_one = np.asarray(
        [sample.normalizer.normalize(field, v, 0) for v in sample.values],
        dtype=np.float32,
    )
    batched = np.asarray(
        sample.normalizer.normalize_many(field, list(sample.values), 0),
        dtype=np.float32,
    )
    assert batched.shape == one_by_one.shape
    np.testing.assert_array_equal(batched, one_by_one)


@pytest.mark.parametrize("sample", SAMPLES, ids=SAMPLE_IDS)
def test_normalize_many_handles_an_empty_batch(sample):
    """An ownship with no intruders still has to produce a correctly shaped
    ``(0, width)`` array - the width cannot be inferred from the elements."""
    field = _build(
        observations.DistToOwnNm,
        low=sample.bounds[0],
        high=sample.bounds[1],
        normalizer=sample.normalizer,
    )
    out = np.asarray(sample.normalizer.normalize_many(field, [], 0))
    assert out.shape == (0, sample.normalizer.output_size(field))


# ---- invariant 2: action round-trip --------------------------------------- #


@pytest.mark.parametrize("sample", SAMPLES, ids=SAMPLE_IDS)
@pytest.mark.parametrize("cls", ACTION_FIELDS, ids=_ids(ACTION_FIELDS))
def test_action_round_trips_through_denormalize(cls, sample):
    """A policy emits normalized values; the sim needs physical ones back. Any
    asymmetry here is a silent, systematic command bias."""
    field = _sweep_field(cls, sample)
    for value in sample.values:
        encoded = sample.normalizer.normalize(field, value, 0)
        decoded = sample.normalizer.denormalize(field, encoded, 0)
        decoded = float(np.asarray(decoded).reshape(-1)[0])
        assert decoded == pytest.approx(value, rel=1e-6, abs=1e-6), (
            f"{cls.__name__} + {sample.name}: {value} -> {encoded} -> {decoded}"
        )


# ---- invariant 3: declared width matches emitted width -------------------- #


@pytest.mark.parametrize("sample", SAMPLES, ids=SAMPLE_IDS)
@pytest.mark.parametrize("cls", ALL_FIELDS, ids=_ids(ALL_FIELDS))
def test_emitted_width_matches_the_declared_output_size(cls, sample):
    """``output_size`` sizes the Box space at build time; if the value emitted
    at step time is wider, the space is a lie and assembly breaks."""
    field = _sweep_field(cls, sample)
    width = sample.normalizer.output_size(field)
    emitted = sample.normalizer.normalize(field, sample.values[0], 0)
    assert len(emitted) == width
    # Nothing array-shaped should survive: a nested element makes the assembled
    # observation row ragged against every scalar field beside it.
    assert all(np.asarray(v).ndim == 0 for v in emitted), f"nested value: {emitted!r}"
    low, high = sample.normalizer.output_bounds(field)
    assert len(low) == len(high) == width


@pytest.mark.parametrize("sample", SAMPLES, ids=SAMPLE_IDS)
@pytest.mark.parametrize("cls", ALL_FIELDS, ids=_ids(ALL_FIELDS))
def test_normalized_values_land_inside_the_declared_output_bounds(cls, sample):
    """The Box space is built from ``output_bounds``; a value outside it fails
    ``contains`` and trips env-checker wrappers."""
    if sample.name == "Raw":
        pytest.skip("RawNormalizer declares the field's own bounds, tested elsewhere")
    field = _sweep_field(cls, sample)
    low, high = sample.normalizer.output_bounds(field)
    for value in sample.values:
        for i, out in enumerate(sample.normalizer.normalize(field, value, 0)):
            assert low[i] - 1e-6 <= out <= high[i] + 1e-6, (
                f"{cls.__name__} + {sample.name}: {value} -> {out} outside "
                f"[{low[i]}, {high[i]}]"
            )


# ---- invariant 4: clipped means clipped ----------------------------------- #


@pytest.mark.parametrize(
    ("name", "normalizer", "bounds", "expected"),
    [
        pytest.param("MinMax", nz.MinMaxNormalizer(clipped=True), (-10.0, 10.0), 10.0),
        pytest.param("Symmetric", nz.SymmetricNormalizer(clipped=True), (-10.0, 10.0), 10.0),
        pytest.param("SignedPower", nz.SignedPowerNormalizer(clipped=True), (-10.0, 10.0), 10.0),
        pytest.param("Power", nz.PowerNormalizer(clipped=True), (0.0, 100.0), 100.0),
    ],
    ids=["MinMax", "Symmetric", "SignedPower", "Power"],
)
def test_clipped_denormalize_commands_the_bound_and_no_further(
    name, normalizer, bounds, expected
):
    """An unsquashed policy head emits values outside the space routinely.
    All four read the clamp from ``normalized_interval``, the same constant that
    builds the Box space, so the flag cannot mean different things on
    different strategies.
    """
    field = actions.HdgDeltaDeg(low=bounds[0], high=bounds[1], normalizer=normalizer)
    assert normalizer.denormalize(field, [1.5], 0) == pytest.approx(expected)


# ---- golden values -------------------------------------------------------- #
#
# A round-trip passes even when both directions are wrong in compensating ways,
# so pin the actual numbers for one field with bounds (-10, 10), span 20.


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-10.0, 0.0), (-5.0, 0.25), (0.0, 0.5), (5.0, 0.75), (10.0, 1.0)],
)
def test_minmax_maps_bounds_to_the_unit_interval(value, expected):
    field = observations.LatDeg(low=-10.0, high=10.0)
    assert nz.MinMaxNormalizer().normalize(field, value, 0) == pytest.approx([expected])


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-10.0, -1.0), (-5.0, -0.5), (0.0, 0.0), (5.0, 0.5), (10.0, 1.0)],
)
def test_symmetric_maps_bounds_to_minus_one_to_one(value, expected):
    field = observations.LatDeg(low=-10.0, high=10.0)
    assert nz.SymmetricNormalizer().normalize(field, value, 0) == pytest.approx([expected])


@pytest.mark.parametrize(
    ("value", "expected"),
    # cube-root curve: a quantity 1/8 of the way out reads as 1/2 the action.
    [(-10.0, -1.0), (-1.25, -0.5), (0.0, 0.0), (1.25, 0.5), (10.0, 1.0)],
)
def test_signed_power_resolves_finely_near_the_centre(value, expected):
    field = observations.LatDeg(low=-10.0, high=10.0)
    got = nz.SignedPowerNormalizer(power=3.0).normalize(field, value, 0)
    assert got == pytest.approx([expected])


@pytest.mark.parametrize(
    ("value", "expected"),
    # sqrt curve anchored at low: 1/16 of the range reads as 1/4 of the output.
    [(0.0, 0.0), (6.25, 0.25), (25.0, 0.5), (56.25, 0.75), (100.0, 1.0)],
)
def test_power_resolves_finely_near_the_lower_bound(value, expected):
    field = observations.DistToOwnNm(low=0.0, high=100.0)
    assert nz.PowerNormalizer(power=2.0).normalize(field, value, 0) == pytest.approx([expected])


@pytest.mark.parametrize(
    ("deg", "expected"),
    [(0.0, (1.0, 0.0)), (90.0, (0.0, 1.0)), (180.0, (-1.0, 0.0)), (270.0, (0.0, -1.0))],
)
def test_circular_encodes_the_compass_as_cos_sin(deg, expected):
    field = observations.HdgDeg(low=0.0, high=360.0)
    got = nz.CircularNormalizer().normalize(field, deg, 0)
    assert got == pytest.approx(list(expected), abs=1e-9)


def test_power_of_one_degenerates_to_its_linear_counterpart():
    """The docstrings promise ``power == 1`` recovers the linear strategy."""
    field = observations.LatDeg(low=-10.0, high=10.0)
    for value in (-10.0, -3.0, 0.0, 7.0, 10.0):
        assert nz.SignedPowerNormalizer(power=1.0).normalize(
            field, value, 0
        ) == pytest.approx(nz.SymmetricNormalizer().normalize(field, value, 0))
        assert nz.PowerNormalizer(power=1.0).normalize(
            field, value, 0
        ) == pytest.approx(nz.MinMaxNormalizer().normalize(field, value, 0))


# ---- guards and error messages -------------------------------------------- #


# ---- the output interval is a free parameter ------------------------------ #


@pytest.mark.parametrize(
    "cls",
    [
        nz.MinMaxNormalizer,
        nz.SymmetricNormalizer,
        nz.SignedPowerNormalizer,
        nz.PowerNormalizer,
    ],
    ids=["MinMax", "Symmetric", "SignedPower", "Power"],
)
def test_the_normalized_range_can_be_chosen_per_instance(cls):
    """The range a strategy emits belongs to the policy head reading it, not to
    the curve underneath, so the two are configured separately."""
    field = observations.LatDeg(low=-10.0, high=10.0)
    interval = (-math.pi, math.pi)
    normalizer = cls(normalized_low=interval[0], normalized_high=interval[1])

    assert normalizer.normalized_interval == interval
    assert (normalizer.normalized_low, normalizer.normalized_high) == interval
    low, high = normalizer.output_bounds(field)
    assert (low[0], high[0]) == interval
    # Endpoints map to endpoints whatever the curve in between.
    assert normalizer.normalize(field, -10.0, 0)[0] == pytest.approx(-math.pi)
    assert normalizer.normalize(field, 10.0, 0)[0] == pytest.approx(math.pi)


def test_minmax_and_symmetric_differ_only_in_their_default_interval():
    """Both are the identity curve, so giving one the other's interval makes
    them indistinguishable - which is why the interval is a parameter rather
    than a third class."""
    field = observations.LatDeg(low=-10.0, high=10.0)
    as_symmetric = nz.MinMaxNormalizer(normalized_low=-1.0, normalized_high=1.0)
    for value in (-10.0, -3.0, 0.0, 7.0, 10.0):
        assert as_symmetric.normalize(field, value, 0) == pytest.approx(
            nz.SymmetricNormalizer().normalize(field, value, 0)
        )


def test_a_full_turn_field_on_the_von_mises_support_round_trips():
    """``[-pi, pi]`` is exactly a von Mises' support. The identification at the
    interval's ends (-pi and +pi are one sample) lands on the field's own ends
    (0 and 360 are one heading), so no wrapping is needed.
    """
    field = actions.HdgDeg(low=0.0, high=360.0)
    normalizer = nz.SymmetricNormalizer(
        normalized_low=-math.pi, normalized_high=math.pi
    )
    assert normalizer.output_size(field) == 1  # one slot, not Circular's two
    assert normalizer.denormalize(field, [-math.pi], 0) == pytest.approx(0.0)
    assert normalizer.denormalize(field, [math.pi], 0) == pytest.approx(360.0)
    assert normalizer.denormalize(field, [0.0], 0) == pytest.approx(180.0)
    for deg in (0.0, 37.0, 180.0, 359.0):
        encoded = normalizer.normalize(field, deg, 0)
        assert -math.pi <= encoded[0] <= math.pi
        assert normalizer.denormalize(field, encoded, 0) == pytest.approx(deg)


@pytest.mark.parametrize(
    ("low", "high"),
    [(1.0, 1.0), (5.0, -5.0), ("a", "b")],
    ids=["flat", "inverted", "text"],
)
def test_a_degenerate_normalized_range_is_rejected_at_construction(low, high):
    """Inverted silently flips the sign of every action; flat divides by zero.
    Both would otherwise surface far from the constructor that caused them."""
    with pytest.raises(ValueError, match="normalized_"):
        nz.SymmetricNormalizer(normalized_low=low, normalized_high=high)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"normalized_high": 10.0}, (0.0, 10.0)),
        ({"normalized_low": -5.0}, (-5.0, 1.0)),
        ({"normalized_low": -5.0, "normalized_high": 10.0}, (-5.0, 10.0)),
        ({}, (0.0, 1.0)),
    ],
    ids=["high only", "low only", "both", "neither"],
)
def test_each_end_of_the_normalized_range_moves_independently(kwargs, expected):
    """Passing one end keeps the class default for the other, so a one-sided
    tweak does not force you to restate the end you were happy with."""
    assert nz.MinMaxNormalizer(**kwargs).normalized_interval == expected


@pytest.mark.parametrize("cls", [nz.SignedPowerNormalizer, nz.PowerNormalizer])
@pytest.mark.parametrize("power", [0.0, -1.0])
def test_a_non_positive_power_is_rejected_at_construction(cls, power):
    with pytest.raises(ValueError, match="power must be > 0"):
        cls(power=power)


@pytest.mark.parametrize("sample", SAMPLES, ids=SAMPLE_IDS)
def test_a_degenerate_span_is_rejected_rather_than_dividing_by_zero(sample):
    if isinstance(sample.normalizer, (nz.RawNormalizer, nz.CircularNormalizer)):
        pytest.skip(f"{sample.name} is range-independent")
    field = observations.LatDeg(low=5.0, high=5.0)
    with pytest.raises(ValueError, match="bounds must have high > low"):
        sample.normalizer.normalize(field, 5.0, 0)


@pytest.mark.parametrize(
    ("field_factory", "match"),
    [
        pytest.param(
            lambda: observations.AltFt(low=0.0, high=360.0), "in degrees",
            id="wrong unit, right span",
        ),
        pytest.param(
            lambda: observations.DistToOwnNm(low=0.0, high=360.0), "in degrees",
            id="distance",
        ),
        pytest.param(
            lambda: observations.LatDeg(), "one full turn",
            id="degrees that do not wrap",
        ),
        pytest.param(
            lambda: observations.HdgDeg(low=-math.pi, high=math.pi), "one full turn",
            id="radians mislabelled as degrees",
        ),
    ],
)
def test_circular_refuses_a_field_that_does_not_wrap(field_factory, match):
    """cos/sin of a non-angle is not an error, just nonsense - an altitude in
    feet aliases every 360 ft onto the same point, and a field holding radians
    collapses the whole circle into a 6.3 degree arc. Both used to pass
    silently on the observation path."""
    with pytest.raises(ValueError, match=match):
        nz.CircularNormalizer().output_bounds(field_factory())


@pytest.mark.parametrize(
    "field_factory",
    [
        lambda: observations.HdgDeg(low=0.0, high=360.0),
        lambda: observations.TrkDeg(low=0.0, high=360.0),
        lambda: observations.AngleDifference(
            left=observations.TrkDeg(), right=observations.HdgDeg()
        ),
        lambda: actions.HdgDeg(low=0.0, high=360.0),
    ],
    ids=["HdgDeg", "TrkDeg", "AngleDifference(-180,180)", "action HdgDeg"],
)
def test_circular_accepts_a_genuine_full_turn(field_factory):
    """The guard must not reject the fields it exists to serve - including a
    signed difference on (-180, 180), which wraps just as 0..360 does."""
    assert nz.CircularNormalizer().output_bounds(field_factory()) == (
        [-1.0, -1.0],
        [1.0, 1.0],
    )


def test_circular_takes_no_normalized_range():
    """``atan2`` recovers the angle from the pair's direction, which survives a
    positive scaling but not a translation - so an interval off-centre from
    zero would decode to the wrong angle. The unit circle is the only sensible
    choice, so there is nothing to configure."""
    with pytest.raises(TypeError):
        nz.CircularNormalizer(normalized_low=0.0, normalized_high=1.0)
    assert nz.CircularNormalizer().normalized_interval == (-1.0, 1.0)


def test_circular_refuses_an_action_field_narrower_than_a_full_turn():
    """Decoding is ``atan2``, whose range is a full circle; a narrower field has
    no unambiguous inverse."""
    field = actions.HdgDeltaDeg(low=-30.0, high=30.0)
    with pytest.raises(ValueError, match="at least 360 degrees"):
        nz.CircularNormalizer().denormalize(field, [0.0, 1.0], 0)


def test_circular_denormalize_needs_exactly_two_values():
    field = actions.HdgDeg(low=0.0, high=360.0)
    with pytest.raises(ValueError, match="expected two action values"):
        nz.CircularNormalizer().denormalize(field, [0.5], 0)


@pytest.mark.parametrize(
    "normalizer",
    [nz.MinMaxNormalizer(), nz.SymmetricNormalizer(), nz.SignedPowerNormalizer(), nz.PowerNormalizer()],
    ids=["MinMax", "Symmetric", "SignedPower", "Power"],
)
def test_scalar_denormalize_rejects_a_multi_value_action(normalizer):
    field = actions.HdgDeltaDeg(low=-10.0, high=10.0)
    with pytest.raises(ValueError, match="expected one action value"):
        normalizer.denormalize(field, [0.1, 0.2], 0)


# ---- the services layer that calls all of this ---------------------------- #


def test_services_rejects_a_multi_value_raw_input_to_a_scalar_normalizer():
    field = observations.LatDeg(low=-10.0, high=10.0, normalizer=nz.MinMaxNormalizer())
    with pytest.raises(ValueError, match="expects one raw value"):
        services._normalize_field_value(field, [1.0, 2.0], 0)


def test_services_rejects_an_action_of_the_wrong_width():
    field = actions.HdgDeltaDeg(low=-10.0, high=10.0, normalizer=nz.MinMaxNormalizer())
    with pytest.raises(ValueError, match="expects 1 values"):
        services._denormalize_action_value(field, [0.1, 0.2], 0)


def test_prev_action_norm_refuses_a_normalizer():
    """It stores the policy's own output, already in action space. Scaling it
    again measures it against bounds it was never drawn from - and its bounds
    are per-component, so the result used to be a nested ``[array([x])]``
    against a declared width of 1."""
    with pytest.raises(ValueError, match="already in action space"):
        observations.PrevActionNorm(normalizer=nz.MinMaxNormalizer())


@pytest.mark.parametrize("sample", SAMPLES, ids=SAMPLE_IDS)
@pytest.mark.parametrize(
    "cls", [observations.FlightPhaseOneHot], ids=["FlightPhaseOneHot"]
)
def test_per_component_bounds_are_refused_by_name(cls, sample):
    """The general guard behind the specific one: any field whose bounds are
    per-component fails in the normalizer that cannot scale it, naming the
    field, rather than producing a row that goes ragged at concatenation."""
    if isinstance(sample.normalizer, (nz.RawNormalizer, nz.CircularNormalizer)):
        pytest.skip(f"{sample.name} is range-independent and never reads bounds")
    field = cls(normalizer=sample.normalizer)
    with pytest.raises(TypeError, match="per-component bounds"):
        sample.normalizer.normalize(field, 0.5, 0)


@pytest.mark.parametrize("sample", SAMPLES, ids=SAMPLE_IDS)
@pytest.mark.parametrize("cls", ALL_FIELDS, ids=_ids(ALL_FIELDS))
def test_services_agrees_with_the_normalizer_it_delegates_to(cls, sample):
    """``services`` is the only caller; its wrappers must not reshape or clip
    on their own."""
    field = _sweep_field(cls, sample)
    value = sample.values[2]
    direct = sample.normalizer.normalize(field, value, 0)
    through = services._normalize_field_value(field, value, 0)
    assert list(through) == pytest.approx(list(direct))
    assert services._field_output_size(field) == sample.normalizer.output_size(field)
