"""Every observation field batches exactly like it computes one value.

The observation assembler never calls the one-at-a-time methods: ownship rows
come from ``get_many`` and intruder rows from ``get_pair_matrix``, normalized with
``_normalize_field_values_batch``. Tasks and rewards, though, call ``get`` and
``get_pair``. If the two disagree, an agent observes a different number than
its reward is computed from, and nothing else notices: this sweep found
``get_pair`` on flat-earth ``kwikqdrdist`` while ``get_pairs`` used spherical
``qdrdist`` (0.01 nm and 0.05 deg apart, and bearings in [0, 360) against the
field's own [-180, 180] bounds), and batch paths doing float32 arithmetic.

Compared as float32, which is what reaches an observation. Unlike
``test_normalizers``, which pins the normalizers' arithmetic on fixed bounds and
exact values, this runs every field against live traffic - dynamic bounds,
real float64 state, conflicts, routes, stateful fields - in one env.
"""

from __future__ import annotations

import dataclasses
import inspect

import bluesky as bs
import numpy as np
import pytest

import bluesky_sandbox.interface.fields.observations as observations
from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.core import services
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions
from bluesky_sandbox.interface.fields.base import ObsField, PairObsField
from bluesky_sandbox.interface.wrappers.observations import normalizer as nz
from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
from bluesky_sandbox.sim.queryables import Waypoint
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion


def _concrete(base: type) -> list[type]:
    return [
        obj
        for name in sorted(dir(observations))
        if not name.startswith("_")
        and inspect.isclass(obj := getattr(observations, name))
        and issubclass(obj, base)
        and obj not in (ObsField, PairObsField)
        and not inspect.isabstract(obj)
    ]


OWN_CLASSES = [c for c in _concrete(ObsField) if not issubclass(c, PairObsField)]
PAIR_CLASSES = _concrete(PairObsField)

# Composite fields wrap another field; the rest build with their defaults.
_EXTRA_KWARGS = {
    "Difference": lambda: {
        "left": observations.CasKts(),
        "right": observations.CasKts(),
    },
    "AngleDifference": lambda: {
        "left": observations.TrkDeg(),
        "right": observations.TrkDeg(),
    },
    "LaggedObs": lambda: {"inner": observations.CasKts(), "steps": 2},
    "LaggedPair": lambda: {"inner": observations.DistToOwnNm(), "steps": 2},
}

# Per-component bounds: no scalar normalizer applies (see test_normalizers).
_MULTI_OUTPUT = frozenset({"FlightPhaseOneHot", "PrevActionNorm"})

NORMALIZERS = {
    "MinMax": nz.MinMaxNormalizer(),
    "MinMaxClipped": nz.MinMaxNormalizer(clipped=True),
    "Symmetric": nz.SymmetricNormalizer(),
    "SignedPower": nz.SignedPowerNormalizer(power=3.0),
    "Power": nz.PowerNormalizer(power=2.0),
    "Circular": nz.CircularNormalizer(),
}


def _build(cls: type):
    return cls(**_EXTRA_KWARGS.get(cls.__name__, dict)())


class _Scenario:
    """20 aircraft on one route in a small box: conflicts and LoS by step 1."""

    def __init__(self, aircraft_type="B744") -> None:
        self.spawn = SpawnConfig(
            regions=[
                SpawnRegion(
                    bounds=RegionBounds(BoxFootprint(51.9, 52.1, 4.5, 4.9)),
                    n_aircraft=20,
                    params={"alt_ft": (8_000, 12_000), "spd_kts": (230, 290)},
                    route=["merge"],
                )
            ],
            aircraft_type=aircraft_type,
            conflict_free_spawn=False,
        )

    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(
            airspace_bounds=None,
            spawn=self.spawn,
            queryables={
                "merge": Waypoint(lat=52.0, lon=4.8, alt_ft=9_000, speed_kts=250)
            },
            max_aircraft=20,
        )


@pytest.fixture(scope="module")
def fields():
    """Every field bound to a live env, stepped with random actions.

    Configured as both ownship and intruder fields, so each is bound exactly
    as a real config would bind it (stateful ones included).
    """
    own = [_build(c) for c in OWN_CLASSES]
    pair = [_build(c) for c in PAIR_CLASSES]
    env = BlueskyEnv(
        scenario=_Scenario(),
        config=EnvConfig(
            dt=6.0,
            obs_fields=own,
            intruder_obs_fields=pair + own,
            action_fields=[
                actions.HdgDeltaDeg(low=-30, high=30),
                actions.SpdDeltaKts(low=-10, high=10),
            ],
        ),
    )
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    for _ in range(4):  # past the lag fields' depth, into conflicts
        env.step({a: rng.uniform(-1, 1, 2).astype(np.float32) for a in bs.traf.id})
    # Guard against a vacuous pass: the geometry the pair fields describe exists.
    assert bs.traf.ntraf >= 10
    assert len(bs.traf.cd.confpairs) > 0
    by_name = {type(f).__name__: f for f in [*env.config.obs_fields, *pair]}
    by_name.update(
        (type(f).__name__, f)
        for f in env.config.intruder_obs_fields
        if isinstance(f, PairObsField)
    )
    yield by_name
    env.close()


def _assert_same(batched, single, what: str) -> None:
    b = np.asarray(batched, dtype=np.float32)
    s = np.asarray(single, dtype=np.float32)
    assert b.shape == s.shape, f"{what}: shape {b.shape} vs {s.shape}"
    np.testing.assert_array_equal(b, s, err_msg=what)


def _all_indices() -> tuple[int, ...]:
    return tuple(range(bs.traf.ntraf))


def _others(own: int) -> tuple[int, ...]:
    return tuple(j for j in _all_indices() if j != own)


@pytest.mark.parametrize("cls", OWN_CLASSES, ids=lambda c: c.__name__)
def test_get_many_matches_get(fields, cls):
    field = fields[cls.__name__]
    batched = field.get_many(_all_indices())
    for idx in _all_indices():
        _assert_same(batched[idx], field.get(idx), f"aircraft {idx}")


@pytest.mark.parametrize("cls", PAIR_CLASSES, ids=lambda c: c.__name__)
def test_get_pairs_matches_get_pair(fields, cls):
    field = fields[cls.__name__]
    for own in _all_indices():
        others = _others(own)
        batched = field.get_pairs(own, others)
        for k, other in enumerate(others):
            _assert_same(
                batched[k], field.get_pair(own, other), f"own {own}, other {other}"
            )


@pytest.mark.parametrize("cls", PAIR_CLASSES, ids=lambda c: c.__name__)
def test_get_pair_matrix_matches_get_pairs(fields, cls):
    """The assembler reads each ownship's row of one matrix per field."""
    field = fields[cls.__name__]
    owns = np.array(_all_indices())
    matrix = np.asarray(field.get_pair_matrix(owns))
    assert matrix.shape[:2] == (len(owns), bs.traf.ntraf)
    for row, own in enumerate(owns):
        others = np.array(_others(int(own)))
        _assert_same(
            matrix[row][others], field.get_pairs(int(own), others), f"own {own}"
        )
    # A subset of ownships reads the same rows.
    subset = owns[1::3]
    _assert_same(field.get_pair_matrix(subset), matrix[1::3], "ownship subset")


def _normalized_field(field, normalizer):
    """The field with ``normalizer`` attached, or None if they don't pair."""
    if type(field).__name__ in _MULTI_OUTPUT:
        return None
    try:
        attached = dataclasses.replace(field, normalizer=normalizer)
        normalizer.output_bounds(attached)  # Circular refuses non-angles here
    except (TypeError, ValueError):
        return None
    return attached


@pytest.mark.parametrize("name", sorted(NORMALIZERS))
@pytest.mark.parametrize("cls", OWN_CLASSES + PAIR_CLASSES, ids=lambda c: c.__name__)
def test_batch_normalization_matches_one_at_a_time(fields, cls, name):
    """The intruder block normalizes a batch; an ownship row normalizes one
    value. The same aircraft must read the same in both."""
    field = _normalized_field(fields[cls.__name__], NORMALIZERS[name])
    if field is None:
        pytest.skip(f"{name} does not apply to {cls.__name__}")
    for own in (0, bs.traf.ntraf // 2):
        others = _others(own)
        if isinstance(field, PairObsField):
            raw = field.get_pairs(own, others)
        else:
            raw = np.asarray(field.get_many(_all_indices()))[np.asarray(others)]
        batched = services._normalize_field_values_batch(field, raw, own)
        single = [
            services._normalize_field_value(field, raw[k], own)
            for k in range(len(others))
        ]
        _assert_same(batched, single, f"own {own}")


def test_the_intruder_block_matches_per_ownship_assembly():
    """The assembler builds each field once for all ownships - a pair matrix,
    and one normalization when bounds do not vary by aircraft. Pin it to the
    straightforward way: each ownship's intruders, normalized at its bounds."""
    names = sorted(NORMALIZERS)
    fields = []
    for i, cls in enumerate(PAIR_CLASSES + OWN_CLASSES):
        attached = _normalized_field(_build(cls), NORMALIZERS[names[i % len(names)]])
        fields.append(_build(cls) if attached is None else attached)
    # Mixed types, so envelope bounds really differ between ownships.
    types = ["B744", "A320"]
    env = BlueskyEnv(
        scenario=_Scenario(aircraft_type=types),
        config=EnvConfig(
            dt=6.0,
            allowed_aircraft=types,
            obs_fields=[observations.CasKts()],
            intruder_obs_fields=fields,
            action_fields=[],
        ),
    )
    try:
        env.reset(seed=0)
        for _ in range(3):
            env.step({})
        assembler = env._observation_assembler
        bound = env.config.intruder_obs_fields
        for obs_ in (assembler.get_obs(), assembler.get_obs(list(bs.traf.id)[::4])):
            for acid, agent_obs in obs_.items():
                own = list(bs.traf.id).index(acid)
                others = np.array(_others(own), dtype=np.intp)
                every = np.arange(bs.traf.ntraf)
                columns = []
                for field in bound:
                    if isinstance(field, PairObsField):
                        raw = field.get_pairs(own, others)
                    else:
                        raw = np.asarray(field.get_many(every))[others]
                    columns.append(
                        services._normalize_field_values_batch(field, raw, own)
                    )
                _assert_same(agent_obs["intruders"], np.hstack(columns), acid)
    finally:
        env.close()


def test_comm_noise_is_drawn_in_the_same_order_either_way(fields, monkeypatch):
    """Receiver noise comes from one seeded stream: the matrix must consume it
    exactly as one get_pairs call per ownship would, or seeded runs change."""
    from bluesky_sandbox.interface.fields import _state  # noqa: PLC0415

    field = observations.IntruderCommMessage(noise_std=0.3)
    for i in range(bs.traf.ntraf):
        _state.record_comm_message(i, 0, 0.1 * (i % 7) - 0.3)
    owns = np.array(_all_indices()[::2])

    monkeypatch.setattr(_state, "_COMM_NOISE_RNG", np.random.default_rng(5))
    matrix = field.get_pair_matrix(owns)
    monkeypatch.setattr(_state, "_COMM_NOISE_RNG", np.random.default_rng(5))
    for row, own in enumerate(owns):
        others = np.array(_others(int(own)))
        np.testing.assert_array_equal(
            matrix[row][others], field.get_pairs(int(own), others)
        )
    assert np.ptp(matrix[~np.isnan(matrix)]) > 0.1, "the noise really was on"
