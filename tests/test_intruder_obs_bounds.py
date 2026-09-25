"""``EnvConfig.intruder_obs_bounds``: whose envelope scales an intruder row.

A non-pair field with per-aircraft bounds (``CasKts``: the performance model's
speed envelope) can scale an intruder by the observing aircraft's envelope or
by the intruder's own. With one aircraft type the two agree; these tests mix a
B744 and an A320, whose envelopes differ.
"""

from __future__ import annotations

import bluesky as bs
import numpy as np
import pytest

import bluesky_sandbox.interface.fields.observations as obs
from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.wrappers.observations import normalizer as nz
from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion

_TYPES = ["B744", "A320"]


class _Scenario:
    def __init__(self) -> None:
        self.spawn = SpawnConfig(
            regions=[
                SpawnRegion(
                    bounds=RegionBounds(BoxFootprint(51.0, 53.0, 3.5, 6.0)),
                    n_aircraft=8,
                    params={"alt_ft": (20_000, 30_000), "spd_kts": (250, 290)},
                )
            ],
            aircraft_type=_TYPES,
            conflict_free_spawn=False,
        )

    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(
            airspace_bounds=None, spawn=self.spawn, queryables={}, max_aircraft=8
        )


def _obs(mode: str | None, field) -> dict:
    kwargs = {} if mode is None else {"intruder_obs_bounds": mode}
    env = BlueskyEnv(
        scenario=_Scenario(),
        config=EnvConfig(
            dt=12.0,
            allowed_aircraft=_TYPES,
            obs_fields=[field],
            intruder_obs_fields=[field],
            action_fields=[],
            **kwargs,
        ),
    )
    try:
        obs_, _info = env.reset(seed=0)
        assert set(bs.traf.type) == set(_TYPES), "the test needs both types"
        vmin = np.asarray(bs.traf.perf.vmin[: bs.traf.ntraf])
        assert np.ptp(vmin) > 1.0, "the envelopes must differ for this to test anything"
        return {
            "obs": obs_,
            "ids": list(bs.traf.id),
            "bounds": [field.bounds(i) for i in range(bs.traf.ntraf)],
            "cas": np.asarray(bs.traf.cas, dtype=np.float64) / 0.514444,
        }
    finally:
        env.close()


def _intruder_row(result, own: str, other: str) -> float:
    ids = result["ids"]
    others = [a for a in ids if a != own]
    return float(result["obs"][own]["intruders"][others.index(other)][0])


def test_the_default_scales_intruders_by_the_ownship_envelope():
    result = _obs(None, obs.CasKts(normalizer=nz.MinMaxNormalizer()))
    ids = result["ids"]
    own, other = ids[0], ids[1]
    lo, hi = result["bounds"][0]
    expected = (result["cas"][1] - lo) / (hi - lo)
    assert _intruder_row(result, own, other) == pytest.approx(expected, abs=1e-5)


def test_intruder_bounds_make_a_row_match_that_aircrafts_own_values():
    result = _obs("intruder", obs.CasKts(normalizer=nz.MinMaxNormalizer()))
    ids = result["ids"]
    for own in ids:
        for other in ids:
            if other != own:
                own_row = float(result["obs"][other]["ownship"][0])
                assert _intruder_row(result, own, other) == own_row


def test_the_two_modes_differ_only_where_envelopes_differ():
    field = obs.CasKts(normalizer=nz.MinMaxNormalizer())
    by_own, by_intruder = _obs("ownship", field), _obs("intruder", field)
    ids = by_own["ids"]
    for i, own in enumerate(ids):
        for j, other in enumerate(ids):
            if other == own:
                continue
            same_envelope = by_own["bounds"][i] == by_own["bounds"][j]
            same_value = _intruder_row(by_own, own, other) == _intruder_row(
                by_intruder, own, other
            )
            assert same_value == same_envelope


def test_fixed_bounds_make_the_setting_irrelevant():
    field = obs.CasKts(low=100.0, high=400.0, normalizer=nz.MinMaxNormalizer())
    by_own, by_intruder = _obs("ownship", field), _obs("intruder", field)
    for own in by_own["ids"]:
        np.testing.assert_array_equal(
            by_own["obs"][own]["intruders"], by_intruder["obs"][own]["intruders"]
        )


def test_an_unknown_setting_is_refused():
    with pytest.raises(ValueError, match="intruder_obs_bounds must be one of"):
        EnvConfig(intruder_obs_bounds="own")
