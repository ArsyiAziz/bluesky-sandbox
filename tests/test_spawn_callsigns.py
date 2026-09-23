"""A callsign is issued at most once per episode.

BlueSky refuses only a *live* duplicate callsign, so a new aircraft can take the
callsign of one deleted earlier. Everything keyed by callsign then mistakes it
for that aircraft: the substep monitors hand it the old aircraft's conflict and
dwell history, and a trainer sees a finished agent's name come back.
"""

from __future__ import annotations

import re

import bluesky as bs
import numpy as np
import pytest

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.core.spawning import _CallsignIssuer
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
from bluesky_sandbox.sim.sampling.distributions import Categorical
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion


class _Scenario:
    """A steady 10-aircraft region drawing every callsign from one prefix."""

    def __init__(self) -> None:
        self.spawn = SpawnConfig(
            regions=[
                SpawnRegion(
                    bounds=RegionBounds(BoxFootprint(51.0, 53.0, 3.5, 6.0)),
                    n_aircraft=10,
                    params={"alt_ft": (8_000, 30_000), "spd_kts": (230, 270)},
                    callsign_prefixes=["KL"],
                    maintain=True,
                )
            ],
            aircraft_type="B744",
            conflict_free_spawn=False,
        )

    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(
            airspace_bounds=None, spawn=self.spawn, queryables={}, max_aircraft=10
        )


class _ClearsTheSkyEachStep(BlueskyEnv):
    """Deletes every aircraft on each step's first substep; maintain refills."""

    def on_before_step(self) -> None:
        self._clear_now = True

    def on_sim_step(self) -> None:
        if getattr(self, "_clear_now", False) and bs.traf.ntraf:
            bs.traf.delete(list(range(bs.traf.ntraf)))
            self._clear_now = False


def test_no_callsign_is_created_twice_in_an_episode():
    env = _ClearsTheSkyEachStep(
        scenario=_Scenario(), config=EnvConfig(dt=12.0, obs_fields=[], action_fields=[])
    )
    created: list[str] = []
    runtime = env._runtime
    original = runtime.create_aircraft

    def logging_create(callsign, *args, **kwargs):
        created.append(callsign)
        return original(callsign, *args, **kwargs)

    runtime.create_aircraft = logging_create
    try:
        for seed in (0, 1):
            created.clear()
            env.reset(seed=seed)
            for _ in range(40):
                env.step({})
            # ~400 spawns from the 999 "KL" numbers: without the rule, reuse of
            # a deleted callsign is all but certain.
            assert len(created) > 300
            assert len(created) == len(set(created))
    finally:
        env.close()


def _issuer(seed: int = 0) -> _CallsignIssuer:
    issuer = _CallsignIssuer()
    issuer.start_episode(np.random.default_rng(seed))
    return issuer


def test_a_prefix_issues_all_999_callsigns_in_random_order_then_refuses():
    issuer = _issuer()
    # Nothing is ever live (``live`` empty): each aircraft is gone by the time
    # the next is named, which is exactly when reuse used to happen.
    issued = [issuer.issue(set(), "KL") for _ in range(999)]
    assert sorted(issued) == [f"KL{n:03d}" for n in range(1, 1000)]
    assert issued[:20] != sorted(issued[:20])  # random-looking, not a counter
    # Checked before asking again: a miscounted prefix would otherwise reject
    # draws forever instead of failing.
    assert not issuer._has_left("KL")
    with pytest.raises(RuntimeError, match=r"\['KL'\].*999"):
        issuer.issue(set(), "KL")


def test_a_live_callsign_the_spawner_did_not_issue_is_avoided_and_kept_reserved():
    issuer = _issuer()
    live = {f"KL{n:03d}" for n in range(1, 999)}  # all but KL999
    assert issuer.issue(live, "KL") == "KL999"
    # Those aircraft are gone now, but their callsigns stay out of reach.
    with pytest.raises(RuntimeError, match="'KL'"):
        issuer.issue(set(), "KL")


def test_a_full_prefix_hands_over_to_the_regions_others_by_weight():
    issuer = _issuer()
    options = Categorical({"BA": 1.0, "KL": 9.0, "DL": 1.0})
    for _ in range(999):
        issuer.issue(set(), "BA", options)
    # The region keeps drawing BA; BA is full, so KL and DL stand in at 9:1.
    stand_ins = [issuer.issue(set(), "BA", options)[:2] for _ in range(400)]
    assert set(stand_ins) == {"KL", "DL"}
    assert 0.82 < stand_ins.count("KL") / len(stand_ins) < 0.97


def test_a_region_runs_dry_only_when_every_prefix_has():
    issuer = _issuer()
    options = ["BA", "KL"]
    issued = [issuer.issue(set(), "BA", options) for _ in range(2 * 999)]
    assert len(set(issued)) == 2 * 999
    with pytest.raises(RuntimeError, match=r"\['BA', 'KL'\]"):
        issuer.issue(set(), "BA", options)


def test_prefixes_that_cannot_be_listed_have_no_stand_in():
    class _Custom:  # a TypeDistribution whose support is unknown
        def rvs(self, random_state=None):
            return "BA"

    issuer = _issuer()
    for _ in range(999):
        issuer.issue(set(), "BA", _Custom())
    with pytest.raises(RuntimeError, match="cannot be listed"):
        issuer.issue(set(), "BA", _Custom())


def test_random_letter_callsigns_are_three_letters_and_a_flight_number():
    issuer = _issuer()
    issued = [issuer.issue(set()) for _ in range(500)]
    assert all(re.fullmatch(r"[A-Z]{3}(00[1-9]|0[1-9]\d|[1-9]\d\d)", c) for c in issued)
    assert len(set(issued)) == 500


def test_an_issuer_needs_an_episode():
    with pytest.raises(RuntimeError, match="start_episode"):
        _CallsignIssuer().issue(set())


def test_naming_never_moves_traffic(monkeypatch):
    env = BlueskyEnv(
        scenario=_Scenario(), config=EnvConfig(dt=12.0, obs_fields=[], action_fields=[])
    )
    try:
        env.reset(seed=0)
        names = list(bs.traf.id)
        where = (bs.traf.lat.copy(), bs.traf.lon.copy(), bs.traf.alt.copy())

        # Burn naming draws at episode start: different names, same traffic.
        original = _CallsignIssuer.start_episode

        def start_then_burn(self, rng):
            original(self, rng)
            self._rng.integers(1000, size=7)

        monkeypatch.setattr(_CallsignIssuer, "start_episode", start_then_burn)
        env.reset(seed=0)
        assert list(bs.traf.id) != names
        for before, after in zip(where, (bs.traf.lat, bs.traf.lon, bs.traf.alt)):
            np.testing.assert_array_equal(before, after)
    finally:
        env.close()


def test_a_new_episode_may_issue_the_same_callsigns_again():
    env = BlueskyEnv(
        scenario=_Scenario(), config=EnvConfig(dt=12.0, obs_fields=[], action_fields=[])
    )
    try:
        env.reset(seed=0)
        first = set(bs.traf.id)
        env.reset(seed=0)
        assert set(bs.traf.id) == first  # same seed, same names: nothing carried over
    finally:
        env.close()
