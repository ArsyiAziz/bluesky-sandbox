"""Target arrival times: when an aircraft is due over each fix of its route.

A route step with ``arrival_slack_s`` gives its fix a time, assigned at spawn:
the leg's nominal time - flown at the spawn speed, or its climb or descent at
the aircraft's own maximum rate, whichever is longer - plus the slack, never
sooner than the aircraft could fly it flat out. The ActiveRouteWaypoint arrival
fields read it back.
"""

from __future__ import annotations

import bluesky as bs
import numpy as np
import pytest
from bluesky.tools.aero import ft, kts, nm, vcas2tas
from bluesky.tools.geo import kwikqdrdist

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.sim.arrival import arrival_times
from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
from bluesky_sandbox.sim.queryables import Waypoint
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion

_QUERYABLES = {
    "fix": Waypoint(lat=52.0, lon=6.0),
    "exit": Waypoint(lat=52.0, lon=7.0),
}


class _Scenario:
    def __init__(self, route) -> None:
        self.spawn = SpawnConfig(
            regions=[
                SpawnRegion(
                    bounds=RegionBounds(BoxFootprint(51.9, 52.1, 4.0, 4.2)),
                    n_aircraft=3,
                    params={"alt_ft": (20_000, 20_000), "spd_kts": (250, 250)},
                    route=route,
                )
            ],
            aircraft_type="B744",
            conflict_free_spawn=False,
        )

    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(
            airspace_bounds=None,
            spawn=self.spawn,
            queryables=_QUERYABLES,
            max_aircraft=3,
        )


_FIELDS = [
    obs.ActiveRouteWaypointHasArrivalTime(),
    obs.ActiveRouteWaypointTimeToGoS(),
    obs.ActiveRouteWaypointArrivalErrorS(),
    obs.ActiveRouteWaypointEteS(),
    obs.ActiveRouteWaypointHasArrivalTime(route_offset=1),
    obs.ActiveRouteWaypointTimeToGoS(route_offset=1),
]


def _env(route) -> BlueskyEnv:
    return BlueskyEnv(
        scenario=_Scenario(route),
        config=EnvConfig(dt=5.0, obs_fields=_FIELDS, action_fields=[act.HdgDeltaDeg()]),
    )


def _read(observations) -> dict[str, np.ndarray]:
    rows = np.stack(list(observations.values()))
    names = ["has", "ttg", "error", "ete", "has_next", "ttg_next"]
    return dict(zip(names, rows.T))


@pytest.fixture(scope="module")
def timed():
    route = [
        {"waypoint": "fix", "arrival_slack_s": 0.0},
        {"waypoint": "exit", "arrival_slack_s": (30.0, 60.0)},
    ]
    env = _env(route)
    yield env
    env.close()


def test_a_fix_with_no_slack_is_due_at_its_nominal_time(timed):
    observations, _ = timed.reset(seed=0)
    seen = _read(observations)
    np.testing.assert_array_equal(seen["has"], 1.0)
    # Flown at the spawn speed, the aircraft arrives on time.
    np.testing.assert_allclose(seen["ttg"], seen["ete"], rtol=0.02)
    np.testing.assert_allclose(seen["error"], 0.0, atol=0.02 * seen["ete"].max())


def test_a_later_fix_is_due_after_its_leg_and_its_slack(timed):
    observations, _ = timed.reset(seed=0)
    seen = _read(observations)
    np.testing.assert_array_equal(seen["has_next"], 1.0)
    leg_s = seen["ttg_next"] - seen["ttg"]
    tas = float(vcas2tas(250 * kts, 20_000 * ft))
    nominal_s = kwikqdrdist(52.0, 6.0, 52.0, 7.0)[1] * nm / tas
    assert np.all(leg_s >= nominal_s + 30.0 - 1.0)
    assert np.all(leg_s <= nominal_s + 60.0 + 1.0)


def test_time_to_go_runs_down_with_the_clock(timed):
    observations, _ = timed.reset(seed=0)
    before = _read(observations)["ttg"]
    observations, *_ = timed.step({a: np.zeros(1, np.float32) for a in observations})
    np.testing.assert_allclose(_read(observations)["ttg"], before - 5.0, atol=1e-6)


def test_a_fix_with_no_slack_given_has_no_time():
    env = _env([{"waypoint": "fix"}, {"waypoint": "exit", "arrival_slack_s": 0.0}])
    try:
        seen = _read(env.reset(seed=0)[0])
        np.testing.assert_array_equal(seen["has"], 0.0)
        np.testing.assert_array_equal(seen["ttg"], 0.0)
        np.testing.assert_array_equal(seen["error"], 0.0)
        np.testing.assert_array_equal(seen["has_next"], 1.0)
    finally:
        env.close()


def test_no_fix_is_due_sooner_than_the_aircraft_can_fly_it():
    env = _env([{"waypoint": "fix", "arrival_slack_s": -1e6}])
    try:
        seen = _read(env.reset(seed=0)[0])
        # Due as early as flying flat out allows: late at the speed it flies.
        assert np.all(seen["ttg"] > 0.0) and np.all(seen["error"] > 0.0)
    finally:
        env.close()


def test_a_big_level_change_on_a_short_leg_sets_the_time(timed):
    timed.reset(seed=0)
    idx = 0
    near = Waypoint(lat=float(bs.traf.lat[idx]), lon=float(bs.traf.lon[idx]) + 0.05)
    deep = Waypoint(
        lat=near.lat, lon=near.lon, alt_ft=float(bs.traf.alt[idx]) / ft - 10_000
    )
    now = float(bs.sim.simt)
    rng = np.random.default_rng(0)
    (flat,) = arrival_times(idx, [near.target], [0.0], rng, now)
    (descend,) = arrival_times(idx, [deep.target], [0.0], rng, now)
    descent_s = 10_000 * ft / abs(float(bs.traf.perf.vsmin[idx]))
    assert flat - now < descent_s
    assert descend - now == pytest.approx(descent_s)
