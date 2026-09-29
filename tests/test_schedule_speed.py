"""A target arrival time is a speed constraint on its fix, only a derived one.

Where the fix has no speed gate but a time to be over it, its speed is the one
that still meets that time from where the aircraft is now - slower when early,
faster when late - so a route-relative speed action's zero means "on schedule"
and the speed observations read against it. A gated fix keeps its gate; no
arrival time, no speed.
"""

from __future__ import annotations

import bluesky as bs
import numpy as np
import pytest
from bluesky.tools.aero import ft, kts, nm, vtas2cas
from bluesky.tools.geo import kwikqdrdist

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.sim.arrival import planned_legs
from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
from bluesky_sandbox.sim.queryables import Waypoint
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion

_QUERYABLES = {
    "free": Waypoint(lat=52.0, lon=6.0),
    "gated": Waypoint(lat=52.0, lon=6.0, alt_ft=20_000, speed_kts=230),
    "exit": Waypoint(lat=52.0, lon=7.0),
}


class _Scenario:
    def __init__(self, route) -> None:
        self.spawn = SpawnConfig(
            regions=[
                SpawnRegion(
                    bounds=RegionBounds(BoxFootprint(51.999, 52.001, 4.0, 4.001)),
                    n_aircraft=1,
                    params={
                        "alt_ft": (20_000, 20_000),
                        "spd_kts": (250, 250),
                        "hdg_deg": (90, 90),  # the fix is dead ahead
                    },
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
            max_aircraft=1,
        )


def _env(route) -> BlueskyEnv:
    return BlueskyEnv(
        scenario=_Scenario(route),
        config=EnvConfig(
            dt=5.0,
            obs_fields=[
                obs.ActiveRouteWaypointHasSpdConstraint(),
                obs.ActiveRouteWaypointSpdDiffKts(),
            ],
            action_fields=[act.ActiveRouteWaypointSpdDeltaKts()],
        ),
    )


def _fly(route):
    env = _env(route)
    observations, _ = env.reset(seed=0)
    (agent,) = observations
    return env, agent


def _clear(env, agent, delta_kts: float, steps: int = 1) -> None:
    for _ in range(steps):
        env.step({agent: np.array([delta_kts], np.float32)})


def _selected_kts(agent) -> float:
    return float(bs.traf.selspd[bs.traf.id.index(agent)]) / kts


def _time_to_go(env, agent) -> float:
    i = bs.traf.id.index(agent)
    return float(obs.ActiveRouteWaypointTimeToGoS().get(i))


def _needed_cas_kts(agent, time_to_go_s: float) -> float:
    i = bs.traf.id.index(agent)
    _, dist_nm = kwikqdrdist(bs.traf.lat[i], bs.traf.lon[i], 52.0, 6.0)
    tas = dist_nm * nm / time_to_go_s + bs.traf.tas[i] - bs.traf.gs[i]
    return float(vtas2cas(tas, bs.traf.alt[i])) / kts


def test_on_schedule_zero_holds_the_speed():
    env, agent = _fly([{"waypoint": "free", "arrival_slack_s": 0.0}])
    try:
        _clear(env, agent, 0.0)
        assert _selected_kts(agent) == pytest.approx(250.0, abs=3.0)
    finally:
        env.close()


def test_early_zero_slows_to_the_speed_that_arrives_on_time():
    env, agent = _fly([{"waypoint": "free", "arrival_slack_s": 120.0}])
    try:
        ttg = _time_to_go(env, agent)
        needed = _needed_cas_kts(agent, ttg - 5.0)  # the step runs first
        _clear(env, agent, 0.0)
        assert _selected_kts(agent) < 240.0
        assert _selected_kts(agent) == pytest.approx(needed, abs=3.0)
    finally:
        env.close()


def test_after_a_slow_deviation_zero_catches_up():
    env, agent = _fly([{"waypoint": "free", "arrival_slack_s": 0.0}])
    try:
        _clear(env, agent, -60.0, steps=24)  # two minutes slow: now late
        _clear(env, agent, 0.0)
        assert _selected_kts(agent) > 255.0
    finally:
        env.close()


def test_a_gated_fix_keeps_its_gate():
    env, agent = _fly([{"waypoint": "gated", "arrival_slack_s": 120.0}, "exit"])
    try:
        _clear(env, agent, 0.0)
        assert _selected_kts(agent) == pytest.approx(230.0, abs=1.0)
    finally:
        env.close()


def test_the_observations_read_the_speed_off_schedule():
    env, agent = _fly([{"waypoint": "free", "arrival_slack_s": 120.0}])
    try:
        observations, *_ = env.step({agent: np.array([0.0], np.float32)})
        has, off_kts = observations[agent]
        i = bs.traf.id.index(agent)
        needed = _needed_cas_kts(agent, _time_to_go(env, agent))
        assert has == 1.0
        assert off_kts == pytest.approx(float(bs.traf.cas[i]) / kts - needed, abs=1.0)
    finally:
        env.close()


def test_with_no_arrival_time_zero_is_the_current_speed():
    env, agent = _fly(["free"])
    try:
        _clear(env, agent, -40.0, steps=6)
        i = bs.traf.id.index(agent)
        now = float(bs.traf.cas[i]) / kts
        _clear(env, agent, 0.0)
        assert _selected_kts(agent) == pytest.approx(now, abs=2.0)
    finally:
        env.close()


def test_arrival_times_plan_each_leg_from_the_gates_passed():
    env, _agent = _fly([{"waypoint": "gated", "arrival_slack_s": 0.0}, "exit"])
    try:
        targets = [_QUERYABLES["gated"].target, _QUERYABLES["exit"].target]
        (level0, cas0), (level1, cas1) = planned_legs(0, targets)
        assert level0 / ft == pytest.approx(20_000.0, abs=1.0)
        assert cas0 / kts == pytest.approx(250.0, abs=1.0)
        assert (level1 / ft, cas1 / kts) == pytest.approx((20_000.0, 230.0))
    finally:
        env.close()
