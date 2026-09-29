"""Route labels show each fix as the library reads it: a speed its arrival time
implies (marked TBO) where it has no gate, and how late or early the aircraft
would be over it."""

from __future__ import annotations

import numpy as np
import pytest

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
from bluesky_sandbox.sim.queryables import Waypoint
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion

pytest.importorskip("pygame")

_QUERYABLES = {
    "timed": Waypoint(lat=52.0, lon=6.0),
    "gated": Waypoint(lat=52.0, lon=7.0, alt_ft=20_000, speed_kts=230),
}


class _Scenario:
    def sample(self, _rng):
        return self.support()

    def support(self):
        region = SpawnRegion(
            bounds=RegionBounds(BoxFootprint(51.999, 52.001, 4.0, 4.001)),
            n_aircraft=1,
            params={
                "alt_ft": (20_000, 20_000),
                "spd_kts": (250, 250),
                "hdg_deg": (90, 90),
            },
            route=[{"waypoint": "timed", "arrival_slack_s": 120.0}, "gated"],
        )
        spawn = SpawnConfig(
            regions=[region], aircraft_type="B744", conflict_free_spawn=False
        )
        return EpisodeSpec(
            airspace_bounds=None, spawn=spawn, queryables=_QUERYABLES, max_aircraft=1
        )


@pytest.fixture
def flying():
    env = BlueskyEnv(
        scenario=_Scenario(),
        render_mode="rgb_array",
        config=EnvConfig(
            dt=5.0,
            obs_fields=[obs.ActiveRouteWaypointArrivalErrorS()],
            action_fields=[act.ActiveRouteWaypointSpdDeltaKts()],
        ),
    )
    observations, _ = env.reset(seed=0)
    (agent,) = observations
    observations, *_ = env.step({agent: np.array([0.0], np.float32)})
    yield env, agent, observations[agent]
    env.close()


def test_a_timed_fix_shows_its_schedule_speed_and_arrival_error(flying):
    env, agent, seen = flying
    timed, gated = env._driver.aircraft_route_waypoints(agent)
    assert timed["scheduled"] and timed["speed_kts"] < 245.0
    assert timed["arrival_error_s"] == pytest.approx(float(seen[0]), abs=1e-3)
    lines = env._driver.format_waypoint_speed_lines(
        0, target_kts=timed["speed_kts"], scheduled=True
    )
    assert lines[0].endswith(" TBO")
    assert env._driver.format_waypoint_time_lines(timed)[0].startswith("TIME ")
    # A gated fix shows its gate, and no time where it has none.
    assert not gated["scheduled"] and gated["speed_kts"] == pytest.approx(230.0)
    assert gated["arrival_error_s"] is None
    assert env._driver.format_waypoint_time_lines(gated) == []


def test_the_labels_render(flying):
    env, agent, _ = flying
    env._driver._selected = agent
    frame = env.render()
    assert frame.ndim == 3 and frame.any()
