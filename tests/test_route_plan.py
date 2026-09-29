"""The route plan: the level and speed an aircraft flies each leg at, unless
cleared otherwise - and a route-relative action measured from it.

With ``nominal_from_plan``, a zero delta on a fix with no gate means "back to
the plan", so a temporary deviation ends with one clearance of 0. Without it,
zero means "stay as you are".
"""

from __future__ import annotations

import bluesky as bs
import numpy as np
import pytest
from bluesky.tools.aero import ft, kts

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
    "gated": Waypoint(lat=52.0, lon=6.0, alt_ft=15_000, speed_kts=230),
    "exit": Waypoint(lat=52.0, lon=7.0),
}


class _Scenario:
    def __init__(self, route) -> None:
        self.spawn = SpawnConfig(
            regions=[
                SpawnRegion(
                    bounds=RegionBounds(BoxFootprint(51.99, 52.01, 4.0, 4.01)),
                    n_aircraft=1,
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
            max_aircraft=1,
        )


def _env(from_plan: bool, route=("free",)) -> BlueskyEnv:
    return BlueskyEnv(
        scenario=_Scenario(list(route)),
        config=EnvConfig(
            dt=5.0,
            obs_fields=[obs.CasKts()],
            action_fields=[
                act.ActiveRouteWaypointSpdDeltaKts(nominal_from_plan=from_plan),
                act.ActiveRouteWaypointAltDeltaFt(nominal_from_plan=from_plan),
            ],
        ),
    )


def _clear(env, agent, spd_kts: float, alt_ft: float) -> None:
    env.step({agent: np.array([spd_kts, alt_ft], np.float32)})


def _selected(agent) -> tuple[float, float]:
    i = bs.traf.id.index(agent)
    return float(bs.traf.selspd[i]) / kts, float(bs.traf.selalt[i]) / ft


@pytest.mark.parametrize("from_plan", [True, False], ids=["plan", "current"])
def test_zero_returns_to_the_plan_or_stays_as_it_is(from_plan):
    env = _env(from_plan)
    try:
        observations, _ = env.reset(seed=0)
        (agent,) = observations
        _clear(env, agent, 30.0, 3_000.0)  # a temporary deviation
        spd, alt = _selected(agent)
        assert spd == pytest.approx(280.0, abs=1.0)
        assert alt == pytest.approx(23_000.0, abs=1.0)
        for _ in range(3):  # partly flown
            _clear(env, agent, 30.0, 3_000.0)
        i = bs.traf.id.index(agent)
        now_kts, now_ft = float(bs.traf.cas[i]) / kts, float(bs.traf.alt[i]) / ft
        _clear(env, agent, 0.0, 0.0)
        spd, alt = _selected(agent)
        if from_plan:
            assert spd == pytest.approx(250.0, abs=1.0)
            assert alt == pytest.approx(20_000.0, abs=1.0)
        else:
            assert spd == pytest.approx(now_kts, abs=1.0)
            assert alt == pytest.approx(now_ft, abs=1.0)
    finally:
        env.close()


def test_a_gated_fix_is_still_the_reference_on_its_own_leg():
    env = _env(True, route=("gated", "exit"))
    try:
        observations, _ = env.reset(seed=0)
        (agent,) = observations
        _clear(env, agent, 0.0, 0.0)
        spd, alt = _selected(agent)
        assert spd == pytest.approx(230.0, abs=1.0)
        assert alt == pytest.approx(15_000.0, abs=1.0)
    finally:
        env.close()


def test_a_gate_is_carried_on_to_the_legs_after_it():
    env = _env(True, route=("gated", "exit"))
    try:
        env.reset(seed=0)
        plan = planned_legs(
            0, [_QUERYABLES["gated"].target, _QUERYABLES["exit"].target]
        )
        (level0, cas0), (level1, cas1) = plan
        assert level0 / ft == pytest.approx(20_000.0, abs=1.0)
        assert cas0 / kts == pytest.approx(250.0, abs=1.0)
        assert level1 / ft == pytest.approx(15_000.0)
        assert cas1 / kts == pytest.approx(230.0)
    finally:
        env.close()
