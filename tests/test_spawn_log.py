"""The spawn log: each aircraft of the episode as it was created."""

from __future__ import annotations

import bluesky as bs
import pytest
from bluesky.tools.aero import ft, kts

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
from bluesky_sandbox.sim.queryables import Waypoint
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion


class _Scenario:
    def __init__(self) -> None:
        self.spawn = SpawnConfig(
            regions=[
                SpawnRegion(
                    bounds=RegionBounds(BoxFootprint(51.0, 53.0, 3.5, 6.0)),
                    n_aircraft=4,
                    params={"alt_ft": (8_000, 30_000), "spd_kts": (230, 270), "hdg_deg": 45.0},
                    callsign_prefixes=["KL"],
                )
            ],
            aircraft_type="B744",
            conflict_free_spawn=False,
        )

    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(airspace_bounds=None, spawn=self.spawn, queryables={}, max_aircraft=4)


@pytest.fixture
def env():
    env = BlueskyEnv(scenario=_Scenario(), config=EnvConfig(dt=5.0, obs_fields=[], action_fields=[]))
    yield env
    env.close()


def test_each_aircraft_is_logged_as_created(env):
    env.reset(seed=0)
    log = env.spawn_log
    assert len(log) == 4 and len({r.callsign for r in log}) == 4
    for record in log:
        assert record.callsign.startswith("KL") and record.actype == "B744"
        assert record.hdg_deg == pytest.approx(45.0)
        assert 8_000 <= record.alt_ft <= 30_000
        assert 230 <= record.cas_kts <= 270 + 1e-6
        idx = bs.traf.id.index(record.callsign)
        assert record.lat_deg == pytest.approx(float(bs.traf.lat[idx]))
        assert record.alt_ft == pytest.approx(float(bs.traf.alt[idx] / ft))
        assert record.gs_kts == pytest.approx(float(bs.traf.gs[idx] / kts))


def test_a_reset_starts_a_new_log(env):
    env.reset(seed=0)
    first = [r.callsign for r in env.spawn_log]
    env.reset(seed=1)
    assert len(env.spawn_log) == 4 and [r.callsign for r in env.spawn_log] != first


def test_a_route_is_logged_with_its_resolved_constraints():
    scenario = _Scenario()
    region = scenario.spawn.regions[0]
    scenario.spawn = type(scenario.spawn)(
        regions=[type(region)(**{**region.__dict__, "route": [{"waypoint": "fix", "alt_ft": 12_000, "speed_kts": 250}]})],
        aircraft_type="B744",
        conflict_free_spawn=False,
    )
    fix = Waypoint(lat=52.0, lon=4.5, alt_tolerance_ft=500, speed_tolerance_kts=10, reach_radius_nm=3)
    support = scenario.support

    def with_fix():
        spec = support()
        return type(spec)(**{**spec.__dict__, "queryables": {"fix": fix}})

    scenario.support = with_fix
    env = BlueskyEnv(scenario=scenario, config=EnvConfig(dt=5.0, obs_fields=[], action_fields=[]))
    try:
        env.reset(seed=0)
        (target,) = env.spawn_log[0].targets
        assert env.spawn_log[0].route == ("fix",)
        assert (target.lat, target.lon, target.alt_ft, target.speed_kts) == (52.0, 4.5, 12_000, 250)
        assert (target.alt_tolerance_ft, target.speed_tolerance_kts, target.reach_radius_nm) == (500, 10, 3)
    finally:
        env.close()
