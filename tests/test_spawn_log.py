"""The spawn log: each aircraft of the episode as it was created."""

from __future__ import annotations

import bluesky as bs
import pytest
from bluesky.tools.aero import ft, kts

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
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
