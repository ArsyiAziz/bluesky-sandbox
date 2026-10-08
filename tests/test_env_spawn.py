"""Spawning from code: ``env.spawn`` / ``env.spawn_from`` create an aircraft
now, the way the episode's own spawns are - callsign, route, hooks, control
state, spawn log - so one made in code cannot differ from one the loop makes."""

from __future__ import annotations

import bluesky as bs
import pytest
from bluesky.tools.aero import ft

from bluesky_sandbox import AircraftControlState
from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.sim.bounds import BoxFootprint, LatLon, RegionBounds
from bluesky_sandbox.sim.queryables import Waypoint
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion


class _Scenario:
    """One aircraft from region ``west`` at reset; room for more from code."""

    def __init__(self) -> None:
        self.spawn = SpawnConfig(
            regions=[
                SpawnRegion(
                    bounds=RegionBounds(BoxFootprint(51.8, 52.2, 3.6, 3.9)),
                    n_aircraft=1,
                    params={"alt_ft": 10_000, "spd_kts": 250, "hdg_deg": 90.0},
                    callsign_prefixes=["KL"],
                    name="west",
                    route=["fix"],
                )
            ],
            routes={"to_fix": ["fix"]},
            aircraft_type="B744",
            conflict_free_spawn=False,
        )

    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(
            airspace_bounds=None,
            spawn=self.spawn,
            queryables={"fix": Waypoint(lat=52.0, lon=4.5)},
            max_aircraft=self.spawn.max_aircraft(),
        )


class _Env(BlueskyEnv):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.spawned: list[tuple[str, list[str] | None]] = []

    def on_aircraft_spawned(self, callsign, route):
        self.spawned.append((callsign, route))


@pytest.fixture
def env():
    env = _Env(scenario=_Scenario(), config=EnvConfig(dt=5.0, obs_fields=[], action_fields=[]))
    env.reset(seed=0)
    yield env
    env.close()


def test_an_aircraft_spawned_in_code_is_made_as_the_loops_are(env):
    acid = env.spawn(LatLon(52.0, 4.0), alt_ft=12_000, spd_kts=240, hdg_deg=90, route="to_fix")
    record = env.spawn_log[-1]
    assert record.callsign == acid and record.actype == "B744" and record.route == ("fix",)
    assert record.alt_ft == pytest.approx(12_000) and record.hdg_deg == pytest.approx(90)
    assert (acid, ["fix"]) in env.spawned  # the hook saw it
    assert acid in env.agents  # controlled, by the default control-state hook
    idx = bs.traf.id.index(acid)
    assert float(bs.traf.alt[idx] / ft) == pytest.approx(12_000)


def test_a_chosen_callsign_is_taken_and_not_reused(env):
    assert env.spawn((52.0, 4.2), alt_ft=9_000, spd_kts=230, callsign="test1") == "TEST1"
    with pytest.raises(ValueError, match="already in use"):
        env.spawn((52.0, 4.3), alt_ft=9_000, spd_kts=230, callsign="TEST1")


def test_background_traffic_is_not_an_agent(env):
    acid = env.spawn((52.1, 4.1), alt_ft=15_000, spd_kts=260, controlled=False)
    assert acid not in env.agents
    assert dict(env.aircraft_control_states())[acid] is AircraftControlState.BACKGROUND


def test_its_speed_and_heading_may_be_drawn(env):
    acid = env.spawn((52.1, 4.2), alt_ft=20_000)
    record = env.spawn_log[-1]
    assert record.callsign == acid and record.cas_kts > 100 and 0 <= record.hdg_deg < 360


def test_conflict_free_spawns_only_where_it_is_clear(env):
    first = env.spawn_log[0]
    on_top = env.spawn(
        (first.lat_deg, first.lon_deg), alt_ft=first.alt_ft, spd_kts=250, hdg_deg=270, conflict_free=True
    )
    assert on_top is None
    clear = env.spawn((53.5, 6.0), alt_ft=30_000, spd_kts=250, hdg_deg=0, conflict_free=True)
    assert clear is not None


def test_a_region_spawns_one_more_as_its_own(env):
    acid = env.spawn_from("west")
    record = env.spawn_log[-1]
    assert record.callsign == acid and acid.startswith("KL")
    assert record.region_index == 0 and record.route == ("fix",)
    with pytest.raises(KeyError, match="not in the design"):
        env.spawn_from("east")


def test_spawning_mid_episode(env):
    env.step({})
    acid = env.spawn((52.0, 4.4), alt_ft=11_000, spd_kts=250, hdg_deg=180)
    assert acid in env.agents and env.spawn_log[-1].time_s > 0


def test_without_a_cap_code_spawns_freely():
    scenario = _Scenario()
    env = _Env(scenario=scenario, config=EnvConfig(dt=5.0, obs_fields=[], action_fields=[]))
    try:
        env.reset(seed=0)
        assert env.episode_max_aircraft == 1  # what the region produces
        for i in range(3):
            assert env.spawn((52.0, 4.0 + 0.1 * i), alt_ft=9_000, spd_kts=230) is not None
        assert len(bs.traf.id) == 4
    finally:
        env.close()


def test_a_cap_is_the_episodes_max_and_code_stays_within_it():
    scenario = _Scenario()
    scenario.spawn.aircraft_cap = 2
    env = _Env(scenario=scenario, config=EnvConfig(dt=5.0, obs_fields=[], action_fields=[]))
    try:
        env.reset(seed=0)
        assert env.episode_max_aircraft == 2  # one from the region, room for one from code
        assert env.spawn((52.0, 4.2), alt_ft=9_000, spd_kts=230) is not None
        with pytest.raises(RuntimeError, match="aircraft_cap"):
            env.spawn((52.0, 4.3), alt_ft=9_000, spd_kts=230)
    finally:
        env.close()


def test_padding_past_its_rows_is_an_error_not_a_drop():
    from bluesky_sandbox.interface.fields import observations as obs
    from bluesky_sandbox.interface.wrappers.observations.pad import IntruderPaddingWrapper

    config = EnvConfig(dt=5.0, obs_fields=[obs.CasKts()], intruder_obs_fields=[obs.CasKts()], action_fields=[])
    env = IntruderPaddingWrapper(_Env(scenario=_Scenario(), config=config), max_intruders=1)
    try:
        env.reset(seed=0)
        env.unwrapped.spawn((52.0, 4.2), alt_ft=9_000, spd_kts=230)
        env.step({})  # two aircraft: one intruder each - fits
        env.unwrapped.spawn((52.0, 4.3), alt_ft=9_000, spd_kts=230)
        with pytest.raises(RuntimeError, match="padding rows"):
            env.step({})
    finally:
        env.close()


def test_a_cap_below_the_regions_thins_their_traffic():
    from dataclasses import replace

    scenario = _Scenario()
    region = replace(scenario.spawn.regions[0], n_aircraft=4)
    scenario.spawn.regions = [region]
    scenario.spawn.aircraft_cap = 2
    env = _Env(scenario=scenario, config=EnvConfig(dt=5.0, obs_fields=[], action_fields=[]))
    try:
        env.reset(seed=0)
        assert env.episode_max_aircraft == 2 and len(env.spawn_log) == 2
    finally:
        env.close()
