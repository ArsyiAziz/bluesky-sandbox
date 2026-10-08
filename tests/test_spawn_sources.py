"""Spawn sources: each plans its aircraft at reset - a replay of data, a
function, a mixture - and the environment spawns them at their times, as its
regions' are; each source's own clearance and route rules apply."""

from __future__ import annotations

import bluesky as bs
import pytest

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.sim.bounds import BoxFootprint, LatLon, RegionBounds
from bluesky_sandbox.sim.queryables import Waypoint
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import (
    Mixture,
    PlannedSource,
    RegionSource,
    Replay,
    SpawnConfig,
    SpawnRegion,
    SpawnRequest,
    nearest_entry,
)

WEST = LatLon(52.0, 3.6)
EAST = LatLon(52.0, 5.4)


class _Scenario:
    def __init__(self, *sources, cap=None) -> None:
        self.spawn = SpawnConfig(
            regions=[],
            routes={"west_in": ["w_fix", "goal"], "east_in": ["e_fix", "goal"]},
            aircraft_type="B744",
            sources=list(sources),
            aircraft_cap=cap,
        )

    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(
            airspace_bounds=None,
            spawn=self.spawn,
            queryables={
                "w_fix": Waypoint(lat=52.0, lon=3.8),
                "e_fix": Waypoint(lat=52.0, lon=5.2),
                "goal": Waypoint(lat=52.0, lon=4.5),
            },
            max_aircraft=self.spawn.max_aircraft(),
            shapes={"west": RegionBounds(BoxFootprint(51.9, 52.1, 3.5, 3.7))},
        )


def _env(*sources, cap=None):
    env = BlueskyEnv(scenario=_Scenario(*sources, cap=cap), config=EnvConfig(dt=5.0, obs_fields=[], action_fields=[]))
    env.reset(seed=0)
    return env


def _run(env, seconds):
    for _ in range(int(seconds / 5)):
        env.step({})


def test_a_replay_spawns_its_requests_at_their_times_in_its_window():
    tracks = [
        SpawnRequest(at=WEST, alt_ft=10_000, spd_kts=250, hdg_deg=90, callsign="ADS1", time_s=0),
        SpawnRequest(at=EAST, alt_ft=11_000, spd_kts=250, hdg_deg=270, callsign="ADS2", time_s=40),
        SpawnRequest(at=WEST, alt_ft=12_000, spd_kts=250, hdg_deg=90, callsign="ADS3", time_s=500),
    ]
    env = _env(Replay(tracks, window=(30, 300), time_scale=0.5, name="adsb"))
    try:
        assert env.episode_max_aircraft == 1  # only ADS2 is in the window
        assert not env.spawn_log  # due at (40 - 30) * 0.5 = 5 s
        _run(env, 10)
        (record,) = env.spawn_log
        assert record.callsign == "ADS2" and record.source == "adsb" and record.alt_ft == pytest.approx(11_000)
    finally:
        env.close()


def test_a_function_plans_from_the_episodes_geometry_and_routes_join_the_nearest_entry():
    def plan(rng, ctx):
        west = ctx.shapes["west"].center
        return [
            SpawnRequest(at=west, alt_ft=9_000, spd_kts=240, hdg_deg=90),
            SpawnRequest(at=EAST, alt_ft=9_000, spd_kts=240, hdg_deg=270),
        ]

    env = _env(PlannedSource(plan, assign_route=nearest_entry, max_aircraft=2))
    try:
        routes = sorted(r.route for r in env.spawn_log)
        assert routes == [("e_fix", "goal"), ("w_fix", "goal")]
    finally:
        env.close()


@pytest.mark.parametrize(("when_blocked", "now", "later"), [("defer", 1, 2), ("skip", 1, 1), ("allow", 2, 2)])
def test_a_blocked_spawn_is_handled_by_its_sources_rule(when_blocked, now, later):
    twins = [SpawnRequest(at=WEST, alt_ft=10_000, spd_kts=250, hdg_deg=90, time_s=0) for _ in range(2)]
    env = _env(Replay(twins, conflict_free=True, when_blocked=when_blocked))
    try:
        assert len(env.spawn_log) == now
        if when_blocked == "defer":
            bs.traf.delete(0)  # the first leaves: the spot clears
        _run(env, 10)
        assert len(env.spawn_log) == later
    finally:
        env.close()


def test_sources_count_into_max_aircraft_unless_open_or_capped():
    open_ended = PlannedSource(lambda rng, ctx: [], name="open")
    replay = Replay([SpawnRequest(at=WEST, alt_ft=9_000, time_s=0)] * 3)
    assert SpawnConfig(regions=[], sources=[replay, open_ended]).max_aircraft() == 3
    env = _env(Replay([SpawnRequest(at=LatLon(52.0, 3.6 + 0.2 * i), alt_ft=9_000, spd_kts=250, hdg_deg=0) for i in range(5)]), cap=2)
    try:
        assert len(env.spawn_log) == 2  # the cap trims the plan
    finally:
        env.close()


def test_a_mixture_draws_each_aircraft_from_one_of_its_sources():
    region = SpawnRegion(
        RegionBounds(BoxFootprint(51.8, 52.2, 4.8, 5.0)),
        n_aircraft=1,
        params={"alt_ft": 20_000, "spd_kts": 260, "hdg_deg": 270.0},
        route="east_in",
    )
    recorded = Replay([SpawnRequest(at=WEST, alt_ft=8_000, spd_kts=230, hdg_deg=90, route="west_in")])
    env = _env(Mixture([recorded, RegionSource(region)], weights=[1, 3], n=8))
    try:
        assert env.episode_max_aircraft == 8 and len(env.spawn_log) == 8
        routes = {r.route for r in env.spawn_log}
        assert routes <= {("w_fix", "goal"), ("e_fix", "goal")} and len(routes) == 2
        for r in env.spawn_log:
            assert r.alt_ft == pytest.approx(20_000 if r.route == ("e_fix", "goal") else 8_000)
    finally:
        env.close()


# ---- one plan, one draw, one clearing ------------------------------------ #


def _config(*sources, cap=None) -> SpawnConfig:
    region = SpawnRegion(
        shape=RegionBounds(BoxFootprint(51.9, 52.1, 3.5, 3.7)),
        n_aircraft=3,
        params={"alt_ft": 20_000.0, "spd_kts": (240.0, 280.0)},
        callsign_prefixes=["KLM"],
        route="west_in",
    )
    return SpawnConfig(
        regions=[region],
        routes={"west_in": ["w_fix", "goal"]},
        aircraft_type="B744",
        sources=list(sources),
        aircraft_cap=cap,
    )


def test_a_region_source_draws_as_its_region_does():
    import numpy as np

    from bluesky_sandbox.sim.spawn import PlanContext

    config = _config()
    drawn = config.draw_request(0, np.random.default_rng(3))
    via_source = RegionSource(config.regions[0]).draw(np.random.default_rng(3), PlanContext(spawn=config))
    assert via_source == drawn and drawn.callsign_prefix == "KLM"
    assert drawn.route == ["w_fix", "goal"]


def test_the_plan_is_the_regions_and_the_sources_by_time_within_the_cap():
    import numpy as np

    from bluesky_sandbox.sim.spawn import PlanContext

    replay = Replay([SpawnRequest(at=EAST, alt_ft=30_000.0, time_s=t) for t in (5.0, 50.0)], name="adsb", route="west_in")
    config = _config(replay)
    planned = config.plan_episode(np.random.default_rng(0), PlanContext(spawn=config))
    assert [p.source_index for p in planned].count(0) == 2
    assert [p.region_index for p in planned].count(0) == 3
    assert [p.request.time_s for p in planned] == sorted(p.request.time_s for p in planned)
    sourced = [p.request for p in planned if p.source_index == 0]
    assert all(r.actype == "B744" and r.route == ["w_fix", "goal"] for r in sourced)  # resolved
    capped = _config(replay, cap=4).plan_episode(np.random.default_rng(0), PlanContext(spawn=config))
    assert len(capped) == 4


def test_the_episode_spawns_its_plan():
    import numpy as np

    from bluesky_sandbox.sim.spawn import PlanContext

    replay = Replay([SpawnRequest(at=EAST, alt_ft=30_000.0, time_s=0.0)], name="adsb")
    scenario = _Scenario(replay)
    scenario.spawn = _config(replay)
    env = BlueskyEnv(scenario=scenario, config=EnvConfig(dt=5.0, obs_fields=[], action_fields=[]))
    try:
        env.reset(seed=7)
        log = env.spawn_log
        planned = scenario.spawn.plan_episode(np.random.default_rng(0), PlanContext(spawn=scenario.spawn))
        assert len(log) == len(planned) == 4
        assert sum(r.source == "adsb" for r in log) == 1
        assert all(r.callsign.startswith("KLM") for r in log if r.source is None)
    finally:
        env.close()
