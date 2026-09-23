"""``QueryStateMonitor``'s vectorized dwell tracking agrees with the queryables.

Each substep the monitor decides, for every aircraft at once, whether it is
inside each tracked region and whether each tracked waypoint's constraints are
satisfied, and accumulates time and minimum distances. The queryables answer the
same questions one aircraft at a time (``contains_aircraft``,
``current_state``). This samples those per-aircraft answers every substep and
checks the monitor's step and episode figures against them.
"""

from __future__ import annotations

import math
from collections import defaultdict

import bluesky as bs
import pytest

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.sim.bounds import BoxFootprint, ConstantAltitudeBand, RegionBounds
from bluesky_sandbox.sim.queryables import QueryRegion, Waypoint
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion

_DT = 12.0
_REGION = "sector"
_WAYPOINT = "merge"


def _queryables():
    return {
        _REGION: QueryRegion(
            bounds=RegionBounds(
                BoxFootprint(51.97, 52.03, 4.52, 4.62),
                ConstantAltitudeBand(8_000, 8_400),
            ),
            track_temporal_state=True,
        ),
        _WAYPOINT: Waypoint(
            lat=52.0,
            lon=4.62,
            alt_ft=8_300,
            reach_radius_nm=1.5,
            alt_tolerance_ft=300,
            speed_kts=250,
            speed_tolerance_kts=15,
            track_temporal_state=True,
        ),
    }


class _Scenario:
    def __init__(self) -> None:
        self.queryables = _queryables()
        self.spawn = SpawnConfig(
            regions=[
                SpawnRegion(
                    bounds=RegionBounds(BoxFootprint(51.95, 52.05, 4.5, 4.62)),
                    n_aircraft=10,
                    params={"alt_ft": (8_000, 8_600), "spd_kts": (230, 270)},
                    route=[_WAYPOINT],
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
            queryables=self.queryables,
            max_aircraft=10,
        )


class _SamplesEachSubstep(BlueskyEnv):
    """Asks the queryables themselves, one aircraft at a time, every substep."""

    def start_episode_reference(self) -> None:
        self.held_total = defaultdict(float)

    def on_before_step(self) -> None:
        self.held_substeps = defaultdict(int)
        self.min_distance = defaultdict(lambda: math.inf)
        self.min_abs_alt = defaultdict(lambda: math.inf)

    def on_sim_step(self) -> None:
        simdt = float(self.config.simdt)
        region = self.episode_queryables[_REGION]
        waypoint = self.episode_queryables[_WAYPOINT]
        monitor = self._query_state_monitor
        for acidx, acid in enumerate(bs.traf.id):
            if region.contains_aircraft(acidx):
                self.held_substeps[acid, _REGION] += 1
                self.held_total[acid, _REGION] += simdt
            # ``query`` resolves the aircraft's own route target; ``current``
            # is the waypoint's per-aircraft verdict against it.
            current = monitor.query(acid, acidx, _WAYPOINT, waypoint).current
            if current.satisfied:
                self.held_substeps[acid, _WAYPOINT] += 1
                self.held_total[acid, _WAYPOINT] += simdt
            self.min_distance[acid] = min(self.min_distance[acid], current.distance_nm)
            if math.isfinite(current.alt_diff_ft):
                self.min_abs_alt[acid] = min(
                    self.min_abs_alt[acid], abs(current.alt_diff_ft)
                )


def test_monitor_dwell_matches_the_queryables_per_aircraft_answers():
    env = _SamplesEachSubstep(
        scenario=_Scenario(),
        config=EnvConfig(dt=_DT, obs_fields=[], action_fields=[]),
    )
    simdt = float(env.config.simdt)
    seen = defaultdict(int)
    try:
        for seed in (0, 1):
            env.reset(seed=seed)
            env.start_episode_reference()
            region = env.episode_queryables[_REGION]
            waypoint = env.episode_queryables[_WAYPOINT]
            monitor = env._query_state_monitor
            for _ in range(10):
                env.step({})
                for acidx, acid in enumerate(bs.traf.id):
                    inside = monitor.query(acid, acidx, _REGION, region)
                    held = env.held_substeps[acid, _REGION]
                    assert inside.step.inside == (held > 0)
                    assert inside.time.during_step_s == pytest.approx(held * simdt)
                    assert inside.time.total_s == pytest.approx(
                        env.held_total[acid, _REGION]
                    )

                    result = monitor.query(acid, acidx, _WAYPOINT, waypoint)
                    held = env.held_substeps[acid, _WAYPOINT]
                    assert result.step.satisfied == (held > 0)
                    assert result.time.during_step_s == pytest.approx(held * simdt)
                    assert result.time.total_s == pytest.approx(
                        env.held_total[acid, _WAYPOINT]
                    )
                    assert result.step.min_distance_nm == pytest.approx(
                        env.min_distance[acid], rel=1e-9
                    )
                    assert result.step.min_abs_alt_diff_ft == pytest.approx(
                        env.min_abs_alt[acid], rel=1e-9
                    )
                    seen["inside"] += inside.step.inside
                    seen["satisfied"] += result.step.satisfied
    finally:
        env.close()
    # Guard against a vacuous pass: both conditions really held somewhere.
    assert seen["inside"] > 5
    assert seen["satisfied"] > 5
