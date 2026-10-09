"""What a field costs to read, measured on as many aircraft as the design may
fly (bluesky_sandbox.checks.cost) - each read at a new sim time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import bluesky as bs
import numpy as np
import pytest

from bluesky_sandbox.checks.cost import Like, cost_curve, counts_to, field_cost, fill_traffic, sim_step_ms
from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.fields.base import ObsField, ObsMeta, ObsQuantity, Unit

from test_env_bound import _Scenario


@dataclass(frozen=True)
class OneAtATime(ObsField):
    """No bulk path of its own: the env reads it one aircraft at a time."""

    meta = ObsMeta("one_at_a_time_ft", Unit.FT, ObsQuantity.ALTITUDE)
    low: float = 0.0
    high: float = 40_000.0

    def get(self, idx: Any) -> Any:
        return float(bs.traf.alt[int(idx)])

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@pytest.fixture(scope="module")
def env():
    config = EnvConfig(dt=1.0, obs_fields=[obs.AltFt(), OneAtATime()], intruder_obs_fields=[obs.DistToOwnNm()], action_fields=[])
    env = BlueskyEnv(scenario=_Scenario(), config=config)
    yield env
    env.close()


def test_the_traffic_is_topped_up_to_the_designs_most_or_as_asked(env):
    env.reset(seed=0)
    assert fill_traffic(env) == bs.traf.ntraf == env.episode_max_aircraft
    assert fill_traffic(env, 30) == bs.traf.ntraf == 30
    # Apart: no two of them share a point.
    points = set(zip(np.round(bs.traf.lat, 6), np.round(bs.traf.lon, 6)))
    assert len(points) == 30


def test_each_way_a_field_computes_is_timed_on_every_aircraft(env):
    env.reset(seed=0)
    fill_traffic(env, 30)
    cost = field_cost(env.config.obs_fields[0])
    assert cost.aircraft == 30 and set(cost.ms) == {"batched", "one at a time"}
    assert cost.env_path == "batched" and cost.batched is True and all(v > 0 for v in cost.ms.values())
    pair = field_cost(env.config.intruder_obs_fields[0])
    assert set(pair.ms) == {"pair matrix", "per ownship", "per pair"} and pair.env_path == "pair matrix"
    assert field_cost(env.config.obs_fields[1]).batched is False


def test_each_read_is_at_a_new_sim_time(env):
    env.reset(seed=0)
    before = bs.sim.simt
    field_cost(env.config.obs_fields[0], repeats=3)
    assert bs.sim.simt > before  # nothing kept for one sim time serves the next
    assert sim_step_ms(env, repeats=2) > 0


def test_counts_double_from_one_to_the_most():
    assert counts_to(1) == (1,)
    assert counts_to(120) == (1, 2, 4, 8, 16, 32, 64, 120)
    assert counts_to(64) == (1, 2, 4, 8, 16, 32, 64)


def test_a_curve_has_that_many_aircraft_in_the_air_at_each_count(env):
    env.reset(seed=0)
    like = Like.of(0)
    curve = cost_curve(env, env.config.obs_fields[0], like=like, most=16, repeats=2)
    assert curve.aircraft == (1, 2, 4, 8, 16) and bs.traf.ntraf == 16
    assert set(curve.ms) == {"batched", "one at a time"} and curve.env_path == "batched"
    assert all(len(v) == 5 and all(t is not None and t > 0 for t in v) for v in curve.ms.values())
    assert len(curve.sim_step_ms) == 5
    pair = cost_curve(env, env.config.intruder_obs_fields[0], like=like, most=4, repeats=1)
    assert pair.aircraft == (2, 4)  # a lone aircraft has no pair to time
    # Copies of the one given: its type and level.
    assert set(bs.traf.type) == {like.actype}
    assert np.allclose(np.asarray(bs.traf.alt) / 0.3048, like.alt_ft, atol=1.0)
