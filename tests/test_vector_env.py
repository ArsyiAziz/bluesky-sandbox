"""Copies of a task env as one vector env, each copy in its own process."""

from __future__ import annotations

import numpy as np
import pytest

from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.integrations import sb3_vec_env, vec_env, wrap_parallel_env
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.sim.bounds import BoxFootprint, RegionBounds
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion

pytest.importorskip("supersuit")


class _Scenario:
    def sample(self, _rng):
        return self.support()

    def support(self):
        spawn = SpawnConfig(
            regions=[
                SpawnRegion(
                    bounds=RegionBounds(BoxFootprint(51.0, 53.0, 3.5, 6.0)),
                    n_aircraft=3,
                    params={"alt_ft": (8_000, 30_000), "spd_kts": (230, 270)},
                )
            ],
            aircraft_type="B744",
            conflict_free_spawn=False,
        )
        return EpisodeSpec(airspace_bounds=None, spawn=spawn, queryables={}, max_aircraft=3)


def make_env():
    """Module-level, so a worker process can import it."""
    env = BlueskyEnv(
        scenario=_Scenario(),
        config=EnvConfig(dt=5.0, obs_fields=[obs.AltFt(), obs.LatDeg()], action_fields=[act.HdgDeltaDeg()]),
    )
    return wrap_parallel_env(env, max_agents=3)


def _step(v, n=2):
    for _ in range(n):
        actions = np.stack([v.action_space.sample() for _ in range(v.num_envs)])
        out = v.step(actions)
    return out


def test_one_process_is_built_here():
    v = vec_env(make_env)
    try:
        observations, _ = v.reset(seed=0)
        assert v.num_envs == 3
        _step(v)
    finally:
        v.close()


def test_processes_each_run_their_own_episode():
    v = vec_env(make_env, n_processes=2)
    try:
        observations, _ = v.reset(seed=0)
        assert v.num_envs == 6
        own = np.asarray(observations)
        # Worker i is seeded seed + i: the two copies' aircraft differ.
        assert not np.allclose(own[:3], own[3:])
        _step(v)
    finally:
        v.close()


def test_a_process_count_below_one_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        vec_env(make_env, n_processes=0)


def test_sb3_takes_the_seed_it_is_given():
    pytest.importorskip("stable_baselines3")
    first = sb3_vec_env(make_env)
    try:
        first.seed(7)
        a = first.reset()
    finally:
        first.close()
    second = sb3_vec_env(make_env)
    try:
        second.seed(7)
        b = second.reset()
    finally:
        second.close()
    np.testing.assert_allclose(a, b)
