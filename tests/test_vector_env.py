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


def make_env(render_mode=None, **drawing):
    """Module-level, so a worker process can import it."""
    env = BlueskyEnv(
        scenario=_Scenario(),
        render_mode=render_mode,
        **drawing,
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


def test_watching_draws_one_copy_the_others_headless(monkeypatch):
    pytest.importorskip("pygame")
    # No display needed: SDL draws to a dummy one, in this process and its workers.
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    v = vec_env(make_env, n_processes=2, watch=True)
    try:
        v.reset(seed=0)
        _step(v)
        assert v.num_envs == 6
    finally:
        v.close()


def test_a_padded_critic_intruder_block_widens_each_row():
    from gymnasium.spaces import Box, Dict  # noqa: PLC0415

    from bluesky_sandbox import critic_obs, critic_observation_space  # noqa: PLC0415

    space = Dict(
        {
            "ownship": Box(-1, 1, (3,)),
            "intruders": Box(-1, 1, (4, 2)),
            "critic_intruders": Box(0, 5, (4, 1)),
        }
    )
    widened = critic_observation_space(space)
    assert set(widened.spaces) == {"ownship", "intruders"}
    assert widened["intruders"].shape == (4, 3)
    batch = {k: np.stack([s.sample() for _ in range(6)]) for k, s in space.spaces.items()}
    assert critic_obs(batch)["intruders"].shape == (6, 4, 3)


class _LateScenario(_Scenario):
    """Its aircraft spawn a minute in: no agents at all before then."""

    def support(self):
        spec = super().support()
        (region,) = spec.spawn.regions
        late = type(region)(**{**region.__dict__, "spawn_time": 60.0})
        spawn = type(spec.spawn)(regions=[late], aircraft_type="B744", conflict_free_spawn=False)
        return type(spec)(**{**spec.__dict__, "spawn": spawn})


def make_late_env(render_mode=None):
    env = BlueskyEnv(
        scenario=_LateScenario(),
        render_mode=render_mode,
        config=EnvConfig(dt=5.0, obs_fields=[obs.AltFt()], action_fields=[act.HdgDeltaDeg()]),
    )
    return wrap_parallel_env(env, max_agents=3)


def test_an_episode_whose_traffic_spawns_later_is_not_reset_before_it():
    import bluesky as bs  # noqa: PLC0415

    v = vec_env(make_late_env)
    try:
        v.reset(seed=0)
        for _ in range(20):  # 100 s: past the spawn at 60 s
            _, _, terms, truncs, infos = v.step(
                np.stack([v.action_space.sample() for _ in range(v.num_envs)])
            )
        # Had every empty slot counted as done, the env would reset each step
        # and the clock never pass zero.
        assert bs.sim.simt >= 60.0
        assert sum(not i.get("_padded", False) for i in infos) == 3
    finally:
        v.close()
