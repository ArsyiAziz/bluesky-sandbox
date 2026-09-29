"""State fields: computed every step like the observation, seen by no agent.

``state_fields`` / ``intruder_state_fields`` are read in hooks as
``context.state`` (``batch.state`` batched), by part and field name in raw
values - exactly as ``raw_obs`` reads the observation - but no observation, of
actor or critic, carries them.
"""

from __future__ import annotations

import bluesky as bs
import numpy as np
import pytest
from bluesky.tools.aero import ft, kts

from bluesky_sandbox import AircraftControlState
from bluesky_sandbox.config import EnvConfig
from bluesky_sandbox.env import BlueskyEnv
from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.task import DesignKeys
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig
from bluesky_sandbox.ui.designer.design_keys import design_keys

_ALTITUDES_FT = (9_000.0, 11_000.0, 13_000.0)


class _Empty:
    def sample(self, _rng):
        return self.support()

    def support(self):
        return EpisodeSpec(
            airspace_bounds=None,
            spawn=SpawnConfig(regions=[]),
            queryables={},
            max_aircraft=0,
        )


class _ReadsState(BlueskyEnv):
    """Keeps what its hooks saw of the state."""

    seen: dict

    def reward(self, _obs, _action, _terminated, _truncated, context, _info, _rng):
        self.seen[context.acid] = (
            context.state["ownship"]["alt_ft"],
            dict(context.state["intruders"]),
        )
        return 0.0


class _ReadsStateBatched(BlueskyEnv):
    seen: dict

    def reward_batch(self, batch):
        self.seen["alt_ft"] = batch.state["ownship"]["alt_ft"]
        self.seen["dist"] = batch.state["intruders"]["dist_to_own_nm"]
        return np.zeros(len(batch))


def _config() -> EnvConfig:
    return EnvConfig(
        dt=12.0,
        obs_fields=[obs.CasKts()],
        intruder_obs_fields=[obs.DistToOwnNm()],
        state_fields=[obs.AltFt()],
        intruder_state_fields=[obs.DistToOwnNm(), obs.AltFt()],
        action_fields=[act.HdgDeltaDeg()],
    )


def _spawn(env) -> list[str]:
    env.reset(seed=0)
    env.seen = {}
    callsigns = []
    for i, alt in enumerate(_ALTITUDES_FT):
        acid = f"ST{i:03d}"
        assert bs.traf.cre(acid, "B744", 52.0 + 0.1 * i, 4.5, 90, alt * ft, 250 * kts)
        env.set_aircraft_control_state(acid, AircraftControlState.CONTROLLED)
        callsigns.append(acid)
    return callsigns


@pytest.fixture(scope="module")
def env():
    env = _ReadsState(scenario=_Empty(), config=_config())
    yield env
    env.close()


def _step(env, agents):
    return env.step({a: np.zeros(1, np.float32) for a in agents})


def test_state_is_not_in_the_observation_or_its_space(env):
    agents = _spawn(env)
    observations, *_ = _step(env, agents)
    space = env.observation_space(agents[0])
    assert set(space.spaces) == {"ownship", "intruders"}
    assert space["ownship"].shape == (1,)  # CasKts only
    assert set(observations[agents[0]]) == {"ownship", "intruders"}
    assert observations[agents[0]]["intruders"].shape == (2, 1)


def test_a_hook_reads_each_agents_state_by_name(env):
    agents = _spawn(env)
    _step(env, agents)
    for acid in agents:
        alt, intruders = env.seen[acid]
        idx = bs.traf.id.index(acid)
        assert alt == pytest.approx(bs.traf.alt[idx] / ft)
        others = [a for a in bs.traf.id if a != acid]
        assert list(intruders["acid"]) == others
        np.testing.assert_allclose(
            intruders["alt_ft"], [bs.traf.alt[bs.traf.id.index(a)] / ft for a in others]
        )
        np.testing.assert_allclose(
            intruders["dist_to_own_nm"],
            env.raw_observation(acid)["intruders"]["dist_to_own_nm"],
        )


def test_the_env_gives_an_agents_state_as_it_gives_its_observation(env):
    agents = _spawn(env)
    _step(env, agents)
    state = env.raw_state(agents[1])
    assert set(state) == {"ownship", "intruders"}
    assert state["ownship"]["alt_ft"] == pytest.approx(_ALTITUDES_FT[1], abs=50)
    with pytest.raises(KeyError, match="no state part 'critic_ownship'"):
        state["critic_ownship"]


def test_a_batched_hook_reads_every_agents_state_at_once():
    env = _ReadsStateBatched(scenario=_Empty(), config=_config())
    try:
        agents = _spawn(env)
        _step(env, agents)
        assert env.seen["alt_ft"].shape == (3,)
        assert env.seen["dist"].shape == (3, 2)
        np.testing.assert_allclose(
            env.seen["alt_ft"],
            [bs.traf.alt[bs.traf.id.index(a)] / ft for a in agents],
        )
    finally:
        env.close()


def test_without_state_fields_the_state_is_empty(env):
    env2 = BlueskyEnv(
        scenario=_Empty(),
        config=EnvConfig(dt=12.0, obs_fields=[obs.AltFt()], action_fields=[]),
    )
    try:
        agents = _spawn(env2)
        assert dict(env2.raw_state(agents[0])) == {}
    finally:
        env2.close()


def test_a_pair_field_is_refused_as_ownship_state():
    with pytest.raises(ValueError, match="state_fields contains pair-only"):
        EnvConfig(obs_fields=[], action_fields=[], state_fields=[obs.DistToOwnNm()])
    with pytest.raises(TypeError, match="intruder_state_fields must contain"):
        EnvConfig(obs_fields=[], action_fields=[], intruder_state_fields=[3])


def test_the_designer_knows_the_state_keys():
    keys = design_keys(DesignKeys("state"), _config(), _Empty().support())
    assert [k.name for k in keys] == ["ownship", "intruders"]
    assert [k.name for k in keys[0].keys] == ["alt_ft"]
    assert [k.name for k in keys[1].keys] == ["acid", "dist_to_own_nm", "alt_ft"]
