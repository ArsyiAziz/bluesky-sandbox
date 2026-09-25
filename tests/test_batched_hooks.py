"""Batched hooks: one call for every agent, optional, never alongside the per-agent one.

``reward_batch`` / ``terminated_batch`` / ``truncated_batch`` receive a
:class:`StepBatch` - the step for every agent, stacked - and return one value per
agent. A task-info provider with ``batched = True`` is called the same way.
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
from bluesky_sandbox.sim.scenario import EpisodeSpec
from bluesky_sandbox.sim.spawn import SpawnConfig

_ALTITUDES_FT = (9_000.0, 11_000.0, 13_000.0, 15_000.0)
_CEILING_FT = 12_000.0


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


class _PerAgent(BlueskyEnv):
    def reward(self, _obs, _action, _term, _trunc, context, _info, _rng):
        return context.raw_obs["ownship"]["alt_ft"] / 1_000.0

    def terminated(self, _obs, _action, context, _info, _rng):
        return context.raw_obs["ownship"]["alt_ft"] > _CEILING_FT


class _Batched(BlueskyEnv):
    """Keeps each batch it was given, and what it read from it in the step."""

    batches: list
    seen: dict

    def reward_batch(self, batch):
        self.batches.append(batch)
        # Read during the step: the aircraft that terminate are gone after it.
        self.seen["env_raw"] = {a: self.raw_observation(a) for a in batch.acids}
        self.seen["intruders"] = {
            key: batch.raw_obs["intruders"][key] for key in ("acid", "dist_to_own_nm")
        }
        self.seen["env_intruders"] = {
            a: dict(raw["intruders"]) for a, raw in self.seen["env_raw"].items()
        }
        return batch.raw_obs["ownship"]["alt_ft"] / 1_000.0

    def terminated_batch(self, batch):
        self.batches.append(batch)
        return batch.raw_obs["ownship"]["alt_ft"] > _CEILING_FT


def _make(cls, **config):
    config = EnvConfig(
        dt=12.0,
        obs_fields=[obs.AltFt(), obs.CasKts()],
        intruder_obs_fields=[obs.DistToOwnNm(), obs.AltFt()],
        action_fields=[act.HdgDeltaDeg(), act.AutopilotLnav()],
        **config,
    )
    env = cls(scenario=_Empty(), config=config)
    env.batches, env.seen = [], {}
    return env


def _spawn(env):
    env.reset(seed=0)
    agents = []
    for i, alt in enumerate(_ALTITUDES_FT):
        acid = f"BAT{i:03d}"
        assert bs.traf.cre(acid, "B744", 52.0 + 0.1 * i, 4.5, 90, alt * ft, 250 * kts)
        env.set_aircraft_control_state(acid, AircraftControlState.CONTROLLED)
        agents.append(acid)
    return agents


def _actions(agents, given=None):
    given = agents if given is None else given
    return {
        a: {"continuous": np.array([5.0], np.float32), "binary": np.array([0])}
        for a in given
    }


def _run(cls):
    env = _make(cls)
    try:
        return env.step(_actions(_spawn(env)))
    finally:
        env.close()


def test_batched_and_per_agent_hooks_decide_the_same():
    _obs, rewards, terminations, *_rest = _run(_PerAgent)
    _obs, batched_rewards, batched_terminations, *_rest = _run(_Batched)
    assert batched_rewards == pytest.approx(rewards)
    assert batched_terminations == terminations
    # Two of the four fly above the ceiling.
    assert sum(terminations.values()) == 2


@pytest.fixture
def batched():
    env = _make(_Batched)
    agents = _spawn(env)
    _obs, _rew, terminations, _trunc, _info = env.step(
        _actions(agents, given=agents[:3])
    )
    yield env, agents, terminations
    env.close()


def test_one_batch_serves_the_step_and_carries_its_decisions(batched):
    env, agents, terminations = batched
    first, second = env.batches
    assert first is second
    assert list(second.terminated) == [terminations[a] for a in agents]
    assert second.truncated is not None and not second.truncated.any()


def test_the_batch_stacks_each_agents_view(batched):
    env, agents, _terminations = batched
    batch = env.batches[-1]
    assert batch.acids == tuple(agents) and len(batch) == len(agents)
    n = len(agents)
    assert batch.obs["ownship"].shape == (n, 2)
    assert batch.obs["intruders"].shape == (n, n - 1, 2)
    for k, acid in enumerate(agents):
        env_view = env.seen["env_intruders"][acid]
        assert list(env.seen["intruders"]["acid"][k]) == list(env_view["acid"])
        assert np.allclose(
            env.seen["intruders"]["dist_to_own_nm"][k], env_view["dist_to_own_nm"]
        )
        assert batch.acidx[k] not in batch.intruder_idx[k]


def test_an_agent_given_no_action_is_nan_in_the_raw_action(batched):
    env, _agents, _terminations = batched
    batch = env.batches[-1]
    assert list(batch.has_action) == [True, True, True, False]
    assert np.allclose(batch.raw_action["hdg_delta_deg"][:3], 5.0)
    assert np.isnan(batch.raw_action["hdg_delta_deg"][3])


def test_raw_values_kept_past_their_step_refuse_to_build():
    # Built from the new traffic, the old indices would read other aircraft's
    # values. A part already read stays readable: it is a copy.
    env = _make(_PerAgent)
    try:
        agents = _spawn(env)
        raw = env.raw_observation(agents[0])
        ownship = raw["ownship"]
        env.step(_actions(agents))
        with pytest.raises(RuntimeError, match="earlier step"):
            raw["intruders"]
        assert raw["ownship"] is ownship
    finally:
        env.close()


def test_a_batched_task_info_provider_is_called_once_per_step():
    calls = []

    class Provider:
        batched = True

        def __call__(self, batch):
            calls.append(len(batch))
            for info, alt in zip(batch.infos, batch.raw_obs["ownship"]["alt_ft"]):
                info["task"]["alt_ft"] = float(alt)

    env = _make(_PerAgent, task_info_providers=[Provider()])
    try:
        agents = _spawn(env)
        _obs, _rew, _term, _trunc, infos = env.step(_actions(agents))
        assert calls[-1] == len(agents)
        assert infos[agents[2]]["task"]["alt_ft"] == pytest.approx(_ALTITUDES_FT[2])
    finally:
        env.close()


def test_no_batch_is_built_when_no_hook_is_batched():
    env = _make(_PerAgent)
    try:
        env.step(_actions(_spawn(env)))
        assert env._batch is None
    finally:
        env.close()


def test_a_hook_defined_both_ways_is_refused():
    with pytest.raises(TypeError, match=r"both reward\(\) and reward_batch\(\)"):

        class _Both(BlueskyEnv):
            def reward(self, *_args):
                return 0.0

            def reward_batch(self, batch):
                return np.zeros(len(batch))


def test_a_batched_hook_must_return_one_value_per_agent():
    class _Short(BlueskyEnv):
        def reward_batch(self, batch):
            return np.zeros(len(batch) - 1)

    env = _make(_Short)
    try:
        with pytest.raises(ValueError, match=r"reward_batch returned shape \(3,\)"):
            env.step(_actions(_spawn(env)))
    finally:
        env.close()
