"""Reward and cost: a number or a 1-D array per agent, per agent or batched.

``reward`` may return several components; ``cost`` is optional and, when a task
defines it, lands in each agent's ``info["cost"]``. Whichever form, it keeps one
shape from step to step.
"""

from __future__ import annotations

import numpy as np
import pytest

from bluesky_sandbox.env import BlueskyEnv

from test_batched_hooks import _actions, _make, _spawn


def _alt(context):
    return context.obs["ownship"]["alt_ft"]


class _VectorPerAgent(BlueskyEnv):
    def reward(self, _obs, _action, _term, _trunc, context, _info, _rng):
        return np.array([_alt(context) / 1_000.0, -1.0])

    def cost(self, _obs, _action, _term, _trunc, context, _info, _rng):
        return np.array([_alt(context) > 12_000.0, 0.5])


class _VectorBatched(BlueskyEnv):
    def reward_batch(self, batch):
        alt = batch.obs["ownship"]["alt_ft"]
        return np.stack([alt / 1_000.0, -np.ones_like(alt)], axis=1)

    def cost_batch(self, batch):
        alt = batch.obs["ownship"]["alt_ft"]
        return np.stack([alt > 12_000.0, np.full_like(alt, 0.5)], axis=1)


class _ScalarCost(BlueskyEnv):
    def cost(self, _obs, _action, _term, _trunc, context, _info, _rng):
        return _alt(context) / 10_000.0


def _step(cls):
    env = _make(cls)
    try:
        return env.step(_actions(_spawn(env)))
    finally:
        env.close()


@pytest.mark.parametrize("cls", [_VectorPerAgent, _VectorBatched])
def test_a_reward_and_cost_may_be_arrays(cls):
    _obs, rewards, _term, _trunc, infos = _step(cls)
    for acid, reward in rewards.items():
        assert reward.shape == (2,) and reward[1] == -1.0
        assert infos[acid]["cost"].shape == (2,)
    high = [a for a, r in rewards.items() if r[0] > 12.0]
    assert [infos[a]["cost"][0] for a in high] == [1.0] * len(high) and high


def test_per_agent_and_batched_agree():
    _o, rewards, _t, _u, infos = _step(_VectorPerAgent)
    _o, batched, _t, _u, batched_infos = _step(_VectorBatched)
    for acid in rewards:
        np.testing.assert_allclose(batched[acid], rewards[acid])
        np.testing.assert_allclose(batched_infos[acid]["cost"], infos[acid]["cost"])


def test_a_scalar_stays_a_float():
    _obs, rewards, _term, _trunc, infos = _step(_ScalarCost)
    assert all(type(r) is float for r in rewards.values())
    assert all(type(i["cost"]) is float for i in infos.values())


def test_without_a_cost_hook_no_cost_is_reported():
    _obs, _rewards, _term, _trunc, infos = _step(BlueskyEnv)
    assert not any("cost" in info for info in infos.values())


def test_a_reward_that_changes_shape_is_refused():
    class _Growing(BlueskyEnv):
        calls = 0

        def reward(self, *_args):
            type(self).calls += 1
            return np.zeros(type(self).calls)

    env = _make(_Growing)
    try:
        with pytest.raises(ValueError, match=r"reward returned shape \(2,\), but it returned \(1,\)"):
            env.step(_actions(_spawn(env)))
    finally:
        env.close()


def test_a_batched_cost_must_be_one_row_per_agent():
    class _Wide(BlueskyEnv):
        def cost_batch(self, batch):
            return np.zeros((len(batch), 2, 2))

    env = _make(_Wide)
    try:
        with pytest.raises(ValueError, match=r"cost_batch returned shape \(4, 2, 2\)"):
            env.step(_actions(_spawn(env)))
    finally:
        env.close()


def test_a_cost_defined_both_ways_is_refused():
    with pytest.raises(TypeError, match=r"both cost\(\) and cost_batch\(\)"):

        class _Both(BlueskyEnv):
            def cost(self, *_args):
                return 0.0

            def cost_batch(self, batch):
                return np.zeros(len(batch))
