"""Attach the per-intruder risk-envelope keep mask as a training-side signal.

The base env emits only the policy's MDP observation (``ownship`` +
``intruders``), so its observations always match ``observation_space``. The keep
mask - a per-intruder flag for the actor's subset/cardinality augmentation - is
*not* an env observation, so it is added here instead.

``intruders_keep`` is sourced from a task-info provider's
``keep_mask(own_idx, other_indices)`` ("keep" is exactly "contributes to the
intrinsic cost"), aligned to the base env's intruder order (BlueSky traffic
order, excluding the ownship). Apply this wrapper *before*
``IntruderPaddingWrapper`` so the mask is padded together with ``intruders``.

No-op (obs and space unchanged) when no configured provider supplies a keep
mask, or when the env has no intruder fields.
"""

from __future__ import annotations

import bluesky as bs
import numpy as np
from gymnasium.spaces import Box, Dict, Sequence
from pettingzoo.utils.wrappers import BaseParallelWrapper


class IntrudersKeepWrapper(BaseParallelWrapper):
    """Add an ``intruders_keep`` bool mask (per intruder) to each agent's obs."""

    def __init__(self, env) -> None:
        super().__init__(env)
        cfg = env.unwrapped.config
        providers = getattr(cfg, "task_info_providers", ()) or ()
        self._providers = [
            provider
            for provider in providers
            if getattr(provider, "keep_fn", None) is not None
            and callable(getattr(provider, "keep_mask", None))
        ]
        self._active = bool(self._providers) and bool(cfg.intruder_obs_fields)

    def _mask(self, own_idx: int, other_indices, n_others: int) -> np.ndarray:
        for provider in self._providers:
            mask = provider.keep_mask(own_idx, other_indices)
            if mask is not None:
                return np.asarray(mask, dtype=bool).reshape(n_others)
        # Active provider but no mask this step: keep everything (matches space).
        return np.ones(n_others, dtype=bool)

    def _augment(self, observations: dict) -> dict:
        if not self._active:
            return observations
        out = {}
        for agent, obs in observations.items():
            own_idx = bs.traf.id.index(agent)
            other_indices = tuple(i for i in range(bs.traf.ntraf) if i != own_idx)
            out[agent] = {
                **obs,
                "intruders_keep": self._mask(
                    own_idx, other_indices, len(other_indices)
                ),
            }
        return out

    def observation_space(self, agent):
        space = self.env.observation_space(agent)
        if not self._active or not isinstance(space, Dict):
            return space
        spaces = dict(space.spaces)
        spaces["intruders_keep"] = Sequence(
            Box(low=0, high=1, shape=(), dtype=np.bool_), stack=True
        )
        return Dict(spaces)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents
        return self._augment(obs), info

    def step(self, actions):
        obs, rew, term, trunc, info = self.env.step(actions)
        self.agents = self.env.agents
        return self._augment(obs), rew, term, trunc, info
