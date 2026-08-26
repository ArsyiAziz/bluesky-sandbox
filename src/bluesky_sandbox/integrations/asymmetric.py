"""Actor / critic views of an observation (asymmetric actor-critic, CTDE).

An :class:`~bluesky_sandbox.config.EnvConfig` that declares
``critic_obs_fields`` / ``critic_intruder_obs_fields`` emits two extra keys
alongside the ordinary blocks::

    ownship          (own_dim,)              actor + critic
    intruders        (n_intruders, intr_dim) actor + critic
    critic_ownship   (c_own_dim,)            critic only
    critic_intruders (n_intruders, c_dim)    critic only

These helpers are that consumption step, kept free of any RL framework - they
take and return plain dicts and Gymnasium spaces:

    actor_obs(obs)                  drop the privileged blocks
    critic_obs(obs)                 fold them into ownship / intruders
    actor_observation_space(space)  the same, at the space level
    critic_observation_space(space)

Build the policy network from ``actor_observation_space`` and the value network
from ``critic_observation_space``, then feed each the matching view. The policy
stays a function of the ordinary fields, so the deployed agent never depends on
information it will not have.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium.spaces import Box, Sequence
from gymnasium.spaces import Dict as DictSpace

__all__ = [
    "CRITIC_OBS_KEYS",
    "actor_obs",
    "actor_observation_space",
    "critic_obs",
    "critic_observation_space",
    "has_privileged_obs",
]

#: Observation keys the env uses for the privileged, critic-only blocks.
CRITIC_OBS_KEYS = ("critic_ownship", "critic_intruders")


def has_privileged_obs(obs_or_space: Any) -> bool:
    """True when a single observation or a Dict space carries privileged blocks."""
    keys = getattr(obs_or_space, "spaces", obs_or_space)
    try:
        return any(key in keys for key in CRITIC_OBS_KEYS)
    except TypeError:
        return False


def actor_obs(obs: dict[str, Any]) -> dict[str, Any]:
    """The policy's view: ``obs`` without the privileged blocks.

    Returned unchanged (the same object) when there are none, so this is safe to
    call unconditionally on every observation.
    """
    if not has_privileged_obs(obs):
        return obs
    return {k: v for k, v in obs.items() if k not in CRITIC_OBS_KEYS}


def critic_obs(obs: dict[str, Any]) -> dict[str, Any]:
    """The value function's view: privileged blocks folded into the ordinary ones.

    ``critic_ownship`` is appended to ``ownship``; ``critic_intruders`` is
    appended to each ``intruders`` row. Returned unchanged when there are no
    privileged blocks.

    Raises:
        ValueError: if the two intruder blocks disagree on row count. They are
            emitted together, one row per intruder in the same order, so a
            mismatch means the observation was modified after the env produced
            it - typically by subsampling ``intruders`` for the actor and then
            passing that dict here. Merging anyway would silently leave the
            critic unprivileged, which is far harder to notice than this error.
    """
    if not has_privileged_obs(obs):
        return obs
    out = dict(obs)

    extra_own = out.pop("critic_ownship", None)
    if extra_own is not None:
        out["ownship"] = np.concatenate(
            [
                np.asarray(out["ownship"], dtype=np.float32),
                np.asarray(extra_own, dtype=np.float32),
            ],
            axis=-1,
        )

    extra_intr = out.pop("critic_intruders", None)
    if extra_intr is not None:
        extra_arr = np.asarray(extra_intr, dtype=np.float32)
        base = out.get("intruders")
        if base is None:
            out["intruders"] = extra_arr
        else:
            base_arr = np.asarray(base, dtype=np.float32)
            if base_arr.shape[0] != extra_arr.shape[0]:
                raise ValueError(
                    "critic_intruders is not row-aligned with intruders: "
                    f"intruders{base_arr.shape} vs critic_intruders"
                    f"{extra_arr.shape}. The two blocks are emitted together, "
                    "one row per intruder; pass the env's unmodified "
                    "observation here and use actor_obs() for the policy."
                )
            out["intruders"] = np.concatenate([base_arr, extra_arr], axis=-1)

    return out


def actor_observation_space(space: Any) -> Any:
    """Space the policy network is built from: ``space`` minus the critic keys."""
    if not isinstance(space, DictSpace) or not has_privileged_obs(space):
        return space
    kept = {k: v for k, v in space.spaces.items() if k not in CRITIC_OBS_KEYS}
    return DictSpace(kept)


def critic_observation_space(space: Any) -> Any:
    """Space the value network is built from: the widened blocks, critic keys gone.

    Mirrors :func:`critic_obs` so the encoder is sized for what it will be fed.
    """
    if not isinstance(space, DictSpace) or not has_privileged_obs(space):
        return space
    spaces = dict(space.spaces)

    extra_own = spaces.pop("critic_ownship", None)
    if extra_own is not None:
        own = spaces["ownship"]
        spaces["ownship"] = Box(
            low=np.concatenate([own.low, extra_own.low]),
            high=np.concatenate([own.high, extra_own.high]),
            dtype=np.float32,
        )

    extra_intr = spaces.pop("critic_intruders", None)
    if extra_intr is not None:
        extra_feat = extra_intr.feature_space
        base = spaces.get("intruders")
        if base is None:
            low, high = extra_feat.low, extra_feat.high
        else:
            base_feat = base.feature_space
            low = np.concatenate([base_feat.low, extra_feat.low])
            high = np.concatenate([base_feat.high, extra_feat.high])
        spaces["intruders"] = Sequence(
            Box(low=low, high=high, dtype=np.float32), stack=True
        )

    return DictSpace(spaces)
