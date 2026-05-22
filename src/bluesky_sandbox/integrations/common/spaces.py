"""Space adaptation helpers shared by integrations."""

from __future__ import annotations

import numpy as np
from gymnasium.spaces import Box, Dict

from bluesky_sandbox.interface.wrappers import IntruderPaddingWrapper

from .wrappers import find_wrapper


def widen_intruder_space(env, new_max_intruders: int) -> None:
    """Resize ``IntruderPaddingWrapper``'s advertised obs space in place.

    This is an eval-only compatibility escape hatch: it lets a checkpoint
    trained at one ``max_intruders`` run on episodes with more aircraft. It
    must run before any downstream wrapper caches the observation space.
    """
    padding = find_wrapper(env, IntruderPaddingWrapper)
    if padding is None:
        raise ValueError("IntruderPaddingWrapper not found in env stack")

    inner = padding.env.observation_space(None)
    feature_box = inner["intruders"].feature_space
    feat_low = np.concatenate(
        [feature_box.low, np.array([0.0], dtype=np.float32)]
    )
    feat_high = np.concatenate(
        [feature_box.high, np.array([1.0], dtype=np.float32)]
    )
    padding._max_intr = int(new_max_intruders)
    padding._n_features = int(feature_box.shape[0])
    padding._n_per_intr = padding._n_features + 1
    spaces = dict(padding._obs_space.spaces)
    spaces["intruders"] = Box(
        low=np.tile(feat_low, (new_max_intruders, 1)),
        high=np.tile(feat_high, (new_max_intruders, 1)),
        dtype=np.float32,
    )
    padding._obs_space = Dict(spaces)
