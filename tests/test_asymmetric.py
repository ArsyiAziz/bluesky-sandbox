"""Actor / critic observation views.

The failure these guard against is silent: if the privileged blocks stop being
merged, training still runs and the critic simply stops being privileged - no
exception, no shape error, just a slightly worse value function that nobody
notices for a week.  So the tests pin the widths on both sides, and pin that a
misaligned merge raises instead of quietly dropping the privileged half.

Imports live inside the tests, as in the rest of this suite - the repo root
only reaches ``sys.path`` when pytest runs a test, not while it collects one.
"""

from __future__ import annotations


def _spaces():
    import numpy as np
    from gymnasium.spaces import Box, Sequence
    from gymnasium.spaces import Dict as DictSpace

    return Box, DictSpace, Sequence, np


def _obs(n_intruders=3, own=4, intr=6, c_own=2, c_intr=5):
    """A synthetic observation shaped exactly like the env's."""
    import numpy as np

    return {
        "ownship": np.zeros(own, np.float32),
        "intruders": np.zeros((n_intruders, intr), np.float32),
        "critic_ownship": np.zeros(c_own, np.float32),
        "critic_intruders": np.zeros((n_intruders, c_intr), np.float32),
    }


def test_actor_view_drops_privileged_blocks():
    from bluesky_sandbox import actor_obs

    assert set(actor_obs(_obs())) == {"ownship", "intruders"}


def test_critic_view_widens_both_blocks():
    from bluesky_sandbox import critic_obs

    view = critic_obs(_obs())
    assert set(view) == {"ownship", "intruders"}
    assert view["ownship"].shape == (4 + 2,)
    assert view["intruders"].shape == (3, 6 + 5)


def test_zero_intruders_keeps_the_merged_width():
    """An agent alone in the sector must still emit rows of the merged width,
    or a batch of such agents ends up ragged in the feature axis."""
    from bluesky_sandbox import critic_obs

    view = critic_obs(_obs(n_intruders=0))
    assert view["intruders"].shape == (0, 6 + 5)


def test_misaligned_rows_raise():
    """Subsampling ``intruders`` for the actor and then asking for the critic
    view is the realistic mistake; merging anyway would un-privilege the critic
    without any signal."""
    import pytest

    from bluesky_sandbox import critic_obs

    obs = _obs(n_intruders=3)
    obs["intruders"] = obs["intruders"][:1]
    with pytest.raises(ValueError, match="not row-aligned"):
        critic_obs(obs)


def test_observations_without_privileged_blocks_pass_through():
    from bluesky_sandbox import actor_obs, critic_obs

    obs = {k: v for k, v in _obs().items() if not k.startswith("critic_")}
    assert actor_obs(obs) is obs
    assert critic_obs(obs) is obs


def test_spaces_match_the_views_they_describe():
    """The encoders are built from the spaces and fed the views; if the two
    disagree the mismatch only surfaces at the first forward pass."""
    from bluesky_sandbox import (
        actor_obs,
        actor_observation_space,
        critic_obs,
        critic_observation_space,
    )

    Box, DictSpace, Sequence, np = _spaces()
    space = DictSpace(
        {
            "ownship": Box(-1, 1, (4,), np.float32),
            "intruders": Sequence(Box(-1, 1, (6,), np.float32), stack=True),
            "critic_ownship": Box(-1, 1, (2,), np.float32),
            "critic_intruders": Sequence(Box(-1, 1, (5,), np.float32), stack=True),
        }
    )
    obs = _obs()

    a_space, c_space = actor_observation_space(space), critic_observation_space(space)
    a_obs, c_obs = actor_obs(obs), critic_obs(obs)

    assert a_space.spaces["ownship"].shape == a_obs["ownship"].shape
    assert c_space.spaces["ownship"].shape == c_obs["ownship"].shape
    assert (
        a_space.spaces["intruders"].feature_space.shape[0]
        == a_obs["intruders"].shape[-1]
    )
    assert (
        c_space.spaces["intruders"].feature_space.shape[0]
        == c_obs["intruders"].shape[-1]
    )
