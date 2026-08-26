"""PettingZoo wrapper composition shared by integrations."""

from __future__ import annotations

from pettingzoo import ParallelEnv

from bluesky_sandbox.interface.wrappers import (
    IntruderPaddingWrapper,
    IntrudersKeepWrapper,
    StableIDsParallelWrapper,
    TimeLimitWrapper,
)

from .spaces import widen_intruder_space


def wrap_parallel_env(
    inner: ParallelEnv,
    *,
    max_episode_seconds: float = 1800.0,
    max_agents: int = 20,
    override_max_intruders: int | None = None,
    hold_background_until_episode_done: bool = False,
) -> StableIDsParallelWrapper:
    """Compose the PettingZoo wrapper stack around ``inner`` and pin stable IDs.

    ``inner`` is an already-constructed task env: tasks live outside this
    library, so it takes the env rather than a name to build one from.

    The result is still a PettingZoo-style parallel env. External integrations
    may wrap it for SB3, custom vector loops, or other training stacks.
    Intruder padding lives here because SuperSuit/SB3 require fixed-shape
    observations; raw task envs keep variable-length intruder sequences.

    Wrapper order is load-bearing. The per-intruder risk-envelope keep mask (a
    training-side signal) is attached in base intruder order, before any
    reordering or padding, so it stays aligned; applying it later silently
    misaligns the mask against the observations rather than raising.
    """
    # No-op when the task configures no keep provider.
    inner = IntrudersKeepWrapper(inner)
    inner = IntruderPaddingWrapper(inner)
    if override_max_intruders is not None:
        widen_intruder_space(inner, override_max_intruders)
    return StableIDsParallelWrapper(
        TimeLimitWrapper(inner, max_seconds=max_episode_seconds),
        max_agents=max_agents,
        hold_background_until_episode_done=hold_background_until_episode_done,
    )
