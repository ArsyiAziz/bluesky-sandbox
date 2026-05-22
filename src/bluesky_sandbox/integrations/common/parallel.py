"""PettingZoo env construction shared by integrations."""

from __future__ import annotations

from bluesky_sandbox.interface.wrappers import (
    IntruderPaddingWrapper,
    IntrudersKeepWrapper,
    StableIDsParallelWrapper,
    TimeLimitWrapper,
)

from .registry import resolve_make_env
from .spaces import widen_intruder_space


def build_parallel_env(
    *,
    env: str = "point_merge",
    render_mode: str | None = None,
    realtime: bool = False,
    max_episode_seconds: float = 1800.0,
    max_agents: int = 20,
    override_max_intruders: int | None = None,
    hold_background_until_episode_done: bool = False,
    env_kwargs: dict | None = None,
) -> StableIDsParallelWrapper:
    """Compose the PettingZoo wrapper stack and pin stable agent IDs.

    The result is still a PettingZoo-style parallel env. External integrations
    may wrap it for SB3, custom vector loops, or other training stacks.
    Intruder padding lives here because SuperSuit/SB3 require fixed-shape
    observations; raw task envs keep variable-length intruder sequences.
    """
    make_env = resolve_make_env(env)
    inner = make_env(
        render_mode=render_mode,
        realtime=realtime,
        **(env_kwargs or {}),
    )
    # Attach the per-intruder risk-envelope keep mask (training-side signal) in
    # base intruder order, before any reordering/padding so it stays aligned.
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
