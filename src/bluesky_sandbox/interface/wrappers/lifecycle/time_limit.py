from __future__ import annotations

from pettingzoo.utils.wrappers import BaseParallelWrapper


class TimeLimitWrapper(BaseParallelWrapper):
    """Truncate agents that have been in the environment longer than ``max_seconds``.

    This supplements (not replaces) the task's ``truncated(...)`` method.
    An agent is truncated when ``info["time_in_env"] >= max_seconds``, regardless
    of what the task returns.

    The wrapper marks the agent for deletion in the base environment
    set so the aircraft is removed from the BlueSky simulation on the next step,
    consistent with native truncations.

    Parameters
    ----------
    env:
        A BlueskyEnv or any wrapper around one.
    max_seconds:
        Maximum simulation time (seconds) an agent may spend in the environment
        before being truncated.

    Examples
    --------
    ::

        env = TimeLimitWrapper(
            MyTaskEnv(render_mode=None),
            max_seconds=3600.0,
        )
    """

    def __init__(self, env, max_seconds: float) -> None:
        super().__init__(env)
        self._max_seconds = max_seconds

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self.agents = self.env.agents
        base_env = self.env.unwrapped
        for agent, info in infos.items():
            if not truncations[agent] and info["time_in_env"] >= self._max_seconds:
                truncations[agent] = True
                base_env.mark_aircraft_for_deletion(agent)
        return observations, rewards, terminations, truncations, infos
