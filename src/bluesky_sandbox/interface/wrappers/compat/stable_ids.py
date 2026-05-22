"""Stable, fixed-size agent ID pool for SuperSuit / SB3 compatibility.

SuperSuit's ``pettingzoo_env_to_vec_env_v1`` requires a parallel env to:
  * expose a constant ``possible_agents`` list across episodes,
  * return observations for every agent in ``possible_agents`` on every
    ``reset`` / ``step`` (this wrapper pads with zeros for absent slots).

BlueskyBaseEnvironment violates both - aircraft callsigns are randomly
generated each spawn and ``n_aircraft`` varies per episode. This wrapper
maps each episode's first ``N`` real callsigns to ``agent_0 .. agent_{N-1}``
and pads the unused slots up to ``max_agents``.

Lifecycle model (black-death style mapping)::

    once an aircraft terminates / truncates, its stable slot is unmapped
    and never reused within the episode. The wrapper reports
    ``info["_agent_done"]`` for that transition, then pads the slot until
    the underlying simulator reaches env-wide done. That lets background
    traffic continue without triggering an early vector reset when visual
    background holding is enabled.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium.spaces import Box, Dict
from pettingzoo import ParallelEnv


class StableIDsParallelWrapper(ParallelEnv):
    """Pin a PettingZoo ParallelEnv to a fixed pool of stable agent IDs.

    Parameters
    ----------
    env:
        A PettingZoo ParallelEnv. Its observation_space must be agent-
        agnostic (same Box or Dict for every agent). Its action_space is
        expected to be fixed-size and finite; field-level action normalizers
        determine the external bounds for each action dimension.
    max_agents:
        Upper bound on simultaneous aircraft. Must be >= the spawn config's
        deterministic upper bound (``spawn.max_aircraft()``); a smaller
        value raises at construction. Spawning fewer aircraft than
        ``max_agents`` is fine - the unused slots stay padded until the
        simulator reaches env-wide done.
    hold_background_until_episode_done:
        Keep stable slots non-terminal after per-agent termination until
        the underlying simulator finishes background traffic. Use this for
        visual eval; leave it off for training-style FAF termination.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env: ParallelEnv,
        max_agents: int,
        *,
        hold_background_until_episode_done: bool = False,
    ) -> None:
        self.env = env
        self.max_agents = int(max_agents)
        self.hold_background_until_episode_done = bool(hold_background_until_episode_done)
        # Fail fast on undersized slot pools: the deterministic upper bound
        # on aircraft this episode could spawn is ``spawn.max_aircraft()``,
        # so anything smaller will eventually run out of stable slots at
        # runtime - better to detect at construction than mid-episode.
        max_ac = env.unwrapped.episode_spawn.max_aircraft()
        if self.max_agents < max_ac:
            raise ValueError(
                f"max_agents ({self.max_agents}) must be >= "
                f"spawn.max_aircraft() ({max_ac}); the spawn config can "
                f"yield more aircraft than the stable-ID pool can hold."
            )
        self.possible_agents: list[str] = [f"agent_{i}" for i in range(self.max_agents)]
        self.metadata = getattr(env, "metadata", {"render_modes": []})
        self.render_mode = getattr(env, "render_mode", None)

        self._stable_to_real: dict[str, str] = {}
        self._real_to_stable: dict[str, str] = {}
        # Slots that have ever been assigned this episode. Cleared on reset
        # but not on death: under the black-death mapping model a slot whose
        # aircraft has terminated must never be reused for a later-spawning
        # aircraft.
        self._used_slots: set[str] = set()

        # Cache the observation space. Field-level normalizers make the
        # configured task spaces stable for every agent, so no live callsign
        # is needed for the probe.
        self._obs_space = env.observation_space(None)
        # Pre-build a per-agent zero "obs" used to pad dead/unspawned slots.
        self._zero_obs = self._make_zero_obs(self._obs_space)
        # Probe the wrapped action space directly so mixed raw/normalised
        # fields keep their declared bounds, e.g. [-1, 1] deltas plus [0, 1]
        # switches.
        self._act_space: Box = env.action_space(None)

    @staticmethod
    def _make_zero_obs(space):
        """A zero-valued obs matching the wrapped env's observation_space."""
        if isinstance(space, Dict):
            return {k: np.zeros(v.shape, dtype=np.float32) for k, v in space.spaces.items()}
        return np.zeros(space.shape, dtype=np.float32)

    @staticmethod
    def _copy_obs(zero):
        """Independent copy of the cached zero obs (Dict or ndarray)."""
        if isinstance(zero, dict):
            return {k: v.copy() for k, v in zero.items()}
        return zero.copy()

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------
    @property
    def unwrapped(self) -> ParallelEnv:
        return self.env.unwrapped

    @property
    def agents(self) -> list[str]:
        # Always claim all stable slots. MarkovVectorEnv uses this to decide
        # which slots receive an action; we filter internally so empty /
        # already-dead slots simply ignore their action.
        return list(self.possible_agents)

    @property
    def num_agents(self) -> int:
        return self.max_agents

    @property
    def max_num_agents(self) -> int:
        return self.max_agents

    def observation_space(self, agent: str):
        return self._obs_space

    def action_space(self, agent: str) -> Box:
        return self._act_space

    def _assign_slot(self, real: str) -> str:
        """Bind ``real`` to the next never-used stable slot and return it.

        The construction-time check on :attr:`max_agents` guarantees the
        pool can hold every aircraft the spawn config could ever emit
        this episode, so a missing free slot here is a programming
        invariant violation - assert rather than silently drop.
        """
        for stable in self.possible_agents:
            if stable not in self._used_slots:
                self._stable_to_real[stable] = real
                self._real_to_stable[real] = stable
                self._used_slots.add(stable)
                return stable
        raise AssertionError(
            f"StableIDsParallelWrapper ran out of stable slots binding "
            f"{real!r}; max_agents={self.max_agents}. The construction "
            f"check should have prevented this."
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, Any], dict[str, dict]]:
        obs_dict, info_dict = self.env.reset(seed=seed, options=options)
        self._stable_to_real.clear()
        self._real_to_stable.clear()
        self._used_slots.clear()
        for real in obs_dict:
            self._assign_slot(real)

        s_obs: dict[str, Any] = {}
        s_info: dict[str, dict] = {}
        for stable in self.possible_agents:
            real = self._stable_to_real.get(stable)
            if real is not None:
                s_obs[stable] = obs_dict[real]
                s_info[stable] = info_dict.get(real, {})
            else:
                s_obs[stable] = self._copy_obs(self._zero_obs)
                s_info[stable] = {}
        return s_obs, s_info

    def step(
        self,
        actions: dict[str, np.ndarray],
    ) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        real_actions: dict[str, np.ndarray] = {}
        for stable, real in self._stable_to_real.items():
            if stable in actions:
                real_actions[real] = np.asarray(actions[stable], dtype=np.float32)

        base = self.env.unwrapped
        # Step the inner env when there's something to do: live aircraft to
        # control OR queued spawns waiting to be drained. The latter is
        # essential for deferred-spawn configs where the t=0 cohort can die
        # before any later-scheduled aircraft materialise - without it the
        # queue would never drain and SuperSuit would auto-reset. Visual eval
        # can also continue stepping after per-agent termination so background
        # aircraft finish landing before env-wide reset.
        has_pending_spawns = bool(getattr(base, "_spawn_queue", []))
        needs_sim_step = (
            self.hold_background_until_episode_done
            and not bool(getattr(base, "episode_done", False))
        )
        if real_actions or has_pending_spawns or needs_sim_step:
            next_obs, rewards, terms, truncs, infos = self.env.step(real_actions)
            # Bind any newly-materialised real callsigns (queue drain) to
            # fresh stable slots before composing the SB3 view.
            for real in next_obs:
                if real not in self._real_to_stable:
                    self._assign_slot(real)
        else:
            # All real aircraft already terminated and nothing queued. The
            # next MarkovVectorEnv loop will see env_done and reset. Still
            # drive the renderer so the pygame event pump fires this
            # iteration; otherwise the auto-reset boundary drops a frame's
            # worth of mouse events and the UI (e.g. slice-plane drag) goes
            # dead until the next real step.
            base._driver.update()
            next_obs, rewards, terms, truncs, infos = {}, {}, {}, {}, {}

        s_obs: dict[str, Any] = {}
        s_rew: dict[str, float] = {}
        s_term: dict[str, bool] = {}
        s_trunc: dict[str, bool] = {}
        s_info: dict[str, dict] = {}

        # Snapshot after the inner step so padded slots can either preserve
        # black-death training semantics or hold visual background traffic.
        episode_done = bool(getattr(base, "episode_done", False))
        spawns_remaining = bool(getattr(base, "_spawn_queue", []))
        for stable in self.possible_agents:
            real = self._stable_to_real.get(stable)
            if real is None:
                # Unmapped slot - two cases, distinguished by _used_slots:
                #   * already-used (in _used_slots): an aircraft lived here
                #     and has finished. In training mode this stays
                #     terminated; in visual-hold mode it waits for env-wide
                #     completion.
                #   * never-used (not in _used_slots): the slot is waiting
                #     for a deferred spawn. Keep it padded while the
                #     simulator can still materialise aircraft.
                # ``_padded`` keeps custom monitors from logging these as
                # synthetic 1-step episodes either way.
                s_obs[stable] = self._copy_obs(self._zero_obs)
                s_rew[stable] = 0.0
                if self.hold_background_until_episode_done:
                    s_term[stable] = episode_done
                else:
                    s_term[stable] = stable in self._used_slots or not spawns_remaining
                s_trunc[stable] = False
                s_info[stable] = {"_padded": True}
                continue

            if real in next_obs:
                s_obs[stable] = next_obs[real]
                s_rew[stable] = float(rewards.get(real, 0.0))
                real_terminal = bool(terms.get(real, False))
                real_truncated = bool(truncs.get(real, False))
                real_done = real_terminal or real_truncated
                hold_for_background = (
                    self.hold_background_until_episode_done
                    and real_done
                    and not episode_done
                )
                s_term[stable] = real_terminal and not hold_for_background
                s_trunc[stable] = real_truncated and not hold_for_background
                info = {**infos.get(real, {}), "_padded": False}
                if real_done:
                    info["_agent_done"] = True
                    info["_agent_terminal"] = real_terminal
                    info["_agent_truncated"] = real_truncated
                if hold_for_background:
                    info["_background_hold"] = True
                s_info[stable] = info
                if real_done:
                    self._stable_to_real.pop(stable, None)
                    self._real_to_stable.pop(real, None)
            else:
                # Real callsign disappeared mid-step.
                s_obs[stable] = self._copy_obs(self._zero_obs)
                s_rew[stable] = 0.0
                s_term[stable] = True
                s_trunc[stable] = False
                s_info[stable] = {"_padded": True}
                self._stable_to_real.pop(stable, None)
                self._real_to_stable.pop(real, None)

        return s_obs, s_rew, s_term, s_trunc, s_info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
