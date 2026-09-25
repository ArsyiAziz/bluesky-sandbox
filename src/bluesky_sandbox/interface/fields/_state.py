"""Per-process stores that stateful fields read and the environment writes.

One BlueSky sim per process, so each store is per env. The environment clears
them all on reset (:func:`reset_all_field_state`) and fields drop an aircraft's
entries on despawn through their ``on_aircraft_removed`` hooks.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._lag import _LAG_HISTORY
from ._pairs import _CD_PAIR_CACHE


def reset_all_field_state(seed: int | None = None) -> None:
    """Clear every per-aircraft field store, whatever this env configures.

    Recording and per-aircraft forgetting are opt-in - a field that nobody
    configures records nothing, so there is nothing to drop. Episode reset is
    the exception: the stores are module-level, so two envs built from
    different configs in one process share them, and the second env would
    inherit whatever the first left behind in a store it has no field for.
    One sweep, called by the environment on reset, restores that isolation
    without reintroducing a per-store list for anyone to forget to update.
    """
    global _COMM_NOISE_RNG
    _LAST_NORM_ACTION.clear()
    _LAG_HISTORY.clear()
    _CD_PAIR_CACHE.clear()
    _TIME_IN_ENV.clear()
    _COMM_MESSAGE.clear()
    _COMM_NOISE_RNG = np.random.default_rng(seed)


class _LastActionBacked:
    """State hooks for the previous-action field.

    The store is written only from here - the field that reads it - so a
    recorded action and its removal cannot be maintained in two places that
    drift apart.
    """

    def on_action_applied(self, acid: str, action) -> None:
        _LAST_NORM_ACTION[acid] = np.asarray(action, dtype=np.float32)

    def on_aircraft_removed(self, acid: str) -> None:
        _LAST_NORM_ACTION.pop(acid, None)


class _LagHistoryBacked:
    """State hooks for lag/stack wrappers.

    History is pushed lazily on read (see ``get_many``), so there is no
    ``on_step`` here - only the per-aircraft drop.
    """

    def on_aircraft_removed(self, acid: str) -> None:
        # Uid-keyed rows need nothing: the next access drops the departed
        # aircraft. Callsign-keyed ones must forget it here, or a new aircraft
        # given the same callsign would inherit its history.
        for ring in _LAG_HISTORY.values():
            ring.forget(acid)


class _TimeInEnvBacked:
    """State hooks for time-in-env.

    Spawn times live in the environment, so the age arrives on the substep
    context; recording and the per-aircraft drop are here.
    """

    def on_step(self, ctx) -> None:
        _TIME_IN_ENV.update(ctx.age_s)

    def on_aircraft_removed(self, acid: str) -> None:
        _TIME_IN_ENV.pop(acid, None)


class _CommBacked:
    """State hooks for the comm-message channel field."""

    def on_aircraft_removed(self, acid: str) -> None:
        _COMM_MESSAGE.pop(acid, None)


# Per-process store of each aircraft's most recent *normalized* action, keyed by
# callsign. The environment writes it when actions are applied (one BlueSky sim
# per process, so this is per-env), and :class:`PrevActionNorm` reads it. Exposing
# the previous action keeps an action-rate reward penalty (``|a_t - a_{t-1}|``)
# Markovian w.r.t. the observation - otherwise that reward depends on unobserved
# history, which the value function cannot predict.
_LAST_NORM_ACTION: dict[str, np.ndarray] = {}


# The environment's flat normalized action-space bounds, published so
# PrevActionNorm can size its observation bounds to the actual action space
# (which depends on the action fields' normalizers) instead of assuming a range.
# Per process - one BlueSky sim / env per process.
_ACTION_SPACE_BOUNDS: tuple[np.ndarray, np.ndarray] | None = None


def set_action_space_bounds(low: Any, high: Any) -> None:
    """Publish the flat normalized action-space bounds (for PrevActionNorm)."""
    global _ACTION_SPACE_BOUNDS
    _ACTION_SPACE_BOUNDS = (
        np.asarray(low, dtype=np.float32).ravel(),
        np.asarray(high, dtype=np.float32).ravel(),
    )


def clear_action_space_bounds() -> None:
    """Forget the published action-space bounds."""
    global _ACTION_SPACE_BOUNDS
    _ACTION_SPACE_BOUNDS = None


# Per-process store of each aircraft's seconds since it entered the environment.
# Published by the environment each step from its own spawn-time bookkeeping
# (``BaseEnvironment.aircraft_spawn_time``), which an ObsField cannot reach: it
# sees only ``bs.traf``, and BlueSky keeps no per-aircraft age. Read by
# TimeInEnvS. 0.0 if unknown.
_TIME_IN_ENV: dict[str, float] = {}


def get_time_in_env(acid: str) -> float:
    """Return the stored seconds-since-spawn for ``acid``, or 0.0 if unknown."""
    return _TIME_IN_ENV.get(acid, 0.0)


# Per-process store of each aircraft's broadcast communication message - a
# small learned signal emitted through a CommBroadcast action channel and read
# back by other agents through IntruderCommMessage. No physical effect on the
# aircraft; purely an information channel between agents, one step delayed
# (emitted at step t, observed by others at t+1). Keyed ``acid -> channel``.
_COMM_MESSAGE: dict[str, dict[int, float]] = {}


def record_comm_message(acid: str, channel: int, value: float) -> None:
    """Store one channel of an aircraft's broadcast message (for IntruderCommMessage)."""
    _COMM_MESSAGE.setdefault(acid, {})[int(channel)] = float(value)


def get_comm_message(acid: str, channel: int) -> float:
    """Return an aircraft's stored message channel, or 0.0 (silence) if unset."""
    return _COMM_MESSAGE.get(acid, {}).get(int(channel), 0.0)


# RNG for receiver-side communication-channel noise (IntruderCommMessage's
# ``noise_std``). Reseeded from the episode seed at reset so training rollouts
# stay reproducible.
_COMM_NOISE_RNG = np.random.default_rng()
