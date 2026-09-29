"""Per-process stores that stateful fields read and the environment writes.

One BlueSky sim per process, so each store is per env. Each registers its reset
with :func:`~._common.on_reset`, and fields drop an aircraft's entries on
despawn through their ``on_aircraft_removed`` hooks.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

import bluesky as bs
import numpy as np

from bluesky_sandbox.sim.aircraft_uids import aircraft_keys

from ._common import on_reset


class AircraftMemory:
    """One remembered value per aircraft that follows the aircraft, not its
    callsign.

    Keyed by the aircraft's uid (:mod:`~bluesky_sandbox.sim.aircraft_uids`):
    BlueSky reuses a deleted aircraft's callsign, and a store keyed by callsign
    hands the new aircraft the old one's value - its last action, its age, its
    last broadcast. Without a runtime it falls back to callsigns, which
    :meth:`forget` drops on despawn. Cleared on every environment reset.

    Written and read by BlueSky traffic index, while the aircraft is live;
    ``default`` is what an aircraft with nothing recorded reads.
    """

    def __init__(self, default: Any = None) -> None:
        self.default = default
        # key -> (callsign when written, value); the callsign lets a despawn
        # forget the entry even once its uid can no longer be looked up.
        self._entries: dict[Hashable, tuple[str, Any]] = {}
        on_reset(lambda _seed: self.clear())

    def write(self, idx: int, value: Any) -> None:
        ids = bs.traf.id
        self._entries[aircraft_keys(ids)[int(idx)]] = (str(ids[int(idx)]), value)

    def write_callsigns(self, values: Mapping[str, Any]) -> None:
        """Write by callsign, for values the environment keeps by callsign."""
        ids = bs.traf.id
        keys = aircraft_keys(ids)
        for idx, acid in enumerate(ids):
            if acid in values:
                self._entries[keys[idx]] = (str(acid), values[acid])

    def read(self, indices) -> list[Any]:
        """Each aircraft's value, or ``default``, in the order of ``indices``."""
        keys = aircraft_keys(bs.traf.id)
        entries, default = self._entries, self.default
        return [
            entry[1] if (entry := entries.get(keys[int(i)])) is not None else default
            for i in np.asarray(indices, dtype=np.intp).ravel()
        ]

    def read_one(self, idx: int) -> Any:
        return self.read([int(idx)])[0]

    def forget(self, acid: str) -> None:
        """Drop what despawned aircraft ``acid`` left behind."""
        for key in [k for k, (owner, _v) in self._entries.items() if owner == acid]:
            del self._entries[key]

    def clear(self) -> None:
        self._entries.clear()


class _LastActionBacked:
    """State hooks for the previous-action field.

    The store is written only from here - the field that reads it - so a
    recorded action and its removal cannot be maintained in two places that
    drift apart.
    """

    def on_action_applied(self, acid: str, action) -> None:
        _LAST_NORM_ACTION.write_callsigns({acid: np.asarray(action, dtype=np.float32)})

    def on_aircraft_removed(self, acid: str) -> None:
        _LAST_NORM_ACTION.forget(acid)


class _TimeInEnvBacked:
    """State hooks for time-in-env.

    Spawn times live in the environment, so the age arrives on the substep
    context; recording and the per-aircraft drop are here.
    """

    def on_step(self, ctx) -> None:
        _TIME_IN_ENV.write_callsigns(ctx.age_s)

    def on_aircraft_removed(self, acid: str) -> None:
        _TIME_IN_ENV.forget(acid)


class _CommBacked:
    """State hooks for the comm-message channel field."""

    def on_aircraft_removed(self, acid: str) -> None:
        _COMM_MESSAGE.forget(acid)


class _ArrivalTimeBacked:
    """State hooks for what reads an aircraft's arrival times."""

    def on_aircraft_removed(self, acid: str) -> None:
        _ARRIVAL_TIMES.forget(acid)


class _ActionMaskBacked:
    """State hooks for the action masks and the field that reads them."""

    def on_aircraft_removed(self, acid: str) -> None:
        _ACTION_MASKS.forget(acid)
        _ACTION_LOCKS.forget(acid)
        _CLEARANCE_EXPIRY.forget(acid)


# Each aircraft's most recent *normalized* action. The environment writes it when actions are applied (one BlueSky sim
# per process, so this is per-env), and :class:`PrevActionNorm` reads it. Exposing
# the previous action keeps an action-rate reward penalty (``|a_t - a_{t-1}|``)
# Markovian w.r.t. the observation - otherwise that reward depends on unobserved
# history, which the value function cannot predict.
_LAST_NORM_ACTION = AircraftMemory()


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


# Each aircraft's seconds since it entered the environment.
# Published by the environment each step from its own spawn-time bookkeeping
# (``BaseEnvironment.aircraft_spawn_time``), which an ObsField cannot reach: it
# sees only ``bs.traf``, and BlueSky keeps no per-aircraft age. Read by
# TimeInEnvS. 0.0 if unknown.
_TIME_IN_ENV = AircraftMemory(default=0.0)


# Each aircraft's broadcast communication message - a
# small learned signal emitted through a CommBroadcast action channel and read
# back by other agents through IntruderCommMessage. No physical effect on the
# aircraft; purely an information channel between agents, one step delayed
# (emitted at step t, observed by others at t+1). Values are ``{channel: value}``.
_COMM_MESSAGE = AircraftMemory(default={})


def record_comm_message(idx: int, channel: int, value: float) -> None:
    """Store one channel of an aircraft's broadcast message (for IntruderCommMessage)."""
    message = dict(_COMM_MESSAGE.read_one(idx))
    message[int(channel)] = float(value)
    _COMM_MESSAGE.write(idx, message)


def comm_messages(channel: int) -> np.ndarray:
    """Every live aircraft's message on ``channel``, 0.0 (silence) if unset."""
    channel = int(channel)
    return np.array(
        [m.get(channel, 0.0) for m in _COMM_MESSAGE.read(range(int(bs.traf.ntraf)))],
        dtype=np.float64,
    )


# RNG for receiver-side communication-channel noise (IntruderCommMessage's
# ``noise_std``). Reseeded from the episode seed at reset so training rollouts
# stay reproducible.
_COMM_NOISE_RNG = np.random.default_rng()


@on_reset
def _reseed_comm_noise(seed: int | None) -> None:
    global _COMM_NOISE_RNG
    _COMM_NOISE_RNG = np.random.default_rng(seed)


# Each aircraft's action masks as its last action set them: ``{target: masked}``,
# each target the name of the action masked. Written by ActionMask when the
# action is applied, read back by it and by PrevActionMasked.
_ACTION_MASKS = AircraftMemory(default={})


def record_action_mask(idx: int, target: str, masked: bool) -> None:
    """Store whether an aircraft's last action masked ``target``."""
    masks = dict(_ACTION_MASKS.read_one(idx))
    masks[target] = bool(masked)
    _ACTION_MASKS.write(idx, masks)


def action_masks(target: str, indices) -> np.ndarray:
    """Whether each aircraft's last action masked ``target``; False if unset."""
    return np.array(
        [m.get(target, False) for m in _ACTION_MASKS.read(indices)], dtype=bool
    )


# Each aircraft's locked actions, by target name: the axis each commands, the
# error on it last seen while locked (None until the first reading), and the sim
# time a timed lock ends (None for one held until captured). A mask locking
# until captured releases its lock the first step that error does not shrink;
# one locking for the clearance's duration, when it runs out. Written and read
# by ActionMask.
_ACTION_LOCKS = AircraftMemory(default={})


def action_lock(
    idx: int, target: str
) -> tuple[Any, float | None, float | None] | None:
    """``(axis, last_error, until)`` of ``target``'s lock for the aircraft at
    ``idx``, or ``None`` when it is not locked."""
    return _ACTION_LOCKS.read_one(idx).get(target)


def set_action_lock(
    idx: int,
    target: str,
    axis: Any = None,
    error: float | None = None,
    until: float | None = None,
) -> None:
    """Lock ``target`` - on ``axis``, the error on it last ``error``, until sim
    time ``until`` if timed - for the aircraft at ``idx``; with no ``axis``,
    release it."""
    locks = dict(_ACTION_LOCKS.read_one(idx))
    if axis is None:
        locks.pop(target, None)
    else:
        locks[target] = (axis, error, until)
    _ACTION_LOCKS.write(idx, locks)


# Each aircraft's target arrival time over each fix of its route, by route
# index, in simulator seconds - None for a fix with none. Assigned at spawn
# (sim.arrival), read by the ActiveRouteWaypoint arrival fields.
_ARRIVAL_TIMES = AircraftMemory(default=())


def set_arrival_times(idx: int, times: tuple[float | None, ...]) -> None:
    """Store the aircraft at ``idx``'s arrival time over each route fix."""
    _ARRIVAL_TIMES.write(idx, tuple(times))


def arrival_time(idx: int, route_index: int) -> float | None:
    """The aircraft at ``idx``'s arrival time over route fix ``route_index``."""
    times = _ARRIVAL_TIMES.read_one(idx)
    return times[route_index] if 0 <= route_index < len(times) else None



# Each aircraft's temporary clearances, by target name: the axis each commands
# and the sim time it expires at. Set when a target with a ClearanceDuration is
# applied; expire_clearances resumes own navigation when they run out.
_CLEARANCE_EXPIRY = AircraftMemory(default={})


def clearance_expiries(idx: int) -> dict[str, tuple[Any, float]]:
    """The aircraft at ``idx``'s running clearances: target -> (axis, expiry s)."""
    return dict(_CLEARANCE_EXPIRY.read_one(idx))


def set_clearance_expiry(
    idx: int, target: str, axis: Any = None, expires_s: float | None = None
) -> None:
    """Start (or restart) ``target``'s clearance for the aircraft at ``idx``,
    expiring at sim time ``expires_s``; with no ``axis``, end it."""
    running = dict(_CLEARANCE_EXPIRY.read_one(idx))
    if axis is None:
        running.pop(target, None)
    else:
        running[target] = (axis, float(expires_s))
    _CLEARANCE_EXPIRY.write(idx, running)
