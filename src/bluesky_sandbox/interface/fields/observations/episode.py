"""An aircraft's episode: its time in the environment and its previous action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

import bluesky as bs
import numpy as np

from .. import _state
from .._common import _BroadcastObs, _indices_array
from .._state import (
    _LAST_NORM_ACTION,
    _TIME_IN_ENV,
    _ActionMaskBacked,
    _LastActionBacked,
    _TimeInEnvBacked,
    action_masks,
    clearance_expiries,
    clearance_holds,
)
from ..actions.mask import action_name, lock_holds
from ..base import ActionField, ObsField, ObsMeta, ObsQuantity, Unit


@dataclass(frozen=True)
class TimeInEnvS(_BroadcastObs, _TimeInEnvBacked, ObsField):
    """Seconds since ownship entered the environment - its age, not sim clock.

    The quantity the time-limit truncation is stated against
    (``info["time_in_env"] >= TIME_BUDGET_S``), so with ``high`` set to that
    budget and a :class:`MinMaxNormalizer` this reads as episode progress in
    ``[0, 1]`` and time REMAINING is its complement.

    Why a value function wants it: under a time limit the return is bounded by
    the time left, so two otherwise identical states early and late in an
    aircraft's life have genuinely different values and a critic without this
    must average them. That is irreducible value error, not underfitting - the
    standard time-limit partial-observability result (Pardo et al. 2018). It
    matters most for the CONSTRAINT critic here, whose discounted cost-to-go at
    ``cost_gamma = 0.99`` reaches ~100 steps and so routinely runs past the
    truncation an early-life state still has ahead of it.

    Purely local (an aircraft knows its own age), so it is legitimate in
    ``obs_fields``; putting it in ``critic_obs_fields`` instead fixes the value
    function while leaving the policy's input distribution untouched.

    Published by the environment each step from its spawn-time bookkeeping (an
    ObsField cannot reach it - BlueSky keeps no per-aircraft age); reads 0 on the
    step an aircraft spawns.
    """

    meta = ObsMeta("time_in_env_s", Unit.S, ObsQuantity.TIME)
    low: Annotated[float, "seconds lower bound"] = 0.0
    high: Annotated[float, "seconds upper bound; set to the task's time budget"] = (
        3600.0
    )

    def _values(self, indices: Any) -> Any:
        return np.asarray(_TIME_IN_ENV.read(indices), dtype=np.float64)

    def _expected(self, idx: int) -> Any:
        return float(_TIME_IN_ENV.read_one(idx))  # published by the env

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class PrevActionNorm(_BroadcastObs, _LastActionBacked, ObsField):
    """Ownship's previous action ``a_{t-1}`` (as the policy emitted it).

    Surfaces the last policy output so an action-rate reward penalty
    ``|a_t - a_{t-1}|`` is Markovian w.r.t. the observation. Set ``dim`` to how
    many action components to expose and ``offset`` to where they start in the
    action vector.

    The stored value is the policy's action *in the task's action space*, whose
    range depends on the action fields' normalizers - ``SymmetricNormalizer`` ->
    ``[-1, 1]``, ``MinMaxNormalizer`` -> ``[0, 1]``, no normalizer -> the
    field's own bounds. Bounds are therefore **dynamic**: the environment
    publishes the live action-space bounds (:func:`set_action_space_bounds`) and
    this field slices them by ``offset``/``dim``, so it matches any action space
    automatically - including mixed per-component ranges - with no manual config.
    Passing ``low``/``high`` overrides that with a fixed range; before the env
    publishes bounds it falls back to ``[-1, 1]``. No normalizer is attached (the
    value is already in action space). Reads all-zero before the first action and
    on spawn (matching a first-step Δ of 0).
    """

    meta = ObsMeta(
        "prev_action_norm", Unit.UNITLESS, ObsQuantity.ACTION, dynamic_bounds=True
    )
    dim: Annotated[int, "number of action components to expose"] = 1
    offset: Annotated[int, "index of the first action component to expose"] = 0
    low: Annotated[float | None, "fixed lower bound; None = read from action space"] = (
        None
    )
    high: Annotated[
        float | None, "fixed upper bound; None = read from action space"
    ] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dim < 1:
            raise ValueError("PrevActionNorm.dim must be >= 1")
        if self.offset < 0:
            raise ValueError("PrevActionNorm.offset must be >= 0")
        if self.normalizer is not None:
            # The stored value is the policy's own output, already in action
            # space - scaling it a second time would measure it against bounds
            # it was never drawn from. Bounds here are also per-component (one
            # pair per exposed action slot), which a scalar Normalizer has no
            # way to reduce to a single span.
            raise ValueError(
                "PrevActionNorm carries the policy's action, already in action "
                "space; leave normalizer unset."
            )

    def output_size(self) -> int:
        return self.dim

    def bounds(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        size = self.output_size()
        if self.bounds_overridden:
            return (
                np.full(size, float(self.low), dtype=np.float32),
                np.full(size, float(self.high), dtype=np.float32),
            )
        if _state._ACTION_SPACE_BOUNDS is not None:
            lo, hi = _state._ACTION_SPACE_BOUNDS
            sl = slice(self.offset, self.offset + size)
            lo_s, hi_s = lo[sl], hi[sl]
            if lo_s.shape[0] == size:
                return (lo_s.astype(np.float32), hi_s.astype(np.float32))
        # Before the env publishes action bounds: assume symmetric [-1, 1].
        return (
            np.full(size, -1.0, dtype=np.float32),
            np.full(size, 1.0, dtype=np.float32),
        )

    def _values(self, indices: Any) -> Any:
        indices = _indices_array(indices)
        if indices.size == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((indices.size, self.dim), dtype=np.float32)
        for row, stored in enumerate(_LAST_NORM_ACTION.read(indices)):
            if stored is not None:
                src = stored[self.offset : self.offset + self.dim]
                out[row, : src.shape[0]] = src
        return out

    def _expected(self, idx: int) -> Any:
        values = np.zeros(self.dim, dtype=np.float32)
        stored = _LAST_NORM_ACTION.read_one(idx)
        if stored is not None:
            exposed = stored[self.offset : self.offset + self.dim]
            values[: len(exposed)] = exposed
        return values


@dataclass(frozen=True)
class PrevActionMasked(_BroadcastObs, _ActionMaskBacked, ObsField):
    """Whether ownship's previous action masked ``target``: 1 when an
    :class:`~bluesky_sandbox.interface.fields.actions.ActionMask` skipped it, 0
    when it let it through - and before the first action.

    ``target`` names the action as the mask does - its class, an instance of
    it, or its name. An action no mask targets always reads 0.
    """

    meta = ObsMeta("prev_action_masked", Unit.SWITCH, ObsQuantity.INDICATOR)
    target: Annotated[
        str | type[ActionField] | ActionField,
        "the masked action: its name, its class or an instance",
    ] = ""
    low: Annotated[float, "lower bound"] = 0.0
    high: Annotated[float, "upper bound"] = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", action_name(self.target))
        super().__post_init__()

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()

    def _values(self, indices: Any) -> Any:
        return action_masks(self.target, _indices_array(indices)).astype(np.float64)

    def _expected(self, idx: int) -> Any:
        masks = _state._ACTION_MASKS.read_one(idx)
        return 1.0 if masks.get(self.target, False) else 0.0


@dataclass(frozen=True)
class ActionLocked(_BroadcastObs, _ActionMaskBacked, ObsField):
    """1 while a clearance of ``target`` is still being flown, so a new one on
    its axis would be refused; else 0.

    For a target whose :class:`~bluesky_sandbox.interface.fields.actions.ActionMask`
    locks until captured: read by the same rule the dispatcher refuses by
    (:func:`~bluesky_sandbox.interface.fields.actions.mask.lock_holds`), so what
    the policy sees and what it is refused always agree. ``target`` names the
    action as the mask does.
    """

    meta = ObsMeta("action_locked", Unit.SWITCH, ObsQuantity.INDICATOR)
    target: Annotated[
        str | type[ActionField] | ActionField,
        "the locked action: its name, its class or an instance",
    ] = ""
    low: Annotated[float, "lower bound"] = 0.0
    high: Annotated[float, "upper bound"] = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", action_name(self.target))
        super().__post_init__()

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()

    def _values(self, indices: Any) -> Any:
        return np.array(
            [lock_holds(int(i), self.target) for i in _indices_array(indices)],
            dtype=np.float64,
        )

    def _expected(self, idx: int) -> Any:
        return 1.0 if lock_holds(idx, self.target) else 0.0


@dataclass(frozen=True)
class ClearanceTimeLeftS(_BroadcastObs, _ActionMaskBacked, ObsField):
    """Seconds left on the running temporary clearance of ``target``, 0 when
    none (see :class:`~bluesky_sandbox.interface.fields.actions.ClearanceDuration`).

    With it the policy knows a deviation is running and when own navigation
    takes the axis back. A clearance counted from capture shows its whole hold
    while it is still being flown. ``target`` names the action as its duration does.
    """

    meta = ObsMeta("clearance_time_left_s", Unit.S, ObsQuantity.TIME)
    target: Annotated[
        str | type[ActionField] | ActionField,
        "the timed action: its name, its class or an instance",
    ] = ""
    low: Annotated[float, "lower bound, s"] = 0.0
    high: Annotated[float, "upper bound, s: the longest clearance"] = 600.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", action_name(self.target))
        super().__post_init__()

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()

    def _left(self, idx: int) -> float:
        running = clearance_expiries(idx).get(self.target)
        if running is None:
            return 0.0
        if running[1] is None:
            # Still being flown: the whole hold is to come.
            return clearance_holds(idx)[self.target][1]
        return max(0.0, running[1] - float(bs.sim.simt))

    def _values(self, indices: Any) -> Any:
        return np.array(
            [self._left(int(i)) for i in _indices_array(indices)], dtype=np.float64
        )

    def _expected(self, idx: int) -> Any:
        return self._left(idx)
