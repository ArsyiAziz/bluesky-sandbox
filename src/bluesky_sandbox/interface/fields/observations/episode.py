"""An aircraft's episode: its time in the environment and its previous action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

import numpy as np

from .. import _state
from .._common import _BroadcastObs, _indices_array
from .._state import (
    _LAST_NORM_ACTION,
    _TIME_IN_ENV,
    _LastActionBacked,
    _TimeInEnvBacked,
)
from ..base import ObsField, ObsMeta, ObsQuantity, Unit


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
