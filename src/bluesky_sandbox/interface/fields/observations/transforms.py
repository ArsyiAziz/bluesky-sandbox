"""Fields built from other fields: differences and lagged (frame-stacked)
values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import bluesky as bs
import numpy as np

from .._common import _BroadcastObs, _indices_array
from .._lag import _lag_ring, _LagHistoryBacked, _register_lag_depth
from .._pairs import _BroadcastPairs, _per_aircraft
from ..base import ObsField, ObsMeta, PairObsField, Unit


def _with_derived_bounds(
    field: ObsField,
    derived: Mapping[str, tuple[float, float]],
) -> ObsField:
    bounds = derived.get(field.meta.name)
    if bounds is not None and not field.bounds_overridden:
        return replace(field, low=bounds[0], high=bounds[1])
    return field


@dataclass(frozen=True)
class LaggedObs(_BroadcastObs, _LagHistoryBacked, ObsField):
    """An ownship field's value from ``steps`` environment steps ago.

    Built by :meth:`ObsField.lagged`. Bounds, normalizer and output size all
    delegate to ``inner``, so a stacked channel lands on exactly the same scale
    as the live one and needs no separate calibration.

    The lag counts OBSERVATION QUERIES at distinct sim times, not wall steps.
    Training observes every live agent every step so the two coincide; a caller
    that observes a subset of agents would see that subset's own lag.
    """

    inner: ObsField | None = None
    steps: int = 1

    @property
    def meta(self) -> ObsMeta:
        # ``dynamic_bounds=True`` regardless of the inner field's policy, matching
        # :class:`Difference`: the bounds are resolved from ``inner`` at runtime
        # (``bounds`` below), so this wrapper carries no static defaults of its own
        # for ``_validate_bound_policy`` to check.
        inner = self._field()
        return replace(
            inner.meta,
            name=f"{inner.meta.name}_lag{int(self.steps)}",
            dynamic_bounds=True,
        )

    def __post_init__(self) -> None:
        inner = self._field()
        if int(self.steps) < 1:
            raise ValueError(f"lagged(steps=) must be >= 1, got {self.steps}.")
        object.__setattr__(self, "_key", repr(inner))
        _register_lag_depth(self._key, self.steps)
        super().__post_init__()

    def _field(self) -> ObsField:
        if not isinstance(self.inner, ObsField):
            raise TypeError(f"LaggedObs.inner must be an ObsField, got {self.inner!r}.")
        return self.inner

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._field().bounds(idx)

    def output_size(self) -> int:
        inner = self._field()
        size = getattr(inner, "output_size", None)
        return int(size()) if callable(size) else 1

    def _values(self, indices: Any) -> Any:
        inner = self._field()
        idxs = _indices_array(indices).ravel()
        ring = _lag_ring("obs", self._key, self.steps)
        simt = float(bs.sim.simt)
        if ring.last_push_simt != simt:
            ring.last_push_simt = simt
            rows = np.unique(idxs)
            if rows.size:
                ring.push(inner.get_many(rows), rows)
        out, missing = ring.read(self.steps, idxs)
        if out is None or missing.any():
            # Aircraft that appeared after this step's push (or the push was made
            # by a sibling lag before it existed): its own value is the only
            # history there is.
            live = np.asarray(inner.get_many(idxs), dtype=np.float64)
            if out is None:
                return live
            out[missing] = live[missing]
        return out


@dataclass(frozen=True)
class LaggedPair(_LagHistoryBacked, PairObsField):
    """An intruder pair field's value from ``steps`` environment steps ago.

    Built by :meth:`PairObsField.lagged`. History is keyed by the ORDERED
    callsign pair, which is what makes this correct at all: BlueSky compacts its
    arrays with ``np.delete`` on every despawn, so intruder row ``k`` at step
    ``t`` is routinely a different aircraft than row ``k`` at ``t-1``.

    Only sound on fields that are invariant to ownship ROTATION. Anything
    expressed in the ownship's track frame (``RelPos*``, ``RelVel*``, the
    along/cross realized accelerations) was computed in the old frame, so
    stacking it aliases the ownship's own turning as intruder motion.
    """

    inner: PairObsField | None = None
    steps: int = 1

    @property
    def meta(self) -> ObsMeta:
        # See :class:`LaggedObs.meta` for why this is always dynamic.
        inner = self._field()
        return replace(
            inner.meta,
            name=f"{inner.meta.name}_lag{int(self.steps)}",
            dynamic_bounds=True,
        )

    def __post_init__(self) -> None:
        inner = self._field()
        if int(self.steps) < 1:
            raise ValueError(f"lagged(steps=) must be >= 1, got {self.steps}.")
        object.__setattr__(self, "_key", repr(inner))
        _register_lag_depth(self._key, self.steps)
        super().__post_init__()

    def _field(self) -> PairObsField:
        if not isinstance(self.inner, PairObsField):
            raise TypeError(
                f"LaggedPair.inner must be a PairObsField, got {self.inner!r}."
            )
        return self.inner

    def bounds(self, own_idx: int) -> tuple[float, float]:
        return self._field().bounds(own_idx)

    def output_size(self) -> int:
        inner = self._field()
        size = getattr(inner, "output_size", None)
        return int(size()) if callable(size) else 1

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return self.get_pairs(own_idx, [int(other_idx)])[0]

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        own = np.array([int(own_idx)], dtype=np.intp)
        return self._lagged(own, _indices_array(other_indices).ravel())[0]

    def get_pair_matrix(self, own_indices: Any) -> np.ndarray:
        every = np.arange(int(bs.traf.ntraf), dtype=np.intp)
        return self._lagged(_indices_array(own_indices).ravel(), every)

    def _lagged(self, owns: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """The inner field ``steps`` pushes back, for each (own, col) pair.

        Each ownship's row is pushed once per sim time, by its first query, with
        the pairs that query asked for.
        """
        inner = self._field()
        ring = _lag_ring("pair", self._key, self.steps)
        simt = float(bs.sim.simt)
        due = np.unique(owns[ring.pushed_at[owns] != simt])
        if due.size and cols.size:
            current = np.asarray(inner.get_pair_matrix(due))[:, cols]
            ring.push(current, due, cols)
            ring.pushed_at[due] = simt
        out, missing = ring.read(self.steps, owns, cols)
        if out is None or missing.any():
            # Pairs that appeared after this step's push - hold their current
            # value (see :meth:`_LagRing.read`).
            live = np.asarray(inner.get_pair_matrix(owns), dtype=np.float64)[:, cols]
            if out is None:
                return live
            out[missing] = live[missing]
        return out


@dataclass(frozen=True)
class Difference(_BroadcastPairs, PairObsField):
    """Difference between an intruder field and an ownship field.

    ``left`` is read from the intruder index and ``right`` is read from the
    ownship index. Use :class:`AngleDifference` for circular degree fields.
    """

    left: ObsField | None = None
    right: ObsField | None = None
    name: str | None = None

    @property
    def meta(self) -> ObsMeta:
        left, right = self._fields()
        return ObsMeta(
            self.name or f"{left.meta.name}_minus_{right.meta.name}",
            left.meta.unit,
            left.meta.quantity,
            is_pair=True,
            dynamic_bounds=True,
        )

    def __post_init__(self) -> None:
        left, right = self._fields()
        if left.meta.unit != right.meta.unit:
            raise ValueError(
                "Difference fields must use matching units; got "
                f"{left.meta.unit!r} and {right.meta.unit!r}."
            )
        if left.meta.circular or right.meta.circular:
            raise ValueError(
                "Difference cannot subtract circular fields. Use AngleDifference."
            )
        super().__post_init__()

    def _fields(self) -> tuple[ObsField, ObsField]:
        if not isinstance(self.left, ObsField):
            raise TypeError(f"Difference.left must be an ObsField, got {self.left!r}.")
        if not isinstance(self.right, ObsField):
            raise TypeError(
                f"Difference.right must be an ObsField, got {self.right!r}."
            )
        return self.left, self.right

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        left, right = self._fields()
        return _per_aircraft(left, other) - _per_aircraft(right, own)

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        left, right = self._fields()
        return left.expected(other_idx) - right.expected(own_idx)

    def bounds(self, own_idx: int) -> tuple[float, float]:
        if self.bounds_overridden:
            return self._configured_bounds()
        left, right = self._fields()
        left_low, left_high = left.bounds(own_idx)
        right_low, right_high = right.bounds(own_idx)
        return left_low - right_high, left_high - right_low

    def with_derived_bounds(
        self,
        derived: Mapping[str, tuple[float, float]],
    ) -> Difference:
        left, right = self._fields()
        return replace(
            self,
            left=_with_derived_bounds(left, derived),
            right=_with_derived_bounds(right, derived),
        )


@dataclass(frozen=True)
class AngleDifference(Difference):
    """Wrapped degree difference between an intruder field and an ownship field."""

    @property
    def meta(self) -> ObsMeta:
        left, right = self._fields()
        return ObsMeta(
            self.name or f"{left.meta.name}_angle_minus_{right.meta.name}",
            Unit.DEG,
            left.meta.quantity,
            is_pair=True,
            circular=True,
            dynamic_bounds=True,
        )

    def __post_init__(self) -> None:
        left, right = self._fields()
        if left.meta.unit != Unit.DEG or right.meta.unit != Unit.DEG:
            raise ValueError(
                "AngleDifference fields must use degree fields; got "
                f"{left.meta.unit!r} and {right.meta.unit!r}."
            )
        if not (left.meta.circular and right.meta.circular):
            raise ValueError(
                "AngleDifference fields must be circular. Use Difference for "
                "non-circular degree fields."
            )
        PairObsField.__post_init__(self)

    def _pairs(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        left, right = self._fields()
        delta = _per_aircraft(left, other) - _per_aircraft(right, own)
        return (delta + 540.0) % 360.0 - 180.0

    def expected_pair(self, own_idx: int, other_idx: int) -> Any:
        left, right = self._fields()
        delta = left.expected(other_idx) - right.expected(own_idx)
        return (delta + 180.0) % 360.0 - 180.0

    def bounds(self, own_idx: int) -> tuple[float, float]:
        if self.bounds_overridden:
            return self._configured_bounds()
        return -180.0, 180.0
