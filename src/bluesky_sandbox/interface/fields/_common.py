"""Unit conversions and index helpers shared by the field modules."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import bluesky as bs
import numpy as np
from bluesky.tools.aero import ft, kts

_M_TO_FT = 1.0 / ft
_MS_TO_KTS = 1.0 / kts
_MS_TO_FTMIN = 60.0 / ft
_MIN_DYNAMIC_SPAN = 1e-6
# Groundspeed floor for an ETE denominator: airborne traffic never reaches it,
# it only keeps a division finite for a stopped/uninitialised aircraft.
_MIN_GS_MS = 1e-3


def _signed_angle_delta_deg(left: float, right: float) -> float:
    return (left - right + 540.0) % 360.0 - 180.0


def _indices_array(indices: Any) -> np.ndarray:
    return np.asarray(indices, dtype=np.intp)


def _traf_array(name: str) -> np.ndarray:
    return np.asarray(getattr(bs.traf, name), dtype=np.float64)


# Every per-process field store, as the function that resets it. Each store
# registers where it is defined, so none can be left out of a reset.
_RESETS: list[Callable[[int | None], None]] = []


def on_reset(fn: Callable[[int | None], None]) -> Callable[[int | None], None]:
    """Register ``fn(seed)`` to run whenever the environment resets field state."""
    _RESETS.append(fn)
    return fn


def reset_field_state(seed: int | None = None) -> None:
    """Reset every per-process field store; the environment calls this on reset.

    The stores are module-level - one BlueSky per process - so two envs built
    from different configs in one process share them, and without this the
    second would inherit what the first left in a store it has no field for.
    """
    for reset in _RESETS:
        reset(seed)


class _BroadcastObs:
    """An ownship field defined by one vectorized function, :meth:`_values`.

    ``get`` and ``get_many`` are that function on one index or many, so they
    agree by construction - the ownship counterpart of ``_BroadcastPairs``.
    """

    def _values(self, indices: np.ndarray) -> Any:
        raise NotImplementedError

    def get(self, idx: Any) -> Any:
        return self._values(_indices_array([int(idx)]))[0]

    def get_many(self, indices: Any) -> Any:
        return self._values(_indices_array(indices))
