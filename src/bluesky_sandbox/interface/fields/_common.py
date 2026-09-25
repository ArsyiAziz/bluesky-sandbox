"""Unit conversions and index helpers shared by the field modules."""

from __future__ import annotations

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
