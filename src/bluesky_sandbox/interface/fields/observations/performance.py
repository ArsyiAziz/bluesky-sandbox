"""Aircraft performance: capability descriptors and the flight phase.

The descriptors are continuous physical performance parameters, for
heterogeneous-fleet tasks: an aircraft is a *point in capability space*, not a
type symbol, so a policy conditioned on these generalizes to types never seen
in training (a learned type-ID embedding cannot). Realistically available -
type is broadcast in ADS-B and known to ATC. Usable in both the own and
intruder blocks (plain ObsFields, like ``VsFtMin``). Bounds are FIXED
fleet-wide scales on purpose: envelope-dynamic bounds would normalize each
aircraft's capability to itself, erasing exactly the cross-type differences
these fields carry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Any

import bluesky as bs
import numpy as np
from bluesky.tools.aero import ft, g0, kts, nm

from bluesky_sandbox.sim.performance.envelope import (
    _warn_type_data_mismatch,
    active_performance_model,
)
from bluesky_sandbox.sim.performance.models import type_limits

from .._common import _M_TO_FT, _MS_TO_FTMIN, _MS_TO_KTS, _BroadcastObs, _indices_array
from ..base import ObsField, ObsMeta, ObsQuantity, Unit


@dataclass(frozen=True)
class PerfVminKts(_BroadcastObs, ObsField):
    """Minimum operating CAS from the performance model, in knots.

    How slow this airframe *can* fly - the sequencing floor: an intruder with
    a higher ``vmin`` than mine cannot match my hold speed and must be led,
    not followed. Reads the live performance model (state-dependent through
    configuration/phase).
    """

    meta = ObsMeta("perf_vmin_kts", Unit.KTS, ObsQuantity.SPEED)
    low: Annotated[float, "fleet-wide CAS scale (low), knots"] = 60.0
    high: Annotated[float, "fleet-wide CAS scale (high), knots"] = 250.0

    def _values(self, indices: Any) -> Any:
        return bs.traf.perf.vmin[_indices_array(indices)] * _MS_TO_KTS

    def expected(self, idx: int) -> Any:
        return float(bs.traf.perf.vmin[idx]) / kts

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class PerfVmaxKts(_BroadcastObs, ObsField):
    """Maximum operating CAS from the performance model, in knots.

    How fast this airframe *can* fly - whether the aircraft ahead can
    accelerate out of the way, or I can close a slot. Reads the live
    performance model.
    """

    meta = ObsMeta("perf_vmax_kts", Unit.KTS, ObsQuantity.SPEED)
    low: Annotated[float, "fleet-wide CAS scale (low), knots"] = 120.0
    high: Annotated[float, "fleet-wide CAS scale (high), knots"] = 400.0

    def _values(self, indices: Any) -> Any:
        return bs.traf.perf.vmax[_indices_array(indices)] * _MS_TO_KTS

    def expected(self, idx: int) -> Any:
        return float(bs.traf.perf.vmax[idx]) / kts

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class PerfVsMaxFtMin(_BroadcastObs, ObsField):
    """Maximum climb rate from the performance model, in ft/min.

    Vertical escape capacity - can this aircraft climb out of a conflict
    layer, and how fast. Reads the live performance model (varies with
    altitude/mass/phase).
    """

    meta = ObsMeta("perf_vs_max_ft_min", Unit.FT_PER_MIN, ObsQuantity.VERTICAL_SPEED)
    low: Annotated[float, "fleet-wide climb-rate scale (low), ft/min"] = 0.0
    high: Annotated[float, "fleet-wide climb-rate scale (high), ft/min"] = 6000.0

    def _values(self, indices: Any) -> Any:
        return bs.traf.perf.vsmax[_indices_array(indices)] * _MS_TO_FTMIN

    def expected(self, idx: int) -> Any:
        return float(bs.traf.perf.vsmax[idx]) * 60.0 / ft

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class PerfVsMinFtMin(_BroadcastObs, ObsField):
    """Maximum DESCENT rate from the performance model, in ft/min (negative).

    The counterpart to :class:`PerfVsMaxFtMin`, which is climb only. Distinct
    physics and a distinct number - on the openap model the two differ by ~20%
    for the same aircraft - so a descending task cannot substitute one for the
    other.
    """

    meta = ObsMeta("perf_vs_min_ft_min", Unit.FT_PER_MIN, ObsQuantity.VERTICAL_SPEED)
    low: Annotated[float, "fleet-wide descent-rate scale (low), ft/min"] = -6000.0
    high: Annotated[float, "fleet-wide descent-rate scale (high), ft/min"] = 0.0

    def _values(self, indices: Any) -> Any:
        return bs.traf.perf.vsmin[_indices_array(indices)] * _MS_TO_FTMIN

    def expected(self, idx: int) -> Any:
        return float(bs.traf.perf.vsmin[idx]) * 60.0 / ft

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class PerfCeilingFt(_BroadcastObs, ObsField):
    """Altitude ceiling from the performance model, in ft.

    How much vertical room is left above. Also the quantity that scales a
    waypoint-relative altitude *action* whenever the ceiling term wins
    (``span = max(wpalt, ceiling - wpalt)``) - measured on safe_rl_v38d at ~9% of
    steps - so without this the policy cannot know how many feet its normalized
    altitude action commands.
    """

    meta = ObsMeta("perf_ceiling_ft", Unit.FT, ObsQuantity.ALTITUDE)
    low: Annotated[float, "fleet-wide ceiling scale (low), ft"] = 0.0
    high: Annotated[float, "fleet-wide ceiling scale (high), ft"] = 60000.0

    def _values(self, indices: Any) -> Any:
        return bs.traf.perf.hmax[_indices_array(indices)] * _M_TO_FT

    def expected(self, idx: int) -> Any:
        return float(bs.traf.perf.hmax[idx]) / ft

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class PerfMassT(_BroadcastObs, ObsField):
    """CURRENT aircraft mass from the performance model, in tonnes.

    Unlike :class:`MtowT`, which is a static per-type constant, this is the live
    mass the performance model is actually flying, and it drives thrust-to-weight,
    achievable rates and turn performance. Spans ~5-190 t across the allowed fleet.
    """

    meta = ObsMeta("perf_mass_t", Unit.T, ObsQuantity.MASS)
    low: Annotated[float, "fleet-wide mass scale (low), t"] = 0.0
    high: Annotated[float, "fleet-wide mass scale (high), t"] = 600.0

    def _values(self, indices: Any) -> Any:
        return bs.traf.perf.mass[_indices_array(indices)] / 1000.0

    def expected(self, idx: int) -> Any:
        return float(bs.traf.perf.mass[idx]) / 1000.0

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class TurnRadiusNm(_BroadcastObs, ObsField):
    """Coordinated-turn radius at current TAS and bank-angle limit, in nm.

    ``R = V^2 / (g * tan(phi))`` with ``V = bs.traf.tas`` and ``phi`` the
    per-aircraft bank limit (``bs.traf.ap.bankdef``, default 25 deg) - the same
    triangle BlueSky's heading dynamics integrate, so this is the radius the
    aircraft actually flies at full authority. Deliberately *state-dependent*
    (scales with V^2): it reads as current agility, and slowing down visibly
    shrinks it - the physical lever behind decelerate-before-capture. At
    cruise (480 kt, 25 deg) it is ~7 nm - larger than a typical reach radius,
    which is why an overshoot costs a full circuit.
    """

    meta = ObsMeta("turn_radius_nm", Unit.NM, ObsQuantity.DISTANCE)
    low: Annotated[float, "turn radius scale (low), nm"] = 0.0
    high: Annotated[float, "turn radius scale (high), nm"] = 20.0

    def _values(self, indices: Any) -> Any:
        indices = _indices_array(indices)
        tas = np.maximum(np.asarray(bs.traf.tas)[indices], 1e-6)
        # Bank *limit* (authority), not ap.turnphi - turnphi is a transient
        # commanded bank inside flyturn legs and zero otherwise.
        phi = np.asarray(bs.traf.ap.bankdef)[indices]
        return (tas * tas) / (g0 * np.tan(phi)) / nm

    def expected(self, idx: int) -> Any:
        # r = v^2 / (g tan(bank)), at the aircraft's bank limit.
        tas = max(float(bs.traf.tas[idx]), 1e-6)
        bank = float(bs.traf.ap.bankdef[idx])
        return tas * tas / (g0 * math.tan(bank)) / nm

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


# MTOW per type, cached: openap's aircraft database, with a medium-class
# fallback for types it does not know (never raise mid-episode over a
# descriptor lookup).
_MTOW_KG_CACHE: dict[str, float] = {}
_MTOW_FALLBACK_KG = 100_000.0


def _mtow_kg(actype: str) -> float:
    key = str(actype).upper()
    cached = _MTOW_KG_CACHE.get(key)
    if cached is None:
        # Same registry the envelope uses, so MTOW and the flight envelope can
        # never come from different databases for one aircraft.
        model = active_performance_model()
        mtow = (type_limits(key, model) or {}).get("MTOW")
        if mtow is None and model != "openap":
            _warn_type_data_mismatch("MTOW")
            mtow = (type_limits(key, "openap") or {}).get("MTOW")
        cached = float(mtow) if mtow else _MTOW_FALLBACK_KG
        _MTOW_KG_CACHE[key] = cached
    return cached


@dataclass(frozen=True)
class MtowT(_BroadcastObs, ObsField):
    """Maximum takeoff weight of the aircraft type, in tonnes.

    The continuous stand-in for wake/size class (ICAO wake categories are
    MTOW bands): a smooth mass descriptor generalizes where a categorical
    one-hot cannot. Looked up once per type from the openap aircraft
    database and cached; unknown types fall back to a medium-class 100 t.
    """

    meta = ObsMeta("mtow_t", Unit.T, ObsQuantity.MASS)
    low: Annotated[float, "fleet-wide mass scale (low), tonnes"] = 0.0
    high: Annotated[float, "fleet-wide mass scale (high), tonnes"] = 600.0

    def _values(self, indices: Any) -> Any:
        types = bs.traf.type
        return np.asarray(
            [_mtow_kg(types[int(i)]) / 1000.0 for i in _indices_array(indices)],
            dtype=np.float64,
        )

    def expected(self, idx: int) -> Any:
        return _mtow_kg(bs.traf.type[idx]) / 1000.0

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


def _scalar_value(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode()
    return value


def _phase_matches(left: Any, right: Any) -> bool:
    left = _scalar_value(left)
    right = _scalar_value(right)
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return str(left).casefold() == str(right).casefold()


def _phase_key(value: Any) -> tuple[str, float | str]:
    value = _scalar_value(value)
    try:
        return "number", float(value)
    except (TypeError, ValueError):
        return "label", str(value).casefold()


@dataclass(frozen=True)
class FlightPhaseOneHot(_BroadcastObs, ObsField):
    """Flight phase as a one-hot vector.

    The default ``phase_values`` encode BlueSky/OpenAP-style raw phase codes
    ``0..6``. Pass a custom tuple when using a performance model that emits a
    different code or label set.
    """

    meta = ObsMeta("flight_phase_one_hot", Unit.UNITLESS, ObsQuantity.PHASE)
    phase_values: tuple[int | float | str, ...] = (0, 1, 2, 3, 4, 5, 6)
    unknown_index: Annotated[
        int | None,
        "index to activate when the raw phase is not in phase_values; None = all zero",
    ] = 0
    low: Annotated[float, "one-hot off value"] = 0.0
    high: Annotated[float, "one-hot on value"] = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.phase_values:
            raise ValueError("FlightPhaseOneHot.phase_values cannot be empty")
        normalized = tuple(_phase_key(value) for value in self.phase_values)
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                "FlightPhaseOneHot.phase_values must not contain duplicates"
            )
        if self.unknown_index is not None and not (
            0 <= self.unknown_index < len(self.phase_values)
        ):
            raise ValueError(
                "FlightPhaseOneHot.unknown_index must be a valid phase index or None"
            )

    def output_size(self) -> int:
        return len(self.phase_values)

    def _values(self, indices: Any) -> Any:
        indices = _indices_array(indices)
        values = np.zeros((indices.size, len(self.phase_values)), dtype=np.float32)
        for row, raw_phase in enumerate(bs.traf.perf.phase[indices]):
            for phase_idx, phase_value in enumerate(self.phase_values):
                if _phase_matches(raw_phase, phase_value):
                    values[row, phase_idx] = 1.0
                    break
            else:
                if self.unknown_index is not None:
                    values[row, self.unknown_index] = 1.0
        return values

    def expected(self, idx: int) -> Any:
        values = np.zeros(len(self.phase_values), dtype=np.float32)
        phase = bs.traf.perf.phase[idx]
        matches = [
            k for k, v in enumerate(self.phase_values) if _phase_matches(phase, v)
        ]
        if matches:
            values[matches[0]] = 1.0
        elif self.unknown_index is not None:
            values[self.unknown_index] = 1.0
        return values

    def bounds(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        size = self.output_size()
        return (
            np.full(size, float(self.low), dtype=np.float32),
            np.full(size, float(self.high), dtype=np.float32),
        )
