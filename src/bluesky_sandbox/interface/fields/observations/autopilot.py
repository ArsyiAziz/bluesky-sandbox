"""The autopilot's selections - heading, speed, altitude, LNAV/VNAV - and the
aircraft's error from each.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

import bluesky as bs
import numpy as np

from .._common import (
    _MIN_DYNAMIC_SPAN,
    _BroadcastObs,
    _indices_array,
    _InFeet,
    _InKnots,
    _InMeters,
    _InMetersPerSecond,
)
from bluesky_sandbox.sim.performance.speeds import selected_cas_ms

from ..base import ObsField, ObsMeta, ObsQuantity, Unit
from .kinematics import (
    _Altitude,
    _AltitudeEnvelopeBounds,
    _CasEnvelopeBounds,
    _Speed,
    _UnitField,
)


@dataclass(frozen=True)
class _ApAltitude(_Altitude):
    """Autopilot selected altitude."""

    def _si_values(self, indices: np.ndarray) -> np.ndarray:
        return bs.traf.selalt[indices]

    def _si_expected(self, idx: int) -> float:
        return float(bs.traf.selalt[idx])


@dataclass(frozen=True)
class _ApAltitudeError(_AltitudeEnvelopeBounds, _UnitField):
    """Autopilot selected altitude minus current altitude."""

    low: Annotated[float | None, "lower bound; None = runtime altitude envelope"] = None
    high: Annotated[float | None, "upper bound; None = runtime altitude envelope"] = (
        None
    )

    def _si_values(self, indices: np.ndarray) -> np.ndarray:
        return bs.traf.selalt[indices] - bs.traf.alt[indices]

    def _si_expected(self, idx: int) -> float:
        return float(bs.traf.selalt[idx]) - float(bs.traf.alt[idx])

    def bounds(self, idx: int) -> tuple[float, float]:
        def resolve() -> tuple[float, float]:
            current = self._convert(bs.traf.alt[idx])
            ceiling = self._convert(self._altitude_ceiling_m(idx))
            span = max(
                max(abs(current), abs(ceiling - current)),
                _MIN_DYNAMIC_SPAN,
            )
            return -span, span

        return self._dynamic_or_configured_bounds(resolve)


# The selected speed as CAS: above the crossover BlueSky holds a Mach there.
@dataclass(frozen=True)
class _ApCas(_CasEnvelopeBounds, _Speed):
    def _si_values(self, indices: np.ndarray) -> np.ndarray:
        return selected_cas_ms(indices)

    def _si_expected(self, idx: int) -> float:
        return float(selected_cas_ms(idx)[0])


@dataclass(frozen=True)
class _ApCasError(_CasEnvelopeBounds, _Speed):
    def _si_values(self, indices: np.ndarray) -> np.ndarray:
        return selected_cas_ms(indices) - bs.traf.cas[indices]

    def _si_expected(self, idx: int) -> float:
        return float(selected_cas_ms(idx)[0]) - float(bs.traf.cas[idx])

    def bounds(self, idx: int) -> tuple[float, float]:
        # Symmetric about the current speed, wide enough to reach either end
        # of the envelope - not the envelope itself, which is for speeds.
        def resolve() -> tuple[float, float]:
            lo, hi = self._speed_bounds_ms(idx)
            current = bs.traf.cas[idx]
            span = max(
                self._convert(max(abs(current - lo), abs(hi - current))),
                _MIN_DYNAMIC_SPAN,
            )
            return -span, span

        return self._dynamic_or_configured_bounds(resolve)


@dataclass(frozen=True)
class ApHdgDeg(_BroadcastObs, ObsField):
    """Autopilot selected heading in degrees."""

    meta = ObsMeta("ap_hdg_deg", Unit.DEG, ObsQuantity.HEADING, circular=True)
    low: Annotated[float, "autopilot heading degrees"] = 0.0
    high: Annotated[float, "autopilot heading degrees"] = 360.0

    def _values(self, indices: Any) -> Any:
        return bs.traf.ap.trk[_indices_array(indices)]

    def expected(self, idx: int) -> Any:
        return float(bs.traf.ap.trk[idx])

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class ApCasKts(_InKnots, _ApCas):
    """Autopilot selected calibrated airspeed in knots."""

    meta = ObsMeta("ap_cas_kts", Unit.KTS, ObsQuantity.SPEED, dynamic_bounds=True)


@dataclass(frozen=True)
class ApCasMs(_InMetersPerSecond, _ApCas):
    """Autopilot selected calibrated airspeed in m/s."""

    meta = ObsMeta("ap_cas_ms", Unit.M_PER_S, ObsQuantity.SPEED, dynamic_bounds=True)


@dataclass(frozen=True)
class ApAltFt(_InFeet, _ApAltitude):
    """Autopilot selected altitude in feet."""

    meta = ObsMeta("ap_alt_ft", Unit.FT, ObsQuantity.ALTITUDE, dynamic_bounds=True)


@dataclass(frozen=True)
class ApAltM(_InMeters, _ApAltitude):
    """Autopilot selected altitude in meters."""

    meta = ObsMeta("ap_alt_m", Unit.M, ObsQuantity.ALTITUDE, dynamic_bounds=True)


@dataclass(frozen=True)
class ApLnavOn(_BroadcastObs, ObsField):
    """Whether LNAV is enabled: the aircraft flies its own route, rather than a
    heading it was given."""

    meta = ObsMeta("ap_lnav_on", Unit.SWITCH, ObsQuantity.AUTOPILOT)
    low: Annotated[float, "autopilot switch off"] = 0.0
    high: Annotated[float, "autopilot switch on"] = 1.0

    def _values(self, indices: Any) -> Any:
        return np.asarray(bs.traf.swlnav[_indices_array(indices)], dtype=bool)

    def expected(self, idx: int) -> Any:
        return float(bool(bs.traf.swlnav[idx]))

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class ApVnavOn(_BroadcastObs, ObsField):
    """Whether VNAV is enabled: the aircraft flies its route's levels, rather
    than an altitude it was given."""

    meta = ObsMeta("ap_vnav_on", Unit.SWITCH, ObsQuantity.AUTOPILOT)
    low: Annotated[float, "autopilot switch off"] = 0.0
    high: Annotated[float, "autopilot switch on"] = 1.0

    def _values(self, indices: Any) -> Any:
        return np.asarray(bs.traf.swvnav[_indices_array(indices)], dtype=bool)

    def expected(self, idx: int) -> Any:
        return float(bool(bs.traf.swvnav[idx]))

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class ApLnavVnavOn(_BroadcastObs, ObsField):
    """Whether both LNAV and VNAV are enabled."""

    meta = ObsMeta("ap_lnav_vnav_on", Unit.SWITCH, ObsQuantity.AUTOPILOT)
    low: Annotated[float, "autopilot switch off"] = 0.0
    high: Annotated[float, "autopilot switch on"] = 1.0

    def _values(self, indices: Any) -> Any:
        indices = _indices_array(indices)
        return np.logical_and(bs.traf.swlnav[indices], bs.traf.swvnav[indices])

    def expected(self, idx: int) -> Any:
        return float(bool(bs.traf.swlnav[idx]) and bool(bs.traf.swvnav[idx]))

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class ApHdgErrorDeg(_BroadcastObs, ObsField):
    """Autopilot selected heading error relative to current track in degrees."""

    meta = ObsMeta("ap_hdg_error_deg", Unit.DEG, ObsQuantity.HEADING)
    low: Annotated[float, "autopilot heading error degrees"] = -180.0
    high: Annotated[float, "autopilot heading error degrees"] = 180.0

    def _values(self, indices: Any) -> Any:
        indices = _indices_array(indices)
        return (bs.traf.ap.trk[indices] - bs.traf.trk[indices] + 540.0) % 360.0 - 180.0

    def expected(self, idx: int) -> Any:
        error = float(bs.traf.ap.trk[idx]) - float(bs.traf.trk[idx])
        return (error + 180.0) % 360.0 - 180.0

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class ApCasErrorKts(_InKnots, _ApCasError):
    """Autopilot selected CAS error relative to current CAS in knots."""

    meta = ObsMeta(
        "ap_cas_error_kts",
        Unit.KTS,
        ObsQuantity.SPEED,
        dynamic_bounds=True,
    )


@dataclass(frozen=True)
class ApAltErrorFt(_InFeet, _ApAltitudeError):
    """Autopilot selected altitude error relative to current altitude in feet."""

    meta = ObsMeta(
        "ap_alt_error_ft",
        Unit.FT,
        ObsQuantity.ALTITUDE,
        dynamic_bounds=True,
    )


@dataclass(frozen=True)
class ApAltErrorM(_InMeters, _ApAltitudeError):
    """Autopilot selected altitude error relative to current altitude in meters."""

    meta = ObsMeta(
        "ap_alt_error_m",
        Unit.M,
        ObsQuantity.ALTITUDE,
        dynamic_bounds=True,
    )
