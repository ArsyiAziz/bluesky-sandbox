"""An aircraft's kinematic state: position, heading and track, altitude,
speeds, and the envelope margins derived from them.

Read for whichever aircraft is observed: the ownship in ``obs_fields``, each
intruder in ``intruder_obs_fields``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import bluesky as bs
import numpy as np
from bluesky.tools.aero import crossoveralt, ft, vcas2tas

from .._common import (
    _M_TO_FT,
    _BroadcastObs,
    _indices_array,
    _InFeet,
    _InFeetPerMinute,
    _InKnots,
    _InMeters,
    _InMetersPerSecond,
)
from bluesky_sandbox.sim.performance.speeds import Crossover, above_crossover, mach_limit

from ..base import ObsField, ObsMeta, ObsQuantity, Unit


@dataclass(frozen=True)
class LatDeg(_BroadcastObs, ObsField):
    """Latitude in degrees."""

    meta = ObsMeta("lat_deg", Unit.DEG, ObsQuantity.LATITUDE)
    low: Annotated[float, "latitude degrees"] = -90.0
    high: Annotated[float, "latitude degrees"] = 90.0

    def _values(self, indices: Any) -> Any:
        return bs.traf.lat[_indices_array(indices)]

    def expected(self, idx: int) -> Any:
        return float(bs.traf.lat[idx])

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class LonDeg(_BroadcastObs, ObsField):
    """Longitude in degrees."""

    meta = ObsMeta("lon_deg", Unit.DEG, ObsQuantity.LONGITUDE)
    low: Annotated[float, "longitude degrees"] = -180.0
    high: Annotated[float, "longitude degrees"] = 180.0

    def _values(self, indices: Any) -> Any:
        return bs.traf.lon[_indices_array(indices)]

    def expected(self, idx: int) -> Any:
        return float(bs.traf.lon[idx])

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class HdgDeg(_BroadcastObs, ObsField):
    """Aircraft heading in degrees."""

    meta = ObsMeta("hdg_deg", Unit.DEG, ObsQuantity.HEADING, circular=True)
    low: Annotated[float, "heading degrees"] = 0.0
    high: Annotated[float, "heading degrees"] = 360.0

    def _values(self, indices: Any) -> Any:
        return bs.traf.hdg[_indices_array(indices)]

    def expected(self, idx: int) -> Any:
        return float(bs.traf.hdg[idx])

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class TrkDeg(_BroadcastObs, ObsField):
    """Aircraft track angle in degrees."""

    meta = ObsMeta("trk_deg", Unit.DEG, ObsQuantity.TRACK, circular=True)
    low: Annotated[float, "track degrees"] = 0.0
    high: Annotated[float, "track degrees"] = 360.0

    def _values(self, indices: Any) -> Any:
        return bs.traf.trk[_indices_array(indices)]

    def expected(self, idx: int) -> Any:
        return float(bs.traf.trk[idx])

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


class _UnitField(_BroadcastObs, ObsField):
    """A quantity computed in SI and reported in its variant's unit.

    ``AltFt`` and ``AltM`` are one quantity in two units, so each is defined
    once, in SI (``_si_values``, ``_si_expected``, and the SI inputs of its
    dynamic bounds), and a unit mixin (:class:`_InFeet`, :class:`_InKnots`, ...)
    sets ``_scale``, the SI-to-unit factor. Values, the test reference and
    dynamic bounds all convert through it, so the variants cannot drift apart.
    Bounds configured with ``low``/``high`` are already in the variant's unit.
    """

    _scale: ClassVar[float] = 1.0

    def _si_values(self, indices: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _si_expected(self, idx: int) -> float:
        raise NotImplementedError

    def _convert(self, si: Any) -> Any:
        return si * self._scale

    def _values(self, indices: Any) -> Any:
        return self._convert(self._si_values(_indices_array(indices)))

    def expected(self, idx: int) -> Any:
        return self._convert(self._si_expected(idx))


class _AltitudeEnvelopeBounds:
    """Bounds backed by BlueSky's aircraft altitude ceiling."""

    @staticmethod
    def _altitude_ceiling_m(idx: int) -> float:
        if bs.traf is None:
            raise RuntimeError(
                "dynamic altitude bounds require initialized BlueSky traffic; "
                "pass explicit low/high bounds for pre-initialization use."
            )
        return float(bs.traf.perf.hmax[idx])


class _CasEnvelopeBounds:
    """Bounds backed by BlueSky's CAS operating-speed envelope."""

    def _speed_bounds_ms(self, idx: int) -> tuple[float, float]:
        return bs.traf.perf.vmin[idx], bs.traf.perf.vmax[idx]


class _TasEnvelopeBounds:
    """Bounds backed by BlueSky's CAS envelope converted to TAS at altitude."""

    def _speed_bounds_ms(self, idx: int) -> tuple[float, float]:
        return (
            vcas2tas(bs.traf.perf.vmin[idx], bs.traf.alt[idx]),
            vcas2tas(bs.traf.perf.vmax[idx], bs.traf.alt[idx]),
        )


@dataclass(frozen=True)
class _Altitude(_AltitudeEnvelopeBounds, _UnitField):
    """Aircraft altitude; bounds from 0 to the performance ceiling."""

    low: Annotated[float | None, "lower bound; None = 0 at runtime"] = None
    high: Annotated[
        float | None, "upper bound; None = BlueSky perf.hmax at runtime"
    ] = None

    def _si_values(self, indices: np.ndarray) -> np.ndarray:
        return bs.traf.alt[indices]

    def _si_expected(self, idx: int) -> float:
        return float(bs.traf.alt[idx])

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(
            lambda: (0.0, self._convert(self._altitude_ceiling_m(idx)))
        )


@dataclass(frozen=True)
class _Speed(_UnitField):
    """A speed with bounds from the aircraft's envelope (``_speed_bounds_ms``)."""

    low: Annotated[float | None, "lower bound; None = runtime speed envelope"] = None
    high: Annotated[float | None, "upper bound; None = runtime speed envelope"] = None

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._dynamic_or_configured_bounds(
            lambda: tuple(self._convert(value) for value in self._speed_bounds_ms(idx))
        )


@dataclass(frozen=True)
class _Cas(_CasEnvelopeBounds, _Speed):
    def _si_values(self, indices: np.ndarray) -> np.ndarray:
        return bs.traf.cas[indices]

    def _si_expected(self, idx: int) -> float:
        return float(bs.traf.cas[idx])


@dataclass(frozen=True)
class _Tas(_TasEnvelopeBounds, _Speed):
    def _si_values(self, indices: np.ndarray) -> np.ndarray:
        return bs.traf.tas[indices]

    def _si_expected(self, idx: int) -> float:
        return float(bs.traf.tas[idx])


@dataclass(frozen=True)
class _Gs(_TasEnvelopeBounds, _Speed):
    def _si_values(self, indices: np.ndarray) -> np.ndarray:
        return bs.traf.gs[indices]

    def _si_expected(self, idx: int) -> float:
        return float(bs.traf.gs[idx])


@dataclass(frozen=True)
class _VerticalSpeed(_UnitField):
    """Vertical speed; bounds from the performance model's descent/climb limits."""

    low: Annotated[float | None, "lower bound; None = perf.vsmin at runtime"] = None
    high: Annotated[float | None, "upper bound; None = perf.vsmax at runtime"] = None

    def _si_values(self, indices: np.ndarray) -> np.ndarray:
        return bs.traf.vs[indices]

    def _si_expected(self, idx: int) -> float:
        return float(bs.traf.vs[idx])

    def bounds(self, idx: int) -> tuple[float, float]:
        def resolve() -> tuple[float, float]:
            vsmin = self._convert(float(bs.traf.perf.vsmin[idx]))
            vsmax = self._convert(float(bs.traf.perf.vsmax[idx]))
            assert vsmin <= 0.0
            return vsmin, vsmax

        return self._dynamic_or_configured_bounds(resolve)


@dataclass(frozen=True)
class AltFt(_InFeet, _Altitude):
    """Aircraft altitude in feet.

    ``low=None`` and ``high=None`` mean bounds are read from BlueSky's
    aircraft altitude ceiling at runtime.
    """

    meta = ObsMeta("alt_ft", Unit.FT, ObsQuantity.ALTITUDE, dynamic_bounds=True)


@dataclass(frozen=True)
class AltM(_InMeters, _Altitude):
    """Aircraft altitude in meters.

    ``low=None`` and ``high=None`` mean bounds are read from BlueSky's
    aircraft altitude ceiling at runtime.
    """

    meta = ObsMeta("alt_m", Unit.M, ObsQuantity.ALTITUDE, dynamic_bounds=True)


@dataclass(frozen=True)
class CasKts(_InKnots, _Cas):
    """Calibrated airspeed in knots.

    ``low=None`` and ``high=None`` mean bounds are read from BlueSky's
    current operating-speed envelope at runtime.
    """

    meta = ObsMeta("cas_kts", Unit.KTS, ObsQuantity.SPEED, dynamic_bounds=True)


@dataclass(frozen=True)
class CasMs(_InMetersPerSecond, _Cas):
    """Calibrated airspeed in m/s.

    ``low=None`` and ``high=None`` mean bounds are read from BlueSky's
    current operating-speed envelope at runtime.
    """

    meta = ObsMeta("cas_ms", Unit.M_PER_S, ObsQuantity.SPEED, dynamic_bounds=True)


@dataclass(frozen=True)
class TasKts(_InKnots, _Tas):
    """True airspeed in knots."""

    meta = ObsMeta("tas_kts", Unit.KTS, ObsQuantity.SPEED, dynamic_bounds=True)


@dataclass(frozen=True)
class TasMs(_InMetersPerSecond, _Tas):
    """True airspeed in m/s."""

    meta = ObsMeta("tas_ms", Unit.M_PER_S, ObsQuantity.SPEED, dynamic_bounds=True)


@dataclass(frozen=True)
class VsFtMin(_InFeetPerMinute, _VerticalSpeed):
    """Vertical speed in ft/min.

    ``low=None`` and ``high=None`` mean bounds are read from BlueSky's current
    aircraft performance envelope at runtime, as ``(vsmin, vsmax)``.
    """

    meta = ObsMeta(
        "vs_ftmin", Unit.FT_PER_MIN, ObsQuantity.VERTICAL_SPEED, dynamic_bounds=True
    )


@dataclass(frozen=True)
class VsMs(_InMetersPerSecond, _VerticalSpeed):
    """Vertical speed in m/s.

    ``low=None`` and ``high=None`` mean bounds are read from BlueSky's current
    aircraft performance envelope at runtime, as ``(vsmin, vsmax)`` .
    """

    meta = ObsMeta(
        "vs_ms", Unit.M_PER_S, ObsQuantity.VERTICAL_SPEED, dynamic_bounds=True
    )


@dataclass(frozen=True)
class AxMs2(_BroadcastObs, ObsField):
    """Longitudinal (TAS) acceleration in m/s^2 - the aircraft's current speed
    rate. ~0 when holding speed; a physical, orientation-invariant measure of
    speed-axis smoothness (pair with a rate-based action penalty).

    ``low``/``high`` are a **normalization scale**, not a physical cap: with a
    non-clipping normalizer, values beyond them pass through as ``|x|>1`` (no
    information lost). Acceleration has no clean *symmetric* physical bound (the
    thrust-limited accel differs from drag/idle decel), so the default is a fixed
    representative span that covers the typical range; override for a different
    scale.
    """

    meta = ObsMeta("ax_ms2", Unit.M_PER_S, ObsQuantity.SPEED)
    low: Annotated[float, "accel m/s^2 normalization scale (low)"] = -3.0
    high: Annotated[float, "accel m/s^2 normalization scale (high)"] = 3.0

    def _values(self, indices: Any) -> Any:
        return np.asarray(bs.traf.ax, dtype=np.float64)[_indices_array(indices)]

    def expected(self, idx: int) -> Any:
        return float(bs.traf.ax[idx])

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class MachNumber(_BroadcastObs, ObsField):
    """Ownship Mach number (``bs.traf.M``).

    At cruise altitude the speed envelope is Mach-limited, not CAS-limited, so
    Mach exposes the speed regime and remaining speed authority that ``CasKts``
    alone does not (two aircraft at equal CAS but different altitudes fly
    different TAS, hence different closing dynamics). Bounds default to the
    subsonic ``[0, 1]``; tighten ``high`` toward the type's Mmo (~0.87 for a
    B744) for a fuller normalized range.
    """

    meta = ObsMeta("mach_number", Unit.UNITLESS, ObsQuantity.SPEED)
    low: Annotated[float, "Mach normalization scale (low)"] = 0.0
    high: Annotated[float, "Mach normalization scale (high)"] = 1.0

    def _values(self, indices: Any) -> Any:
        return np.asarray(bs.traf.M, dtype=np.float64)[_indices_array(indices)]

    def expected(self, idx: int) -> Any:
        return float(bs.traf.M[idx])

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class CrossoverAltMarginFt(_BroadcastObs, ObsField):
    """Signed altitude margin to the CAS/Mach crossover, in feet.

    ``alt - crossoveralt(cas, Mmo)``: **positive above** the crossover (the
    Mach-limited regime, where a CAS increase is capped by Mmo), **negative
    below** (the CAS regime). Makes the speed action's regime boundary explicit so
    the policy need not infer it from Alt+CAS. Its *sign* is the "above crossover"
    boolean; the magnitude says how deep into the regime the aircraft is - a
    smoother, more informative signal than a bare flag.
    """

    meta = ObsMeta("crossover_alt_margin_ft", Unit.FT, ObsQuantity.ALTITUDE)
    low: Annotated[float, "crossover-margin ft normalization scale (low)"] = -20000.0
    high: Annotated[float, "crossover-margin ft normalization scale (high)"] = 20000.0
    crossover: Annotated[
        Crossover | None,
        "the speed schedule's crossover; None = each aircraft's CAS against its Mmo",
    ] = None

    def _values(self, indices: Any) -> Any:
        i = _indices_array(indices)
        alt = np.asarray(bs.traf.alt, dtype=np.float64)[i]
        if self.crossover is not None:
            return (alt - self.crossover.altitude_m) * _M_TO_FT
        cas = np.asarray(bs.traf.cas, dtype=np.float64)[i]
        mmo = mach_limit(i)
        limited = np.isfinite(mmo)
        # No Mach limit (a rotorcraft): never at a crossover - as far below as the scale goes.
        margin = (alt - np.asarray(crossoveralt(cas, np.where(limited, mmo, 0.8)))) * _M_TO_FT
        return np.where(limited, margin, -abs(float(self.low)) if self.low is not None else -1e5)

    def expected(self, idx: int) -> Any:
        if self.crossover is not None:
            return (float(bs.traf.alt[idx]) - self.crossover.altitude_m) / ft
        cas, mmo = float(bs.traf.cas[idx]), float(mach_limit(idx)[0])
        if not np.isfinite(mmo):
            return -abs(float(self.low)) if self.low is not None else -1e5
        return (float(bs.traf.alt[idx]) - float(crossoveralt(cas, mmo))) / ft

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class AboveCrossover(_BroadcastObs, ObsField):
    """Whether the aircraft is above the CAS/Mach crossover for the speed it is
    holding: 1 in the Mach regime - speed commanded and checked as Mach - 0 in
    the CAS regime. The same decision the crossover speed actions and waypoint
    speed constraints make (from the selected speed, not the current one), so
    the flag the policy sees is the regime its next speed command is in.
    ``CrossoverAltMarginFt`` says how deep into it.

    Given a speed schedule's ``crossover`` - the one a crossover speed action's
    :class:`~bluesky_sandbox.interface.fields.actions.MachRegime` has - it is
    whether the aircraft is above that altitude: the regime that action acts in."""

    meta = ObsMeta("above_crossover", Unit.SWITCH, ObsQuantity.SPEED)
    low: Annotated[float, "flag normalization scale (low)"] = 0.0
    high: Annotated[float, "flag normalization scale (high)"] = 1.0
    crossover: Annotated[
        Crossover | None,
        "the speed schedule's crossover; None = the speed held against Mmo",
    ] = None

    def _values(self, indices: Any) -> Any:
        return above_crossover(_indices_array(indices), crossover=self.crossover).astype(np.float64)

    def expected(self, idx: int) -> Any:
        return float(above_crossover(idx, crossover=self.crossover)[0])

    def bounds(self, idx: int) -> tuple[float, float]:
        return self._configured_bounds()


@dataclass(frozen=True)
class GsKts(_InKnots, _Gs):
    """Ground speed in knots.

    BlueSky has no separate ground-speed performance envelope. Default bounds
    use the TAS-equivalent operating-speed envelope; override constructor
    bounds if wind can push ground speed outside that range.
    """

    meta = ObsMeta("gs_kts", Unit.KTS, ObsQuantity.SPEED, dynamic_bounds=True)


@dataclass(frozen=True)
class GsMs(_InMetersPerSecond, _Gs):
    """Ground speed in m/s.

    BlueSky has no separate ground-speed performance envelope. Default bounds
    use the TAS-equivalent operating-speed envelope; override constructor
    bounds if wind can push ground speed outside that range.
    """

    meta = ObsMeta("gs_ms", Unit.M_PER_S, ObsQuantity.SPEED, dynamic_bounds=True)
