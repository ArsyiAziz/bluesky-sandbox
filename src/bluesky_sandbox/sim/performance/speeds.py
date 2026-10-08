"""Shared CAS/Mach crossover speed helpers.

Aircraft are controlled and separated in *calibrated airspeed* (CAS) at low
altitude and *Mach* above the CAS/Mach crossover altitude - the altitude where a
given CAS equals a given Mach. This module computes the regime and the
regime-relative speed error for a target CAS in one place, so the speed *action*
(:mod:`bluesky_sandbox.interface.fields.actions`), the waypoint *constraint*
(:mod:`bluesky_sandbox.sim.queryables`) and the speed-error *observation*
(:mod:`bluesky_sandbox.interface.fields.observations`) all switch regimes at the same
altitude and agree on what "on speed" means.

Depends only on ``bluesky`` (aircraft perf + aero conversions), so both the
``fields`` and ``queryables`` layers can import it without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import bluesky as bs
import numpy as np
from bluesky.tools.aero import casmach_thr, crossoveralt, ft, kts, vcas2mach, vcasormach, vmach2cas

_MS_TO_KTS = 1.0 / kts
# Symmetric-error scale floors, so a target pinned against an envelope edge still
# normalizes without dividing by ~0. Mach floor doubles as a sane default band.
_MIN_CAS_SCALE_KTS = 1.0
_MIN_MACH_SCALE = 0.02


@dataclass(frozen=True)
class CrossoverSpeedState:
    """Regime-aware speed state of one aircraft against a target CAS.

    ``in_mach`` is True above the CAS/Mach crossover altitude (control in Mach),
    False below (control in CAS). ``*_diff`` are current-minus-target in each
    regime; ``*_scale`` are symmetric normalizing scales (distance from the target
    to the feasible-envelope edge). Use :attr:`active_diff` / :attr:`active_scale`
    / :attr:`normalized_error` to work in whichever regime is currently active.
    """

    in_mach: bool
    target_ms: float       # feasible target CAS, m/s (clamped to [vmin, ceiling])
    target_mach: float     # target as Mach at current altitude (<= Mmo)
    cas_diff_kts: float     # current CAS - target CAS, kt
    mach_diff: float        # current Mach - target Mach
    cas_scale_kts: float
    mach_scale: float

    @property
    def active_diff(self) -> float:
        """Signed speed error in the active regime (Mach above crossover, else kt)."""
        return self.mach_diff if self.in_mach else self.cas_diff_kts

    @property
    def active_scale(self) -> float:
        """Symmetric normalizing scale for :attr:`active_diff`."""
        return self.mach_scale if self.in_mach else self.cas_scale_kts

    @property
    def normalized_error(self) -> float:
        """Signed active-regime error normalized to ``[-1, 1]`` (0 = on speed)."""
        x = self.active_diff / self.active_scale
        return -1.0 if x < -1.0 else min(x, 1.0)


@dataclass(frozen=True)
class Crossover:
    """A speed schedule's crossover: the altitude where ``cas_kts`` and
    ``mach`` are the same speed - below it the speed is flown as a CAS, above
    it as a Mach, the way a jet climbs at 300 kt then cruises at M0.78. The
    altitude depends on the schedule alone, not on the aircraft or its speed,
    so slowing down above it stays in Mach.

    Without one, the crossover is each aircraft's Mmo: Mach only where the CAS
    held would pass it (see :func:`above_crossover`)."""

    cas_kts: float = 300.0
    mach: float = 0.78
    #: How far past the crossover (ft) an aircraft changes regime: within it,
    #: it stays in the one it holds - a level-off near the crossover does not
    #: flip it back and forth.
    margin_ft: float = 300.0

    def __post_init__(self) -> None:
        if not float(self.cas_kts) > 0.0:
            raise ValueError(f"Crossover cas_kts must be > 0, got {self.cas_kts!r}")
        if not 0.0 < float(self.mach) < 1.0:
            raise ValueError(f"Crossover mach must be in (0, 1), got {self.mach!r}")
        if not float(self.margin_ft) >= 0.0:
            raise ValueError(f"Crossover margin_ft must be >= 0, got {self.margin_ft!r}")

    @property
    def altitude_m(self) -> float:
        """Where the schedule's CAS and Mach meet (m)."""
        return float(crossoveralt(float(self.cas_kts) * kts, float(self.mach)))

    def above(self, indices) -> np.ndarray:
        """Whether each aircraft in ``indices`` is in the Mach regime: past the
        crossover by more than ``margin_ft``, above; short of it by more,
        below; within, the regime of the speed it holds (a Mach, or a CAS)."""
        idx = np.atleast_1d(np.asarray(indices, dtype=np.intp))
        alt = np.asarray(bs.traf.alt, dtype=np.float64)[idx]
        margin = float(self.margin_ft) * ft
        holds_mach = is_mach(np.asarray(bs.traf.selspd, dtype=np.float64)[idx])
        regime = np.where(
            alt >= self.altitude_m + margin, True, np.where(alt <= self.altitude_m - margin, False, holds_mach)
        )
        return regime

    def handover(self, indices) -> list[tuple[int, float]]:
        """The speed holds to change for aircraft in ``indices`` that have
        passed the crossover, as an FMS does - and BlueSky does not: climbing,
        a CAS held becomes the Mach it is there; descending, a Mach held
        becomes its CAS - the same true airspeed, within the envelope. Each
        ``(index, speed)``, the speed as ``SPD`` takes it. Aircraft whose speed
        VNAV governs are left to it."""
        idx = np.atleast_1d(np.asarray(indices, dtype=np.intp))
        if idx.size == 0:
            return []
        traf = bs.traf
        alt = np.asarray(traf.alt, dtype=np.float64)[idx]
        selected = np.asarray(traf.selspd, dtype=np.float64)[idx]
        held = ~np.asarray(traf.swvnavspd, dtype=bool)[idx]
        mach = is_mach(selected)
        limit = np.asarray(traf.perf.mmo, dtype=np.float64)[idx]
        margin = float(self.margin_ft) * ft
        out: list[tuple[int, float]] = []
        for k, i in enumerate(idx):
            if not held[k] or selected[k] <= 0.0:
                continue
            if alt[k] >= self.altitude_m + margin and not mach[k]:
                value = min(float(vcas2mach(selected[k], alt[k])), float(limit[k]))
                out.append((int(i), round(value, 4)))
            elif alt[k] <= self.altitude_m - margin and mach[k]:
                cas = float(vmach2cas(selected[k], alt[k]))
                cas = min(max(cas, float(traf.perf.vmin[i])), float(traf.perf.vmax[i]))
                out.append((int(i), cas / kts))
        return out


def cas_ceiling_ms(idx: int) -> float:
    """Highest *feasible* CAS (m/s) at the aircraft's current altitude: the lower
    of the performance CAS limit and Mmo-expressed-as-CAS (which falls with
    altitude). Above the crossover this is the Mach limit."""
    alt = float(bs.traf.alt[idx])
    vmax = float(bs.traf.perf.vmax[idx])
    mmo = float(bs.traf.perf.mmo[idx])
    return min(vmax, float(vmach2cas(mmo, alt)))


def crossover_display(idx: int, cas_ms: float, alt_m: float) -> tuple[bool, float]:
    """Regime and Mach of a target CAS at an *arbitrary* altitude, for display.

    Returns ``(in_mach, mach)`` where ``in_mach`` is True when ``alt_m`` is above
    the CAS/Mach crossover altitude for this aircraft's Mmo - i.e. the target
    would be held in Mach there - and ``mach`` is the target's Mach at ``alt_m``
    (capped at Mmo). Mirrors the regime split in :func:`crossover_speed_state`,
    but evaluated at a supplied altitude (e.g. a route waypoint's) rather than the
    aircraft's current one, so a waypoint readout can show CAS below / Mach above.
    """
    mmo = float(bs.traf.perf.mmo[idx])
    in_mach = alt_m > float(crossoveralt(cas_ms, mmo))
    mach = min(float(vcas2mach(cas_ms, alt_m)), mmo)
    return in_mach, mach


#: How close to Mmo counts as at it. A target CAS clamped to the Mmo ceiling
#: is Mach-limited by definition; rounding must not decide its regime.
_AT_MMO = 1e-6


def _mach_regime(alt_m, target_cas_ms, mmo):
    """Whether a target CAS is held as Mach at ``alt_m``: above its crossover
    altitude against Mmo - put directly, the target is at or past Mmo there
    (the same rule, without the round trip through ``crossoveralt``, whose
    inverse is not exact)."""
    return np.asarray(vcas2mach(target_cas_ms, alt_m)) >= np.asarray(mmo) - _AT_MMO


def is_mach(speed) -> np.ndarray:
    """Whether each of ``speed`` is a Mach, as BlueSky reads a speed: above
    0.1 and below its CAS/Mach threshold (``casmach_thr``, m/s)."""
    speed = np.asarray(speed, dtype=np.float64)
    return (speed > 0.1) & (speed < casmach_thr)


def as_cas_ms(speed, alt_m) -> np.ndarray:
    """``speed`` - a CAS (m/s) or a Mach, told apart as BlueSky does - as a
    CAS (m/s) at ``alt_m``: BlueSky's own ``vcasormach``."""
    return np.asarray(vcasormach(np.asarray(speed, dtype=np.float64), np.asarray(alt_m, dtype=np.float64))[1])


def selected_cas_ms(indices) -> np.ndarray:
    """The autopilot's selected speed, as CAS (m/s), for each aircraft in
    ``indices``. After a Mach ``SPD`` - above the crossover - BlueSky holds the
    Mach number there, not a CAS: converted at the aircraft's altitude."""
    idx = np.atleast_1d(np.asarray(indices, dtype=np.intp))
    alt = np.asarray(bs.traf.alt, dtype=np.float64)[idx]
    return as_cas_ms(np.asarray(bs.traf.selspd, dtype=np.float64)[idx], alt)


def above_crossover(indices, target_cas_ms=None, crossover: Crossover | None = None) -> np.ndarray:
    """Whether each aircraft in ``indices`` is in the Mach regime: above
    ``crossover``'s altitude where one is given; otherwise for its target CAS
    (m/s; default its selected speed), the decision the crossover speed command
    makes (:func:`crossover_speed_state`) - target clamped to the feasible
    envelope, above ``crossoveralt(target, Mmo)``."""
    if crossover is not None:
        return crossover.above(indices)
    idx = np.atleast_1d(np.asarray(indices, dtype=np.intp))
    target = selected_cas_ms(idx) if target_cas_ms is None else np.broadcast_to(
        np.asarray(target_cas_ms, dtype=np.float64), idx.shape
    )
    alt = np.asarray(bs.traf.alt, dtype=np.float64)[idx]
    mmo = np.asarray(bs.traf.perf.mmo, dtype=np.float64)[idx]
    vmin = np.asarray(bs.traf.perf.vmin, dtype=np.float64)[idx]
    vmax = np.asarray(bs.traf.perf.vmax, dtype=np.float64)[idx]
    ceiling = np.minimum(vmax, vmach2cas(mmo, alt))
    target = np.minimum(np.maximum(target, vmin), ceiling)
    return _mach_regime(alt, target, mmo)


def crossover_speed_state(idx: int, target_cas_ms: float) -> CrossoverSpeedState:
    """Regime-aware speed state for a target CAS (m/s), mirroring the crossover
    speed command: clamp the target to the feasible envelope ``[vmin, ceiling]``,
    then the aircraft is in the Mach regime above ``crossoveralt(target, Mmo)``
    (holding the target as Mach, capped at Mmo) and the CAS regime below.
    """
    alt = float(bs.traf.alt[idx])
    mmo = float(bs.traf.perf.mmo[idx])
    vmin = float(bs.traf.perf.vmin[idx])
    vmax = float(bs.traf.perf.vmax[idx])
    target_ms = min(max(float(target_cas_ms), vmin), cas_ceiling_ms(idx))
    target_mach = min(float(vcas2mach(target_ms, alt)), mmo)
    in_mach = bool(_mach_regime(alt, target_ms, mmo))

    cas_diff_kts = (float(bs.traf.cas[idx]) - target_ms) * _MS_TO_KTS
    mach_diff = float(bs.traf.M[idx]) - target_mach

    cas_scale_kts = max(
        abs(target_ms - vmin), abs(vmax - target_ms)
    ) * _MS_TO_KTS
    cas_scale_kts = max(cas_scale_kts, _MIN_CAS_SCALE_KTS)
    mach_min = float(vcas2mach(vmin, alt))
    mach_scale = max(
        abs(target_mach - mach_min), abs(mmo - target_mach), _MIN_MACH_SCALE
    )
    return CrossoverSpeedState(
        in_mach=in_mach,
        target_ms=target_ms,
        target_mach=target_mach,
        cas_diff_kts=cas_diff_kts,
        mach_diff=mach_diff,
        cas_scale_kts=cas_scale_kts,
        mach_scale=mach_scale,
    )


def cas_tolerance_as_mach(idx: int, target_cas_ms: float, tolerance_kts: float) -> float:
    """Symmetric Mach tolerance equivalent to a CAS ``tolerance_kts`` band around
    ``target_cas_ms`` at the aircraft's current altitude. Lets a single CAS
    tolerance stay well-defined above the CAS/Mach crossover, where the speed
    constraint is evaluated in Mach - the regime the aircraft is controlled in."""
    alt = float(bs.traf.alt[idx])
    tol_ms = float(tolerance_kts) * kts
    hi = float(vcas2mach(target_cas_ms + tol_ms, alt))
    lo = float(vcas2mach(max(target_cas_ms - tol_ms, 0.0), alt))
    return abs(hi - lo) / 2.0


def within_speed_tolerance(
    idx: int,
    target_cas_ms: float,
    tolerance_kts: float | None,
    tolerance_mach: float | None = None,
) -> bool:
    """Whether the aircraft meets a waypoint speed tolerance, regime-aware.

    Below the CAS/Mach crossover the CAS tolerance (``tolerance_kts``) binds;
    above it the Mach tolerance binds - the explicit ``tolerance_mach`` when
    given, otherwise one derived from ``tolerance_kts`` via
    :func:`cas_tolerance_as_mach` - so the check is well-defined for *any*
    sampled target speed/altitude. A ``None`` tolerance in the active regime
    leaves the speed axis unconstrained (returns ``True``). The target CAS is
    clamped to the feasible envelope by :func:`crossover_speed_state`, so an
    out-of-envelope sampled target still yields a satisfiable band.
    """
    state = crossover_speed_state(idx, target_cas_ms)
    if state.in_mach:
        tol = tolerance_mach
        if tol is None and tolerance_kts is not None:
            tol = cas_tolerance_as_mach(idx, target_cas_ms, tolerance_kts)
        return tol is None or abs(state.mach_diff) <= tol
    return tolerance_kts is None or abs(state.cas_diff_kts) <= float(tolerance_kts)


def within_speed_tolerance_many(
    n: int,
    target_cas_ms: np.ndarray,
    tolerance_kts: float | None,
    tolerance_mach: float | None = None,
) -> np.ndarray:
    """Vectorized :func:`within_speed_tolerance` over the first ``n`` traf rows.

    ``target_cas_ms`` is a per-aircraft target CAS array (m/s); a non-finite
    entry means that aircraft's speed axis is unconstrained (``True``), matching
    the scalar path's treatment of a missing target. Element-wise identical to
    the scalar function - same clamped-target regime split, the same raw-target
    CAS->Mach tolerance band - but one numpy pass instead of a per-aircraft
    Python loop, which is what makes per-substep dwell tracking affordable.
    """
    ok = np.ones(n, dtype=bool)
    if n == 0 or (tolerance_kts is None and tolerance_mach is None):
        return ok
    target = np.asarray(target_cas_ms, dtype=np.float64)[:n]
    have = np.isfinite(target)
    if not have.any():
        return ok

    alt = np.asarray(bs.traf.alt, dtype=np.float64)[:n]
    cas = np.asarray(bs.traf.cas, dtype=np.float64)[:n]
    mach = np.asarray(bs.traf.M, dtype=np.float64)[:n]
    vmin = np.asarray(bs.traf.perf.vmin, dtype=np.float64)[:n]
    vmax = np.asarray(bs.traf.perf.vmax, dtype=np.float64)[:n]
    mmo = np.asarray(bs.traf.perf.mmo, dtype=np.float64)[:n]

    # Substitute a finite placeholder on unconstrained rows so the aero
    # conversions stay warning-free; those rows are masked back to True below.
    raw_target = np.where(have, target, cas)
    ceiling = np.minimum(vmax, vmach2cas(mmo, alt))
    tgt = np.clip(raw_target, vmin, ceiling)
    in_mach = alt > crossoveralt(tgt, mmo)

    cas_diff_kts = (cas - tgt) * _MS_TO_KTS
    target_mach = np.minimum(vcas2mach(tgt, alt), mmo)
    mach_diff = mach - target_mach

    if tolerance_mach is not None:
        satisfied_mach = np.abs(mach_diff) <= float(tolerance_mach)
    elif tolerance_kts is not None:
        # As in the scalar path, the derived Mach band brackets the *raw*
        # (unclamped) target CAS.
        tol_ms = float(tolerance_kts) * kts
        hi = vcas2mach(raw_target + tol_ms, alt)
        lo = vcas2mach(np.maximum(raw_target - tol_ms, 0.0), alt)
        satisfied_mach = np.abs(mach_diff) <= np.abs(hi - lo) / 2.0
    else:
        satisfied_mach = np.ones(n, dtype=bool)
    if tolerance_kts is not None:
        satisfied_cas = np.abs(cas_diff_kts) <= float(tolerance_kts)
    else:
        satisfied_cas = np.ones(n, dtype=bool)
    return np.where(have, np.where(in_mach, satisfied_mach, satisfied_cas), True)
