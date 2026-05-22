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
from bluesky.tools.aero import crossoveralt, kts, vcas2mach, vmach2cas

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
    in_mach = alt > float(crossoveralt(target_ms, mmo))

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
